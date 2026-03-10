#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void trmm_kernel(int m, int n, double alpha, double *A, double *B)
{
    int tid         = get_thread_id();
    int num_threads = get_group_size();

    // 列并行分配
    int work_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * work_per_thread;
    int end_j   = min(start_j + work_per_thread, n);

    // 保持 i 循环顺序
    for (int i = 0; i < m; i++) {
        for (int j = start_j; j < end_j; j++) {
            for (int k = i + 1; k < m; k++) {
                B[i * n + j] += A[k * m + i] * B[k * n + j];
            }
            B[i * n + j] = alpha * B[i * n + j];
        }
    }
}

#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double temp_b[24][SIMD_LEN];

__global__ void trmm_kernel_vec(int m, int n, double alpha, double *A, double *B)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int tid_mod = thread_id % 24;

    // 分配向量缓冲区
    lvector double *buf_bkj = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_bij = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_bkj || !buf_bij) {
        if (buf_bkj) vector_free(buf_bkj);
        if (buf_bij) vector_free(buf_bij);
        return;
    }

    // 改成按列并行分配
    int work_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = thread_id * work_per_thread;
    int end_j   = min(start_j + work_per_thread, n);

    // 保证 i 循环按顺序执行
    for (int i = 0; i < m; i++) {
        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, end_j);

            lvector double sum_vec = (lvector double)vec_svbcast(0.0);
            double sum_tail[SIMD_LEN] = {0.0};

            for (int k = i + 1; k < m; k++) {
                double a_val = A[k * m + i]; // 标量因子

                if (j + SIMD_LEN <= end_j) {
                    // 加载 B[k][j:j+SIMD_LEN]
                    vector_load(&B[k * n + j], buf_bkj, VEC_BYTES);
                    lvector double b_kj = vec_ld(0, buf_bkj);

                    lvector double a_vec = (lvector double)vec_svbcast(a_val);
                    lvector double tmp = vec_muli(a_vec, b_kj);

                    sum_vec = vec_mula(tmp, (lvector double)vec_svbcast(1.0), sum_vec);
                } else {
                    // 尾部标量部分（不足 SIMD_LEN）
                    for (int jj = j; jj < vec_end_j; ++jj) {
                        sum_tail[jj - j] += a_val * B[k * n + jj];
                    }
                }
            }

            // 累加到结果并乘以 alpha
            if (j + SIMD_LEN <= end_j) {
                vec_st(sum_vec, 0, buf_bij);
                vector_store(buf_bij, temp_b[tid_mod], VEC_BYTES);
                for (int jj = 0; jj < SIMD_LEN; ++jj) {
                    B[i * n + j + jj] = alpha * (B[i * n + j + jj] + temp_b[tid_mod][jj]);
                }
            } else {
                // 尾部标量部分
                for (int jj = j; jj < vec_end_j; ++jj) {
                    B[i * n + jj] = alpha * (B[i * n + jj] + sum_tail[jj - j]);
                }
            }
        }
    }

    vector_free(buf_bkj);
    vector_free(buf_bij);
}

#include "../TRMM/kernel_vec.h"//大模型生成的存储文件
#include "../TRMM/kernel_cache_llm.h"//SM缓存优化文件