// 多线程、向量化较困难 存在数据依赖
#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void cholesky_kernel(int n, int barrier_id, double *A)
{
    int tid = get_thread_id();

    // Only one thread performs all computations
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double sum = A[i*n + j];
                for (int k = 0; k < j; k++) {
                    sum -= A[i*n + k] * A[j*n + k];
                }
                A[i*n + j] = sum / A[j*n + j];
            }

            double sum = A[i*n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[i*n + k] * A[i*n + k];
            }
            A[i*n + i] = sqrt(sum);
        }
    }
}
__global__ void cholesky_kernel_thread(int n, int barrier_id, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            // 指定某个线程去算这个 j
            if (tid == (j % num_threads)) {
                double sum = A[i*n + j];
                for (int k = 0; k < j; k++) {
                    sum -= A[i*n + k] * A[j*n + k];
                }
                A[i*n + j] = sum / A[j*n + j];
            }
            // 等待该 j 完成，确保 A[i, j] 更新后下一轮可用
            group_barrier(barrier_id);
        }

        // 只有一个线程更新对角元素
        if (tid == 0) {
            double sum = A[i*n + i];
            for (int k = 0; k < i; k++) {
                sum -= A[i*n + k] * A[i*n + k];
            }
            A[i*n + i] = sqrt(sum);
        }
        group_barrier(barrier_id);
    }
}

#define SIMD_LEN    16
#define VEC_BYTES   128
__global__ void cholesky_kernel_vec(int n, int barrier_id, double *A)
{
    int tid = get_thread_id();
    if (tid != 0) return; // 串行跑保证和CPU一致
    
    lvector double *buf_ai = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_aj = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_mul= (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_ai || !buf_aj || !buf_mul) {
        if (buf_ai) vector_free(buf_ai);
        if (buf_aj) vector_free(buf_aj);
        if (buf_mul) vector_free(buf_mul);
        return;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            double sum = A[i*n + j];
            int k = 0;
            for (; k + SIMD_LEN <= j; k += SIMD_LEN) {
                vector_load(&A[i*n + k], buf_ai, VEC_BYTES);
                vector_load(&A[j*n + k], buf_aj, VEC_BYTES);

                lvector double vai = vec_ld(0, buf_ai);
                lvector double vaj = vec_ld(0, buf_aj);
                lvector double vmul = vec_muli(vai, vaj);

                // 存到 vector_malloc 的缓冲区
                vec_st(vmul, 0, buf_mul);
                double partial_sum = vsip_vsumval_d_v(buf_mul, SIMD_LEN);

                sum -= partial_sum;
            }
            // 尾部标量
            for (; k < j; k++) {
                sum -= A[i*n + k] * A[j*n + k];
            }
            A[i*n + j] = sum / A[j*n + j];
        }

        // 对角元素
        double sum = A[i*n + i];
        int k = 0;
        for (; k + SIMD_LEN <= i; k += SIMD_LEN) {
            vector_load(&A[i*n + k], buf_ai, VEC_BYTES);
            lvector double vai = vec_ld(0, buf_ai);
            lvector double vsq = vec_muli(vai, vai);

            vec_st(vsq, 0, buf_mul);
            double partial_sum = vsip_vsumval_d_v(buf_mul, SIMD_LEN);

            sum -= partial_sum;
        }
        for (; k < i; k++) {
            sum -= A[i*n + k] * A[i*n + k];
        }
        A[i*n + i] = sqrt(sum);
    }

    vector_free(buf_ai);
    vector_free(buf_aj);
    vector_free(buf_mul);
}

#include "../CHOLESKY/kernel_vec.h"
#include "../CHOLESKY/kernel_cache_llm.h"//SM缓存优化文件