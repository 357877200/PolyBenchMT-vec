#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void jacobi2D_kernel1(int n, double* A, double* B) 
{
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    
    // 计算每个线程需要处理的行数
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);
    
    // 每个线程处理其负责的行
    for(int i = start_row; i < end_row; i++) {
        for(int j = 1; j < n-1; j++) {
            B[i*n + j] = 0.2f * (A[i*n + j] + 
                                A[i*n + (j-1)] + 
                                A[i*n + (j+1)] + 
                                A[(i+1)*n + j] + 
                                A[(i-1)*n + j]);
        }
    }
}

__global__ void jacobi2D_kernel2(int n, double* A, double* B) 
{
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    
    // 计算每个线程需要处理的行数
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);
    
    // 每个线程处理其负责的行
    for(int i = start_row; i < end_row; i++) {
        for(int j = 1; j < n-1; j++) {
            A[i*n + j] = B[i*n + j];
        }
    }
}
#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void jacobi2D_kernel1_vec(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    /* 检查线程安全性 */
    if (total_threads > 24) return; /* 防止 __gsm__ 数组溢出 */

    /* 计算每个线程需要处理的行数 */
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);

    /* 广播系数 0.2 为向量 */
    lvector double v_coeff = (lvector double)vec_svbcast(0.2);
    lvector double v_one = (lvector double)vec_svbcast(1.0);

    /* 分配向量缓冲区：中心、左、右、上、下五个向量 */
    lvector double *buf = (lvector double *)vector_malloc(sizeof(lvector double) * 5);
    lvector double *v_center = buf + 0; /* A[i*n + j:j+SIMD_LEN] */
    lvector double *v_left = buf + 1;   /* A[i*n + (j-1):j-1+SIMD_LEN] */
    lvector double *v_right = buf + 2;  /* A[i*n + (j+1):j+1+SIMD_LEN] */
    lvector double *v_up = buf + 3;     /* A[(i-1)*n + j:j+SIMD_LEN] */
    lvector double *v_down = buf + 4;   /* A[(i+1)*n + j:j+SIMD_LEN] */

    /* 处理每个线程负责的行 */
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < n - 1; ) {
            int remain_elements = n - 1 - j;

            if (remain_elements >= SIMD_LEN) {
                /* 向量化处理 SIMD_LEN 个元素 */
                size_t off_center = i * n + j;
                size_t off_left = i * n + (j - 1);
                size_t off_right = i * n + (j + 1);
                size_t off_up = (i - 1) * n + j;
                size_t off_down = (i + 1) * n + j;

                /* 加载 A[i*n + j:j+SIMD_LEN], A[i*n + (j-1):j-1+SIMD_LEN], ... */
                vector_load(A + off_center, v_center, VEC_BYTES);
                vector_load(A + off_left, v_left, VEC_BYTES);
                vector_load(A + off_right, v_right, VEC_BYTES);
                vector_load(A + off_up, v_up, VEC_BYTES);
                vector_load(A + off_down, v_down, VEC_BYTES);

                /* 计算 B[i*n + j:j+SIMD_LEN] = 0.2 * (v_center + v_left + v_right + v_up + v_down) */
                lvector double v_sum = (lvector double)vec_svbcast(0.0);
                v_sum = vec_mula(*v_center, v_one, v_sum); /* v_sum += v_center */
                v_sum = vec_mula(*v_left, v_one, v_sum);   /* v_sum += v_left */
                v_sum = vec_mula(*v_right, v_one, v_sum);  /* v_sum += v_right */
                v_sum = vec_mula(*v_up, v_one, v_sum);     /* v_sum += v_up */
                v_sum = vec_mula(*v_down, v_one, v_sum);   /* v_sum += v_down */
                v_sum = vec_muli(v_coeff, v_sum);          /* v_sum *= 0.2 */

                /* 写回 B[i*n + j:j+SIMD_LEN] */
                vector_store(&v_sum, B + i * n + j, VEC_BYTES);

                j += SIMD_LEN; /* 跳过 SIMD_LEN 个元素 */
            } else {
                /* 标量尾部处理 */
                for (; j < n - 1; j++) {
                    B[i * n + j] = 0.2 * (A[i * n + j] +
                                          A[i * n + (j - 1)] +
                                          A[i * n + (j + 1)] +
                                          A[(i - 1) * n + j] +
                                          A[(i + 1) * n + j]);
                }
            }
        }
    }

    vector_free(buf);
}
// 使用vsip库提供的加法
__global__ void jacobi2D_kernel1_vec_vsip(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    /* 检查线程安全性 */
    if (total_threads > 24) return; /* 防止 __gsm__ 数组溢出 */

    /* 计算每个线程需要处理的行数 */
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);

    /* 广播系数 0.2 为向量 */
    lvector double v_coeff = (lvector double)vec_svbcast(0.2);

    /* 分配向量缓冲区：中心、左、右、上、下五个向量 */
    lvector double *buf = (lvector double *)vector_malloc(sizeof(lvector double) * 5);
    lvector double *v_center = buf + 0; /* A[i*n + j:j+SIMD_LEN] */
    lvector double *v_left = buf + 1;   /* A[i*n + (j-1):j-1+SIMD_LEN] */
    lvector double *v_right = buf + 2;  /* A[i*n + (j+1):j+1+SIMD_LEN] */
    lvector double *v_up = buf + 3;     /* A[(i-1)*n + j:j+SIMD_LEN] */
    lvector double *v_down = buf + 4;   /* A[(i+1)*n + j:j+SIMD_LEN] */

    lvector double * v_sum1 = vector_malloc(sizeof(lvector double));

    /* 处理每个线程负责的行 */
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < n - 1; ) {
            int remain_elements = n - 1 - j;

            if (remain_elements >= SIMD_LEN) {
                /* 向量化处理 SIMD_LEN 个元素 */
                size_t off_center = i * n + j;
                size_t off_left = i * n + (j - 1);
                size_t off_right = i * n + (j + 1);
                size_t off_up = (i - 1) * n + j;
                size_t off_down = (i + 1) * n + j;

                /* 加载 A[i*n + j:j+SIMD_LEN], A[i*n + (j-1):j-1+SIMD_LEN], ... */
                vector_load(A + off_center, v_center, VEC_BYTES);
                vector_load(A + off_left, v_left, VEC_BYTES);
                vector_load(A + off_right, v_right, VEC_BYTES);
                vector_load(A + off_up, v_up, VEC_BYTES);
                vector_load(A + off_down, v_down, VEC_BYTES);

                /* 计算 B[i*n + j:j+SIMD_LEN] = 0.2 * (v_center + v_left + v_right + v_up + v_down) */
                vec_st((lvector double)vec_svbcast(0.0), 0, v_sum1);
                vsip_vadd_d_v(v_sum1, v_center, v_sum1, 16);
                vsip_vadd_d_v(v_sum1, v_left, v_sum1, 16);
                vsip_vadd_d_v(v_sum1, v_right, v_sum1, 16);
                vsip_vadd_d_v(v_sum1, v_up, v_sum1, 16);
                vsip_vadd_d_v(v_sum1, v_down, v_sum1, 16);
                lvector double v_sum = vec_muli(v_coeff, *v_sum1);          /* v_sum *= 0.2 */

                /* 写回 B[i*n + j:j+SIMD_LEN] */
                vector_store(&v_sum, B + i * n + j, VEC_BYTES);

                j += SIMD_LEN; /* 跳过 SIMD_LEN 个元素 */
            } else {
                /* 标量尾部处理 */
                for (; j < n - 1; j++) {
                    B[i * n + j] = 0.2 * (A[i * n + j] +
                                          A[i * n + (j - 1)] +
                                          A[i * n + (j + 1)] +
                                          A[(i - 1) * n + j] +
                                          A[(i + 1) * n + j]);
                }
            }
        }
    }

    vector_free(buf);
    vector_free(v_sum1);
}
__global__ void jacobi2D_kernel2_vec(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    /* 检查线程安全性 */
    if (total_threads > 24) return; /* 防止 __gsm__ 数组溢出 */

    /* 计算每个线程需要处理的行数 */
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);

    /* 分配向量缓冲区 */
    lvector double *v_buf = (lvector double *)vector_malloc(sizeof(lvector double));

    /* 处理每个线程负责的行 */
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < n - 1; ) {
            int remain_elements = n - 1 - j;

            if (remain_elements >= SIMD_LEN) {
                /* 向量化复制 SIMD_LEN 个元素 */
                vector_load(B + i * n + j, v_buf, VEC_BYTES);
                vector_store(v_buf, A + i * n + j, VEC_BYTES);

                j += SIMD_LEN; /* 跳过 SIMD_LEN 个元素 */
            } else {
                /* 标量尾部处理 */
                for (; j < n - 1; j++) {
                    A[i * n + j] = B[i * n + j];
                }
            }
        }
    }

    vector_free(v_buf);
}
#include "../JACOBI2D/kernel_vec.h"//大模型生成的存储文件
#include "../JACOBI2D/kernel_cache_llm.h"//SM缓存优化文件