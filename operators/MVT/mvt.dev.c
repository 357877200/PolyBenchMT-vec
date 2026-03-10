#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void mvt_kernel1(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            x1[i] += a[i * n + j] * y_1[j];
        }
    }
}

__global__ void mvt_kernel2(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            x2[i] += a[j * n + i] * y_2[j];
        }
    }
}

#ifdef MINI_DATASET
__global__ void mvt_kernel1_cache(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x1, double, &x1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_1, double, y_1, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x1_i, tmp_y_1_j, tmp_a_ij;
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x1, &x1[i - start_idx], tmp_x1_i);
        for (int j = 0; j < n; j++) {
            CACHEb_RD(y_1, &y_1[j], tmp_y_1_j);
            CACHEs_RD(a, &a[i * n + j], tmp_a_ij);
            tmp_x1_i += tmp_a_ij * tmp_y_1_j;
        }
        CACHEb_WT(x1, &x1[i - start_idx], tmp_x1_i);
    }
    CACHEb_FLUSH(x1);
    CACHEb_INVALID(y_1);
    CACHEs_INVALID(a);
}

__global__ void mvt_kernel2_cache(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x2, double, &x2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_2, double, y_2, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x2_i, tmp_y_2_j, tmp_a_ji;
    for (int j = 0; j < n; j++) {
        CACHEb_RD(y_2, &y_2[j], tmp_y_2_j);
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x2, &x2[i - start_idx], tmp_x2_i);
            CACHEs_RD(a, &a[j * n + i], tmp_a_ji);
            tmp_x2_i += tmp_a_ji * tmp_y_2_j;
            CACHEb_WT(x2, &x2[i - start_idx], tmp_x2_i);
        }
    }
    CACHEb_FLUSH(x2);
    CACHEb_INVALID(y_2);
    CACHEs_INVALID(a);
}
#endif

#ifdef SMALL_DATASET
__global__ void mvt_kernel1_cache(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x1, double, &x1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_1, double, y_1, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x1_i, tmp_y_1_j, tmp_a_ij;
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x1, &x1[i - start_idx], tmp_x1_i);
        for (int j = 0; j < n; j++) {
            CACHEb_RD(y_1, &y_1[j], tmp_y_1_j);
            CACHEs_RD(a, &a[i * n + j], tmp_a_ij);
            tmp_x1_i += tmp_a_ij * tmp_y_1_j;
        }
        CACHEb_WT(x1, &x1[i - start_idx], tmp_x1_i);
    }
    CACHEb_FLUSH(x1);
    CACHEb_INVALID(y_1);
    CACHEs_INVALID(a);
}

__global__ void mvt_kernel2_cache(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x2, double, &x2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_2, double, y_2, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x2_i, tmp_y_2_j, tmp_a_ji;
    for (int j = 0; j < n; j++) {
        CACHEb_RD(y_2, &y_2[j], tmp_y_2_j);
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x2, &x2[i - start_idx], tmp_x2_i);
            CACHEs_RD(a, &a[j * n + i], tmp_a_ji);
            tmp_x2_i += tmp_a_ji * tmp_y_2_j;
            CACHEb_WT(x2, &x2[i - start_idx], tmp_x2_i);
        }
    }
    CACHEb_FLUSH(x2);
    CACHEb_INVALID(y_2);
    CACHEs_INVALID(a);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void mvt_kernel1_cache(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x1, double, &x1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_1, double, y_1, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x1_i, tmp_y_1_j, tmp_a_ij;
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x1, &x1[i - start_idx], tmp_x1_i);
        for (int j = 0; j < n; j++) {
            CACHEb_RD(y_1, &y_1[j], tmp_y_1_j);
            CACHEs_RD(a, &a[i * n + j], tmp_a_ij);
            tmp_x1_i += tmp_a_ij * tmp_y_1_j;
        }
        CACHEb_WT(x1, &x1[i - start_idx], tmp_x1_i);
    }
    CACHEb_FLUSH(x1);
    CACHEb_INVALID(y_1);
    CACHEs_INVALID(a);
}

__global__ void mvt_kernel2_cache(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x2, double, &x2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_2, double, y_2, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x2_i, tmp_y_2_j, tmp_a_ji;
    for (int j = 0; j < n; j++) {
        CACHEb_RD(y_2, &y_2[j], tmp_y_2_j);
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x2, &x2[i - start_idx], tmp_x2_i);
            CACHEs_RD(a, &a[j * n + i], tmp_a_ji);
            tmp_x2_i += tmp_a_ji * tmp_y_2_j;
            CACHEb_WT(x2, &x2[i - start_idx], tmp_x2_i);
        }
    }
    CACHEb_FLUSH(x2);
    CACHEb_INVALID(y_2);
    CACHEs_INVALID(a);
}
#endif

#ifdef LARGE_DATASET
__global__ void mvt_kernel1_cache(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x1, double, &x1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_1, double, y_1, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 14);
    double tmp_x1_i, tmp_y_1_j, tmp_a_ij;
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x1, &x1[i - start_idx], tmp_x1_i);
        for (int j = 0; j < n; j++) {
            CACHEb_RD(y_1, &y_1[j], tmp_y_1_j);
            CACHEs_RD(a, &a[i * n + j], tmp_a_ij);
            tmp_x1_i += tmp_a_ij * tmp_y_1_j;
        }
        CACHEb_WT(x1, &x1[i - start_idx], tmp_x1_i);
    }
    CACHEb_FLUSH(x1);
    CACHEb_INVALID(y_1);
    CACHEs_INVALID(a);
}

__global__ void mvt_kernel2_cache(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x2, double, &x2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y_2, double, y_2, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 14);
    double tmp_x2_i, tmp_y_2_j, tmp_a_ji;
    for (int j = 0; j < n; j++) {
        CACHEb_RD(y_2, &y_2[j], tmp_y_2_j);
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x2, &x2[i - start_idx], tmp_x2_i);
            CACHEs_RD(a, &a[j * n + i], tmp_a_ji);
            tmp_x2_i += tmp_a_ji * tmp_y_2_j;
            CACHEb_WT(x2, &x2[i - start_idx], tmp_x2_i);
        }
    }
    CACHEb_FLUSH(x2);
    CACHEb_INVALID(y_2);
    CACHEs_INVALID(a);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void mvt_kernel1_cache(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEs_INIT(x1, double, x1, 0, 8);
    CACHEs_INIT(y_1, double, y_1, 0, 15);
    CACHEs_INIT(a, double, a, 0, 6);
    double tmp_x1_i, tmp_y_1_j, tmp_a_ij;
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEs_RD(x1, &x1[i], tmp_x1_i);
        for (int j = 0; j < n; j++) {
            CACHEs_RD(y_1, &y_1[j], tmp_y_1_j);
            CACHEs_RD(a, &a[i * n + j], tmp_a_ij);
            tmp_x1_i += tmp_a_ij * tmp_y_1_j;
        }
        CACHEs_WT(x1, &x1[i], tmp_x1_i);
    }
    CACHEs_FLUSH(x1);
    CACHEs_INVALID(y_1);
    CACHEs_INVALID(a);
}

__global__ void mvt_kernel2_cache(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(x2, double, &x2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(y_2, double, y_2, 0, 15);
    CACHEs_INIT(a, double, a, 0, 12);
    double tmp_x2_i, tmp_y_2_j, tmp_a_ji;
    for (int j = 0; j < n; j++) {
        CACHEs_RD(y_2, &y_2[j], tmp_y_2_j);
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x2, &x2[i - start_idx], tmp_x2_i);
            CACHEs_RD(a, &a[j * n + i], tmp_a_ji);
            tmp_x2_i += tmp_a_ji * tmp_y_2_j;
            CACHEb_WT(x2, &x2[i - start_idx], tmp_x2_i);
        }
    }
    CACHEb_FLUSH(x2);
    CACHEs_INVALID(y_2);
    CACHEs_INVALID(a);
}
#endif

#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double temp_x[24][SIMD_LEN];

// 向量化版本的 mvt_kernel1
__global__ void mvt_kernel1_vec(int n, double *a, double *x1, double *y_1)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int tid_mod = thread_id % 24;

    // 分配向量缓冲区
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_temp = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_a || !buf_y || !buf_temp) {
        if (buf_a) vector_free(buf_a);
        if (buf_y) vector_free(buf_y);
        if (buf_temp) vector_free(buf_temp);
        return;
    }

    // 计算任务分配
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (thread_id < remainder) ? thread_id * (elements_per_thread + 1)
                                            : remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (thread_id < remainder ? 1 : 0);

    for (int i = start_idx; i < end_idx; ++i) {
        double sum_scalar = 0.0;
        lvector double sum_vec = (lvector double)vec_svbcast(0.0);

        // 向量化内层循环 j
        for (int j = 0; j < n; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, n);

            if (j + SIMD_LEN <= n) {
                // 加载 a[i][j:j+16] 和 y_1[j:j+16]
                vector_load(&a[i * n + j], buf_a, VEC_BYTES);
                vector_load(&y_1[j], buf_y, VEC_BYTES);

                lvector double a_ij = vec_ld(0, buf_a);
                lvector double y_j = vec_ld(0, buf_y);

                // a[i][j] * y_1[j]
                lvector double tmp = vec_muli(a_ij, y_j);
                sum_vec = vec_mula(tmp, (lvector double)vec_svbcast(1.0), sum_vec);
            } else {
                // 标量尾部处理
                for (int jj = j; jj < vec_end_j; ++jj) {
                    sum_scalar += a[i * n + jj] * y_1[jj];
                }
            }
        }

        // 将向量结果写回临时缓冲区，再累加到标量
        vec_st(sum_vec, 0, buf_temp);
        vector_store(buf_temp, temp_x[tid_mod], VEC_BYTES);
        for (int jj = 0; jj < SIMD_LEN; ++jj) {
            sum_scalar += temp_x[tid_mod][jj];
        }

        x1[i] += sum_scalar;
    }

    // 释放缓冲区
    vector_free(buf_a);
    vector_free(buf_y);
    vector_free(buf_temp);
}

// 向量化版本的 mvt_kernel2
__global__ void mvt_kernel2_vec(int n, double *a, double *x2, double *y_2)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int tid_mod = thread_id % 24;

    // 分配向量缓冲区
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_result = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_a || !buf_x || !buf_result) {
        if (buf_a) vector_free(buf_a);
        if (buf_x) vector_free(buf_x);
        if (buf_result) vector_free(buf_result);
        return;
    }

    // 计算任务分配
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (thread_id < remainder) ? thread_id * (elements_per_thread + 1)
                                            : remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (thread_id < remainder ? 1 : 0);

    // 外层循环 j
    for (int j = 0; j < n; j++) {
        double y_val = y_2[j];
        lvector double y_vec = (lvector double)vec_svbcast(y_val);

        // 内层循环 i，按向量长度处理
        for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
            int vec_end_i = min(i + SIMD_LEN, end_idx);

            if (i + SIMD_LEN <= end_idx) {
                // 加载 a[j*n+i:i+16] 和 x2[i:i+16]
                vector_load(&a[j * n + i], buf_a, VEC_BYTES);
                vector_load(&x2[i], buf_x, VEC_BYTES);

                // 从缓冲区加载到寄存器
                lvector double a_ji = vec_ld(0, buf_a);
                lvector double x_i = vec_ld(0, buf_x);

                // 计算 x2[i] += a[j][i] * y_2[j]
                lvector double mul_res = vec_muli(a_ji, y_vec);
                lvector double new_x_vec = vec_mula(mul_res, (lvector double)vec_svbcast(1.0), x_i);

                // 存储回缓冲区
                vec_st(new_x_vec, 0, buf_result);
                // 从缓冲区写回内存
                vector_store(buf_result, &x2[i], VEC_BYTES);
            } else {
                // 标量尾部处理
                for (int ii = i; ii < vec_end_i; ++ii) {
                    x2[ii] += a[j * n + ii] * y_val;
                }
            }
        }
    }

    // 释放缓冲区
    vector_free(buf_a);
    vector_free(buf_x);
    vector_free(buf_result);
}
#include "../MVT/kernel_vec.h"//大模型生成的存储文件
#include "../MVT/kernel_cache_llm.h"//SM缓存优化文件