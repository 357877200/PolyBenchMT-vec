#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void lu_kernel1(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }
    double tmp = A[k * n + k];
    for (int j = start_idx; j < end_idx; ++j) {
        A[k * n + j] = A[k * n + j] / tmp;
    }
}

__global__ void lu_kernel2(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = (n - k - 1);
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    for (int i = k + 1; i < n; ++i) {
        for (int j = start_idx; j < end_idx; ++j) {
            A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
        }
    }
}

__global__ void lu_kernel1_cache(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }
    double tmp = A[k * n + k];
    CACHEb_INIT(A, double, &A[k * n + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_a;
    for (int j = start_idx; j < end_idx; ++j) {
        CACHEb_RD(A, &A[j - start_idx], tmp_a);
        tmp_a = tmp_a / tmp;
        CACHEb_WT(A, &A[j - start_idx], tmp_a);
    }
    CACHEb_FLUSH(A);
}

__global__ void lu_kernel2_cache(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = (n - k - 1);
    // if (total_elements <= 0) {
    //     return;
    // }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    // if (start_idx >= end_idx) {
    //     return;
    // }
    CACHEd_INIT(A, double, A, 7, 8);
    double tmp_a_ik, tmp_a_kj, tmp_a_ij;
    for (int i = k + 1; i < n; ++i) {
        CACHEd_RD(A, &A[i * n + k], tmp_a_ik);
        for (int j = start_idx; j < end_idx; ++j) {
            CACHEd_RD(A, &A[k * n + j], tmp_a_kj);
            // CACHEd_RD(A, &A[i * n + j], tmp_a_ij);
            A[i * n + j] = A[i * n + j] - tmp_a_ik * tmp_a_kj;
            // CACHEd_WT(A, &A[i * n + j], tmp_a_ij);
        }
    }
    // CACHEd_FLUSH(A);
    CACHEd_INVALID(A);
}
#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void lu_kernel1_vec(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }

    /* 计算任务分配 */
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;
    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    /* 获取 A[k * n + k] 的倒数并广播为向量 */
    double tmp = A[k * n + k];
    lvector double v_inv_tmp = (lvector double)vm_frecd16((lvector double)vec_svbcast(tmp));

    /* 分配向量缓冲区 */
    lvector double *v_buf = (lvector double *)vector_malloc(sizeof(lvector double));

    /* 主循环：处理 j */
    for (int j = start_idx; j < end_idx; ) {
        int remain_elements = end_idx - j;

        if (remain_elements >= SIMD_LEN) {
            /* 向量化处理 SIMD_LEN 个元素 */
            vector_load(A + k * n + j, v_buf, VEC_BYTES);

            /* 计算 A[k * n + j:j+SIMD_LEN] / tmp */
            lvector double v_result = vec_muli(*v_buf, v_inv_tmp);

            /* 写回 A[k * n + j:j+SIMD_LEN] */
            vector_store(&v_result, A + k * n + j, VEC_BYTES);

            j += SIMD_LEN;
        } else {
            /* 标量尾部处理 */
            for (; j < end_idx; ++j) {
                A[k * n + j] = A[k * n + j] / tmp;
            }
        }
    }

    vector_free(v_buf);
}

__global__ void lu_kernel2_vec(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }


    /* 计算任务分配 */
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;
    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    /* 分配向量缓冲区 */
    lvector double *buf = (lvector double *)vector_malloc(sizeof(lvector double) * 2);
    lvector double *v_A_ij = buf + 0; /* A[i * n + j:j+SIMD_LEN] */
    lvector double *v_A_kj = buf + 1; /* A[k * n + j:j+SIMD_LEN] */

    /* 广播 1.0 为向量 */
    lvector double v_one = (lvector double)vec_svbcast(1.0);

    /* 外循环：处理 i */
    for (int i = k + 1; i < n; ++i) {
        /* 广播 A[i * n + k] 为向量 */
        lvector double v_A_ik = (lvector double)vec_svbcast(A[i * n + k]);

        /* 内循环：处理 j */
        for (int j = start_idx; j < end_idx; ) {
            int remain_elements = end_idx - j;

            if (remain_elements >= SIMD_LEN) {
                /* 向量化处理 SIMD_LEN 个元素 */
                vector_load(A + i * n + j, v_A_ij, VEC_BYTES);
                vector_load(A + k * n + j, v_A_kj, VEC_BYTES);

                /* 计算 A[i * n + j:j+SIMD_LEN] -= A[i * n + k] * A[k * n + j:j+SIMD_LEN] */
                lvector double v_product = vec_muli(v_A_ik, *v_A_kj); /* A[i * n + k] * A[k * n + j:j+SIMD_LEN] */
                lvector double v_result = vec_mulb(*v_A_ij, v_one, v_product); /* A[i * n + j] - v_product * 1.0 */

                /* 写回 A[i * n + j:j+SIMD_LEN] */
                vector_store(&v_result, A + i * n + j, VEC_BYTES);

                j += SIMD_LEN;
            } else {
                /* 标量尾部处理 */
                for (; j < end_idx; ++j) {
                    A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
                }
            }
        }
    }

    vector_free(buf);
}

#include "../LU/kernel_vec.h"//大模型生成的存储文件
#include "../LU/kernel_cache_llm.h"//SM缓存优化文件