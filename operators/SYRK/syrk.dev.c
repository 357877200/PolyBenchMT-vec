#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void syrk_kernel(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; ++j) {
            c[i * ni + j] = c[i * ni + j] * beta;
            for (int k = 0; k < nj; k++) {
                c[i * ni + j] += alpha * a[i * nj + k] * a[j * nj + k];
            }
        }
    }
}

#ifdef MINI_DATASET
__global__ void syrk_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }
    CACHEs_INIT(c, double, c, 0, 13);
    CACHEd_INIT(a, double, a, 9, 6);
    double tmp_c, tmp_a1, tmp_a2;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; ++j) {
            CACHEs_RD(c, &c[i * nj + j], tmp_c);
            tmp_c = tmp_c * beta;
            for (int k = 0; k < nj; k++) {
                CACHEd_RD(a, &a[i * nj + k], tmp_a1);
                CACHEd_RD(a, &a[j * nj + k], tmp_a2);
                tmp_c += alpha * tmp_a1 * tmp_a2;
            }
            CACHEs_WT(c, &c[i * nj + j], tmp_c);
        }
    }
    CACHEs_FLUSH(c);
    CACHEd_INVALID(a);
}
#endif

#ifdef SMALL_DATASET
__global__ void syrk_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }
    CACHEs_INIT(c, double, c, 0, 13);
    CACHEd_INIT(a, double, a, 8, 7);
    double tmp_c, tmp_a1, tmp_a2;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; ++j) {
            CACHEs_RD(c, &c[i * nj + j], tmp_c);
            tmp_c = tmp_c * beta;
            for (int k = 0; k < nj; k++) {
                CACHEd_RD(a, &a[i * nj + k], tmp_a1);
                CACHEd_RD(a, &a[j * nj + k], tmp_a2);
                tmp_c += alpha * tmp_a1 * tmp_a2;
            }
            CACHEs_WT(c, &c[i * nj + j], tmp_c);
        }
    }
    CACHEd_INVALID(a);
    CACHEs_FLUSH(c);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void syrk_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }
    CACHEs_INIT(c, double, c, 0, 12);
    CACHEd_INIT(a, double, a, 7, 8);
    double tmp_c, tmp_a1, tmp_a2;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; ++j) {
            CACHEs_RD(c, &c[i * nj + j], tmp_c);
            tmp_c = tmp_c * beta;
            for (int k = 0; k < nj; k++) {
                CACHEd_RD(a, &a[i * nj + k], tmp_a1);
                CACHEd_RD(a, &a[j * nj + k], tmp_a2);
                tmp_c += alpha * tmp_a1 * tmp_a2;
            }
            CACHEs_WT(c, &c[i * nj + j], tmp_c);
        }
    }
    CACHEd_INVALID(a);
    CACHEs_FLUSH(c);
}
#endif

#ifdef LARGE_DATASET
__global__ void syrk_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }
    CACHEs_INIT(c, double, c, 0, 11);
    CACHEd_INIT(a, double, a, 6, 9);
    double tmp_c, tmp_a1, tmp_a2;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; ++j) {
            CACHEs_RD(c, &c[i * nj + j], tmp_c);
            tmp_c = tmp_c * beta;
            for (int k = 0; k < nj; k++) {
                CACHEd_RD(a, &a[i * nj + k], tmp_a1);
                CACHEd_RD(a, &a[j * nj + k], tmp_a2);
                tmp_c += alpha * tmp_a1 * tmp_a2;
            }
            CACHEs_WT(c, &c[i * nj + j], tmp_c);
        }
    }
    CACHEd_INVALID(a);
    CACHEs_FLUSH(c);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void syrk_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }
    CACHEs_INIT(c, double, c, 0, 10);
    CACHEd_INIT(a, double, a, 6, 9);
    double tmp_c, tmp_a1, tmp_a2;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; ++j) {
            CACHEs_RD(c, &c[i * nj + j], tmp_c);
            tmp_c = tmp_c * beta;
            for (int k = 0; k < nj; k++) {
                CACHEd_RD(a, &a[i * nj + k], tmp_a1);
                CACHEd_RD(a, &a[j * nj + k], tmp_a2);
                tmp_c += alpha * tmp_a1 * tmp_a2;
            }
            CACHEs_WT(c, &c[i * nj + j], tmp_c);
        }
    }
    CACHEd_INVALID(a);
    CACHEs_FLUSH(c);
}
#endif

#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void syrk_kernel_vec(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 分配向量缓冲区
    lvector double *buf_a_ik = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_jk = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_a_ik || !buf_a_jk) {
        if (buf_a_ik) vector_free(buf_a_ik);
        if (buf_a_jk) vector_free(buf_a_jk);
        return;
    }

    // 任务分配
    int work_per_thread = (ni + num_threads - 1) / num_threads;
    int start_i = thread_id * work_per_thread;
    int end_i = min(start_i + work_per_thread, ni);

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
    lvector double one_vec   = (lvector double)vec_svbcast(1.0);

    for (int i = start_i; i < end_i; ++i) {
        for (int j = 0; j < ni; ++j) {
            int idx = i * ni + j;

            c[idx] *= beta;

            double sum_scalar = 0.0;
            lvector double sum_vec = (lvector double)vec_svbcast(0.0);

            for (int k = 0; k < nj; k += SIMD_LEN) {
                int vec_end_k = min(k + SIMD_LEN, nj);

                if (k + SIMD_LEN <= nj) {
                    vector_load(&a[i * nj + k], buf_a_ik, VEC_BYTES);
                    vector_load(&a[j * nj + k], buf_a_jk, VEC_BYTES);

                    lvector double a_ik = vec_ld(0, buf_a_ik);
                    lvector double a_jk = vec_ld(0, buf_a_jk);

                    lvector double tmp = vec_muli(a_ik, a_jk);
                    tmp = vec_muli(tmp, alpha_vec);

                    sum_vec = vec_mula(tmp, one_vec, sum_vec);
                } else {
                    for (int kk = k; kk < vec_end_k; ++kk) {
                        sum_scalar += alpha * a[i * nj + kk] * a[j * nj + kk];
                    }
                }
            }

            /* 用sum_f64直接从寄存器求和 */
            sum_scalar += sum_f64(sum_vec);

            c[idx] += sum_scalar;
        }
    }

    vector_free(buf_a_ik);
    vector_free(buf_a_jk);
}
#include "../SYRK/kernel_vec.h"//大模型生成的存储文件
#include "../SYRK/kernel_cache_llm.h"//SM缓存优化文件