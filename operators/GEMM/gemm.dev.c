#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void gemm_kernel(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                            double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);

    for (int idx = start; idx < end; ++idx) {
        c[idx] *= beta;
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            c[idx] += alpha * a[i * nk + k] * b[k * nj + j];
        }
    }
}
#define ELEMS_PER_PART 1024

__global__ void gemm_kernel_cache_fast(
    int ni, int nj, int nk,
    double alpha, double beta,
    double *a, double *b, double *c)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nj;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_a_row = (double*)scalar_malloc(nk * sizeof(double));
    double* cache_b_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_c     = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nj;            // 当前行
        int j_start = idx % nj;      // 当前起始列
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nj - j_start);

        // 1. 缓存 c 当前行块
        scalar_load(&c[i * nj + j_start], cache_c, batch_tasks * sizeof(double));

        // 2. 先乘 beta
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_c[bj] *= beta;
        }

        // 3. 缓存 a 当前行数据
        scalar_load(&a[i * nk], cache_a_row, nk * sizeof(double));

        // 4. GEMM 计算
        for (int k = 0; k < nk; ++k) {
            // 缓存 b[k行][j_start ... j_start+batch_tasks)
            scalar_load(&b[k * nj + j_start], cache_b_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_c[bj] += alpha * cache_a_row[k] * cache_b_row[bj];
            }
        }

        // 5. 写回结果
        scalar_store(cache_c, &c[i * nj + j_start], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_a_row);
    scalar_free(cache_b_row);
    scalar_free(cache_c);
}
__global__ void gemm_kernel_cache(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                  double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);
    int size = (end / nj) * nk + nk - 1 - (start / nj) * nk;
    CACHEs_INIT(b, double, b, 0, 13);
    CACHEs_INIT(c, double, c, 0, 14);
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_c, tmp_a, tmp_b;
    for (int idx = start; idx < end; ++idx) {
        CACHEs_RD(c, &c[idx], tmp_c);
        tmp_c *= beta;
        CACHEs_WT(c, &c[idx], tmp_c);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            CACHEs_RD(c, &c[idx], tmp_c);
            CACHEs_RD(a, &a[i * nk + k], tmp_a);
            CACHEs_RD(b, &b[k * nj + j], tmp_b);
            tmp_c += alpha * tmp_a * tmp_b;
            CACHEs_WT(c, &c[idx], tmp_c);
        }
    }
    CACHEs_FLUSH(c);
    CACHEs_INVALID(a);
    CACHEs_INVALID(b);
}

#ifdef MINI_DATASET
__global__ void gemm_kernel_cache(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                  double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);
    int size = (end / nj) * nk + nk - 1 - (start / nj) * nk;
    CACHEb_INIT(c, double, &c[start], 0, (end - start) * sizeof(double));
    int start_row = start / nj;
    int end_row = (end - 1) / nj;
    int a_elements = (end_row - start_row + 1) * nk;
    CACHEb_INIT(a, double, &a[start_row * nk], 0, a_elements * sizeof(double));
    CACHEs_INIT(b, double, b, 0, 15);
    double tmp_c, tmp_a, tmp_b;
    for (int idx = start; idx < end; ++idx) {
        CACHEb_RD(c, &c[idx - start], tmp_c);
        tmp_c *= beta;
        CACHEb_WT(c, &c[idx - start], tmp_c);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            CACHEb_RD(c, &c[idx - start], tmp_c);
            CACHEb_RD(a, &a[(i - start_row) * nk + k], tmp_a);
            CACHEs_RD(b, &b[k * nj + j], tmp_b);
            tmp_c += alpha * tmp_a * tmp_b;
            CACHEb_WT(c, &c[idx - start], tmp_c);
        }
    }
    CACHEb_FLUSH(c);
    CACHEb_INVALID(a);
    CACHEs_INVALID(b);
}
#endif

#ifdef SMALL_DATASET
__global__ void gemm_kernel_cache(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                  double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);
    int size = (end / nj) * nk + nk - 1 - (start / nj) * nk;
    CACHEb_INIT(c, double, &c[start], 0, (end - start) * sizeof(double));
    int start_row = start / nj;
    int end_row = (end - 1) / nj;
    int a_elements = (end_row - start_row + 1) * nk;
    CACHEb_INIT(a, double, &a[start_row * nk], 0, a_elements * sizeof(double));
    CACHEs_INIT(b, double, b, 0, 13);
    double tmp_c, tmp_a, tmp_b;
    for (int idx = start; idx < end; ++idx) {
        CACHEb_RD(c, &c[idx - start], tmp_c);
        tmp_c *= beta;
        CACHEb_WT(c, &c[idx - start], tmp_c);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            CACHEb_RD(c, &c[idx - start], tmp_c);
            CACHEb_RD(a, &a[(i - start_row) * nk + k], tmp_a);
            CACHEs_RD(b, &b[k * nj + j], tmp_b);
            tmp_c += alpha * tmp_a * tmp_b;
            CACHEb_WT(c, &c[idx - start], tmp_c);
        }
    }
    CACHEb_FLUSH(c);
    CACHEb_INVALID(a);
    CACHEs_INVALID(b);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void gemm_kernel_cache(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                  double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);
    int size = (end / nj) * nk + nk - 1 - (start / nj) * nk;
    CACHEs_INIT(b, double, b, 0, 14);
    // CACHEs_INIT(c, double, c, 0, 14);
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_c, tmp_a, tmp_b;
    for (int idx = start; idx < end; ++idx) {
        // CACHEs_RD(c, &c[idx], tmp_c);
        c[idx] *= beta;
        // CACHEs_WT(c, &c[idx], tmp_c);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            // CACHEs_RD(c, &c[idx], tmp_c);
            CACHEs_RD(a, &a[i * nk + k], tmp_a);
            CACHEs_RD(b, &b[k * nj + j], tmp_b);
            c[idx] += alpha * tmp_a * tmp_b;
            // CACHEs_WT(c, &c[idx], tmp_c);
        }
    }
    // CACHEs_FLUSH(c);
    CACHEs_INVALID(a);
    CACHEs_INVALID(b);
}
#endif

#ifdef LARGE_DATASET
__global__ void gemm_kernel_cache(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                  double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);
    int size = (end / nj) * nk + nk - 1 - (start / nj) * nk;
    CACHEs_INIT(b, double, b, 0, 14);
    // CACHEs_INIT(c, double, c, 0, 14);
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_c, tmp_a, tmp_b;
    for (int idx = start; idx < end; ++idx) {
        // CACHEs_RD(c, &c[idx], tmp_c);
        c[idx] *= beta;
        // CACHEs_WT(c, &c[idx], tmp_c);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            // CACHEs_RD(c, &c[idx], tmp_c);
            CACHEs_RD(a, &a[i * nk + k], tmp_a);
            CACHEs_RD(b, &b[k * nj + j], tmp_b);
            c[idx] += alpha * tmp_a * tmp_b;
            // CACHEs_WT(c, &c[idx], tmp_c);
        }
    }
    // CACHEs_FLUSH(c);
    CACHEs_INVALID(a);
    CACHEs_INVALID(b);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void gemm_kernel_cache(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                  double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);
    int size = (end / nj) * nk + nk - 1 - (start / nj) * nk;
    CACHEs_INIT(b, double, b, 0, 14);
    // CACHEs_INIT(c, double, c, 0, 14);
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_c, tmp_a, tmp_b;
    for (int idx = start; idx < end; ++idx) {
        // CACHEs_RD(c, &c[idx], tmp_c);
        c[idx] *= beta;
        // CACHEs_WT(c, &c[idx], tmp_c);
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            // CACHEs_RD(c, &c[idx], tmp_c);
            CACHEs_RD(a, &a[i * nk + k], tmp_a);
            CACHEs_RD(b, &b[k * nj + j], tmp_b);
            c[idx] += alpha * tmp_a * tmp_b;
            // CACHEs_WT(c, &c[idx], tmp_c);
        }
    }
    // CACHEs_FLUSH(c);
    CACHEs_INVALID(a);
    CACHEs_INVALID(b);
}
#endif
#define SIMD_LEN 16
#define VEC_BYTES 128

/*------------------------------------------------------------------*/
/* gemm_kernel_vec: Compute c[idx] = beta * c[idx] + alpha * Σ_k (a[i * nk + k] * b[k * nj + j]) */
/* Vectorized version of gemm_kernel */
/*------------------------------------------------------------------*/
__global__ void gemm_kernel_vec(int ni, int nj, int nk, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    /* Thread task division */
    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;
    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);

    /* Scale c[idx] *= beta */
    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end = min(idx + SIMD_LEN, end);

        if (idx + SIMD_LEN <= end && (idx % nj) + SIMD_LEN <= nj) { /* Full vector, within row */
            lvector double c_vec;
            vector_load(&c[idx], &c_vec, VEC_BYTES);

            /* c_vec *= beta */
            lvector double beta_vec = (lvector double)vec_svbcast(beta);
            c_vec = vec_muli(c_vec, beta_vec);
            vector_store(&c_vec, &c[idx], VEC_BYTES);
        } else { /* Remainder or crosses row boundary */
            for (int ii = idx; ii < vec_end; ++ii) {
                c[ii] *= beta;
            }
        }
    }

    /* Update c[idx] += alpha * a[i * nk + k] * b[k * nj + j] */
    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; idx += SIMD_LEN) {
            int vec_end = min(idx + SIMD_LEN, end);

            if (idx + SIMD_LEN <= end && (idx % nj) + SIMD_LEN <= nj) { /* Full vector, within row */
                int i = idx / nj;
                int j = idx % nj;

                lvector double c_vec, b_vec;
                vector_load(&c[idx], &c_vec, VEC_BYTES);
                vector_load(&b[k * nj + j], &b_vec, VEC_BYTES);

                /* Broadcast a[i * nk + k] * alpha */
                lvector double a_alpha_vec = (lvector double)vec_svbcast(alpha * a[i * nk + k]);

                /* c_vec += a_alpha_vec * b_vec */
                lvector double prod_vec = vec_muli(a_alpha_vec, b_vec);
                c_vec = vec_mula(prod_vec, (lvector double)vec_svbcast(1.0), c_vec);
                vector_store(&c_vec, &c[idx], VEC_BYTES);
            } else { /* Remainder or crosses row boundary */
                for (int ii = idx; ii < vec_end; ++ii) {
                    int i = ii / nj;
                    int j = ii % nj;
                    c[ii] += alpha * a[i * nk + k] * b[k * nj + j];
                }
            }
        }
    }
}
#include "../GEMM/kernel_vec.h"//大模型生成的存储文件
#include "../GEMM/kernel_cache_llm.h"//SM缓存优化文件