#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void doitgen_kernel1(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    for (int i = start_idx; i < end_idx; i++) {
        sum[r * (nq * np) + i] = (double)0.0;
    }
    for (int s = 0; s < np; s++) {
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            sum[r * (nq * np) + i] += A[r * (nq * np) + q * np + s] * C4[s * np + p];
        }
    }
}

__global__ void doitgen_kernel2(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    for (int i = start_idx; i < end_idx; i++) {
        A[r * (nq * np) + i] = sum[r * (nq * np) + i];
    }
}


#ifdef MINI_DATASET
__global__ void doitgen_kernel1_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np)], 0, (nq * np) * sizeof(double));
    CACHEb_INIT(C4, double, C4, 0, (np * np) * sizeof(double));
    double tmp_sum, tmp_A, tmp_C4;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_WT(sum, &sum[i - start_idx], 0);
    }
    for (int s = 0; s < np; s++) {
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
            CACHEb_RD(A, &A[q * np + s], tmp_A);
            CACHEb_RD(C4, &C4[s * np + p], tmp_C4);
            tmp_sum += tmp_A * tmp_C4;
            CACHEb_WT(sum, &sum[i - start_idx], tmp_sum);
        }
    }
    CACHEb_FLUSH(sum);
    CACHEb_INVALID(A);
    CACHEb_INVALID(C4);
}

__global__ void doitgen_kernel2_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);
    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_sum;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
        CACHEb_WT(A, &A[i - start_idx], tmp_sum);
    }
    CACHEb_FLUSH(A);
    CACHEb_INVALID(sum);
}
#endif

#ifdef SMALL_DATASET
__global__ void doitgen_kernel1_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    int start_A_row = start_idx / np, end_A_row = end_idx / np;

    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_A_row * np], 0,
                ((end_A_row - start_A_row + 1) * np) * sizeof(double));
    double tmp_sum, tmp_A, tmp_C4;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_WT(sum, &sum[i - start_idx], 0);
    }
    for (int s = 0; s < np; s++) {
        CACHEb_INIT(C4, double, &C4[s * np], 0, np * sizeof(double));
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
            CACHEb_RD(A, &A[(q - start_A_row) * np + s], tmp_A);
            CACHEb_RD(C4, &C4[p], tmp_C4);
            tmp_sum += tmp_A * tmp_C4;
            CACHEb_WT(sum, &sum[i - start_idx], tmp_sum);
        }
        CACHEb_INVALID(C4);
    }
    CACHEb_FLUSH(sum);
    CACHEb_INVALID(A);
}

__global__ void doitgen_kernel2_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);
    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_sum;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
        CACHEb_WT(A, &A[i - start_idx], tmp_sum);
    }
    CACHEb_FLUSH(A);
    CACHEb_INVALID(sum);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void doitgen_kernel1_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    int start_A_row = start_idx / np, end_A_row = end_idx / np;

    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_A_row * np], 0,
                ((end_A_row - start_A_row + 1) * np) * sizeof(double));
    CACHEs_INIT(C4, double, C4, 0, 15);
    double tmp_sum, tmp_A, tmp_C4;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_WT(sum, &sum[i - start_idx], 0);
    }
    for (int s = 0; s < np; s++) {
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
            CACHEb_RD(A, &A[(q - start_A_row) * np + s], tmp_A);
            CACHEs_RD(C4, &C4[s * np + p], tmp_C4);
            tmp_sum += tmp_A * tmp_C4;
            CACHEb_WT(sum, &sum[i - start_idx], tmp_sum);
        }
    }
    CACHEb_FLUSH(sum);
    CACHEs_INVALID(C4);
    CACHEb_INVALID(A);
}

__global__ void doitgen_kernel2_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);
    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_sum;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
        CACHEb_WT(A, &A[i - start_idx], tmp_sum);
    }
    CACHEb_FLUSH(A);
    CACHEb_INVALID(sum);
}
#endif

#ifdef LARGE_DATASET
__global__ void doitgen_kernel1_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    int start_A_row = start_idx / np, end_A_row = end_idx / np;

    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_A_row * np], 0,
                ((end_A_row - start_A_row + 1) * np) * sizeof(double));
    CACHEs_INIT(C4, double, C4, 0, 13);
    double tmp_sum, tmp_A, tmp_C4;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_WT(sum, &sum[i - start_idx], 0);
    }
    for (int s = 0; s < np; s++) {
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
            CACHEb_RD(A, &A[(q - start_A_row) * np + s], tmp_A);
            CACHEs_RD(C4, &C4[s * np + p], tmp_C4);
            tmp_sum += tmp_A * tmp_C4;
            CACHEb_WT(sum, &sum[i - start_idx], tmp_sum);
        }
    }
    CACHEb_FLUSH(sum);
    CACHEs_INVALID(C4);
    CACHEb_INVALID(A);
}

__global__ void doitgen_kernel2_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);
    CACHEb_INIT(sum, double, &sum[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(A, double, &A[r * (nq * np) + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_sum;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(sum, &sum[i - start_idx], tmp_sum);
        CACHEb_WT(A, &A[i - start_idx], tmp_sum);
    }
    CACHEb_FLUSH(A);
    CACHEb_INVALID(sum);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void doitgen_kernel1_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    int start_A_row = start_idx / np, end_A_row = end_idx / np;

    CACHEs_INIT(C4, double, C4, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    CACHEs_INIT(sum, double, sum, 0, 14);
    double tmp_sum, tmp_A, tmp_C4;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEs_WT(sum, &sum[i], 0);
    }
    for (int s = 0; s < np; s++) {
        for (int i = start_idx; i < end_idx; i++) {
            int p = i % np;
            int q = i / np;
            CACHEs_RD(sum, &sum[r * (nq * np) + i], tmp_sum);
            CACHEs_RD(A, &A[r * (nq * np) + q * np + s], tmp_A);
            CACHEs_RD(C4, &C4[s * np + p], tmp_C4);
            tmp_sum += tmp_A * tmp_C4;
            CACHEs_WT(sum, &sum[r * (nq * np) + i], tmp_sum);
        }
    }
    CACHEs_FLUSH(sum);
    CACHEs_INVALID(C4);
    CACHEs_INVALID(A);
}

__global__ void doitgen_kernel2_cache(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(sum, double, sum, 0, 14);
    double tmp_sum;
    for (int i = start_idx; i < end_idx; i++) {
        CACHEs_RD(sum, &sum[i], tmp_sum);
        CACHEs_WT(A, &A[i], tmp_sum);
    }
    CACHEs_FLUSH(A);
    CACHEs_INVALID(sum);
}
#endif

#define SIMD_LEN 16
#define VEC_BYTES 128

/*------------------------------------------------------------------*/
/* doitgen_kernel1_vec: Compute sum[r * (nq * np) + i] = Σ_s A[r, q, s] * C4[s, p] */
/* Vectorized version of doitgen_kernel1 */
/*------------------------------------------------------------------*/
__global__ void doitgen_kernel1_vec(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();


    /* Thread task division */
    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    /* Initialize sum[r * (nq * np) + i] to 0 */
    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end = min(i + SIMD_LEN, end_idx);

        if (i + SIMD_LEN <= end_idx) { /* Full vector */
            lvector double zero_vec = (lvector double)vec_svbcast(0.0);
            vector_store(&zero_vec, &sum[r * (nq * np) + i], VEC_BYTES);
        } else { /* Remainder */
            for (int ii = i; ii < vec_end; ++ii) {
                sum[r * (nq * np) + ii] = 0.0;
            }
        }
    }

    /* Compute sum[r * (nq * np) + i] += A[r, q, s] * C4[s, p] */
    for (int s = 0; s < np; ++s) {
        for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
            int vec_end = min(i + SIMD_LEN, end_idx);

            if (i + SIMD_LEN <= end_idx && (i % np) + SIMD_LEN <= np) { /* Full vector, aligned */
                int q = i / np;
                int p = i % np;

                lvector double sum_vec, c4_vec;
                vector_load(&sum[r * (nq * np) + i], &sum_vec, VEC_BYTES);
                /* Load A[r, q, s] as a scalar broadcast since q is constant for this i */
                lvector double a_scalar = (lvector double)vec_svbcast(A[r * (nq * np) + q * np + s]);
                vector_load(&C4[s * np + p], &c4_vec, VEC_BYTES);

                /* sum_vec += a_scalar * c4_vec */
                lvector double prod_vec = vec_muli(a_scalar, c4_vec);
                sum_vec = vec_mula(prod_vec, (lvector double)vec_svbcast(1.0), sum_vec);
                vector_store(&sum_vec, &sum[r * (nq * np) + i], VEC_BYTES);
            } else { /* Remainder or non-aligned p */
                for (int ii = i; ii < vec_end; ++ii) {
                    int p = ii % np;
                    int q = ii / np;
                    double prod = A[r * (nq * np) + q * np + s] * C4[s * np + p];
                    sum[r * (nq * np) + ii] += prod;
                }
            }
        }
    }
}

/*------------------------------------------------------------------*/
/* doitgen_kernel2_vec: Copy sum[r * (nq * np) + i] to A[r * (nq * np) + i] */
/* Vectorized version of doitgen_kernel2 */
/*------------------------------------------------------------------*/
__global__ void doitgen_kernel2_vec(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    /* Thread task division */
    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    /* Copy sum to A */
    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end = min(i + SIMD_LEN, end_idx);

        if (i + SIMD_LEN <= end_idx) { /* Full vector */
            lvector double sum_vec;
            vector_load(&sum[r * (nq * np) + i], &sum_vec, VEC_BYTES);
            vector_store(&sum_vec, &A[r * (nq * np) + i], VEC_BYTES);
        } else { /* Remainder */
            for (int ii = i; ii < vec_end; ++ii) {
                A[r * (nq * np) + ii] = sum[r * (nq * np) + ii];
            }
        }
    }
}
#include "../DOITGEN/kernel_vec.h"//大模型生成的存储文件
#include "../DOITGEN/kernel_cache_llm.h"//SM缓存优化文件