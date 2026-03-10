#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void gramschmidt_kernel1(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();

    if (tid == 0) {
        double nrm = 0.0;
        for (int i = 0; i < ni; i++) {
            nrm += a[i * nj + k] * a[i * nj + k];
        }
        r[k * nj + k] = sqrt(nrm);
    }
}

__global__ void gramschmidt_kernel2(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    for (int i = start_idx; i < end_idx; ++i) {
        q[i * nj + k] = a[i * nj + k] / r[k * nj + k];
    }
}

__global__ void gramschmidt_kernel3(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = nj - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }
    for (int j = start_idx; j < end_idx; ++j) {
        r[k * nj + j] = 0.0;
    }
    for (int i = 0; i < ni; i++) {
        for (int j = start_idx; j < end_idx; ++j) {
            r[k * nj + j] += q[i * nj + k] * a[i * nj + j];
        }
    }
    for (int i = 0; i < ni; i++) {
        for (int j = start_idx; j < end_idx; ++j) {
            a[i * nj + j] -= q[i * nj + k] * r[k * nj + j];
        }
    }
}

__global__ void gramschmidt_kernel1_cache(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    CACHEs_INIT(a, double, a, 0, 15);
    CACHEb_INIT(r, double, &r[k * nj + k], 0, sizeof(double));
    double tmp_a;
    if (tid == 0) {
        double nrm = 0.0;
        for (int i = 0; i < ni; i++) {
            CACHEs_RD(a, &a[i * nj + k], tmp_a);
            nrm += tmp_a * tmp_a;
        }
        CACHEb_WT(r, r, sqrt(nrm));
    }
    CACHEs_INVALID(a);
    CACHEb_FLUSH(r);
}

__global__ void gramschmidt_kernel2_cache(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    CACHEb_INIT(r, double, &r[k * nj + k], 0, sizeof(double));
    double tmp_r;
    CACHEb_RD(r, r, tmp_r);
    for (int i = start_idx; i < end_idx; ++i) {
        q[i * nj + k] = a[i * nj + k] / tmp_r;
    }
    CACHEb_INVALID(r);
}

__global__ void gramschmidt_kernel3_cache(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = nj - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }
    CACHEb_INIT(r, double, &r[k * nj + start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    CACHEs_INIT(q, double, q, 0, 4);
    double tmp_r, tmp_a, tmp_q;
    for (int j = start_idx; j < end_idx; ++j) {
        CACHEb_WT(r, &r[j - start_idx], 0.0);
    }
    for (int i = 0; i < ni; i++) {
        CACHEs_RD(q, &q[i * nj + k], tmp_q);
        for (int j = start_idx; j < end_idx; ++j) {
            CACHEs_RD(a, &a[i * nj + j], tmp_a);
            CACHEb_RD(r, &r[j - start_idx], tmp_r);
            tmp_r += tmp_q * tmp_a;
            CACHEb_WT(r, &r[j - start_idx], tmp_r);
        }
    }
    for (int i = 0; i < ni; i++) {
        CACHEs_RD(q, &q[i * nj + k], tmp_q);
        for (int j = start_idx; j < end_idx; ++j) {
            CACHEb_RD(r, &r[j - start_idx], tmp_r);
            CACHEs_RD(a, &a[i * nj + j], tmp_a);
            tmp_a -= tmp_q * tmp_r;
            CACHEs_WT(a, &a[i * nj + j], tmp_a);
        }
    }
    CACHEs_FLUSH(a);
    CACHEb_INVALID(r);
    CACHEs_INVALID(q);
}
#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double tmp_buf[24][SIMD_LEN];

/*------------------------------------------------------------------*/
/* gramschmidt_kernel1_vec: Compute r[k*nj + k] = sqrt(sum(a[i*nj + k]^2)) */
/* Vectorized version of gramschmidt_kernel1 */
/* Removed unused variable 'gsize' to fix warning */
/*------------------------------------------------------------------*/
__global__ void gramschmidt_kernel1_vec(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int tid_mod = tid % 24;

    /* Only thread 0 computes the norm, but we vectorize the sum */
    if (tid == 0) {
        lvector double nrm_vec = (lvector double)vec_svbcast(0.0);

        /* Process ni elements in chunks of SIMD_LEN */
        for (int i = 0; i < ni; i += SIMD_LEN) {
            int vec_end = min(i + SIMD_LEN, ni);

            if (i + SIMD_LEN <= ni) { /* Full vector */
                lvector double a_vec;
                vector_load(&a[i * nj + k], &a_vec, VEC_BYTES);

                /* nrm_vec += a_vec * a_vec */
                lvector double prod_vec = vec_muli(a_vec, a_vec);
                nrm_vec = vec_mula(prod_vec, (lvector double)vec_svbcast(1.0), nrm_vec);
            } else { /* Remainder */
                double temp = 0.0;
                for (int ii = i; ii < vec_end; ++ii) {
                    temp += a[ii * nj + k] * a[ii * nj + k];
                }
                /* Accumulate scalar remainder into tmp_buf and add to nrm_vec */
                tmp_buf[tid_mod][0] = temp;
                lvector double temp_vec;
                vector_load(tmp_buf[tid_mod], &temp_vec, VEC_BYTES);
                nrm_vec = vec_mula(temp_vec, (lvector double)vec_svbcast(1.0), nrm_vec);
            }
        }

        /* Sum across vector lanes and compute sqrt */
        double nrm = 0.0;
        vector_store(&nrm_vec, tmp_buf[tid_mod], VEC_BYTES);
        for (int j = 0; j < SIMD_LEN; ++j) {
            nrm += tmp_buf[tid_mod][j];
        }
        r[k * nj + k] = sqrt(nrm); /* Scalar sqrt for final result */
    }
}

/*------------------------------------------------------------------*/
/* gramschmidt_kernel2_vec: Compute q[i*nj + k] = a[i*nj + k] / r[k*nj + k] */
/* Vectorized version of gramschmidt_kernel2 (unchanged) */
/*------------------------------------------------------------------*/
__global__ void gramschmidt_kernel2_vec(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int gsize = get_group_size();

    /* Thread task division */
    int total_elements = ni;
    int elements_per_thread = total_elements / gsize;
    int remainder = total_elements % gsize;
    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    lvector double r_vec = (lvector double)vec_svbcast(r[k * nj + k]);

    /* Process elements in vector chunks */
    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end = min(i + SIMD_LEN, end_idx);

        if (i + SIMD_LEN <= end_idx) { /* Full vector */
            lvector double a_vec, q_vec;
            vector_load(&a[i * nj + k], &a_vec, VEC_BYTES);

            /* q_vec = a_vec / r_vec */
            q_vec = vm_fdivd16(a_vec, r_vec);
            vector_store(&q_vec, &q[i * nj + k], VEC_BYTES);
        } else { /* Remainder */
            for (int ii = i; ii < vec_end; ++ii) {
                q[ii * nj + k] = a[ii * nj + k] / r[k * nj + k];
            }
        }
    }
}

/*------------------------------------------------------------------*/
/* gramschmidt_kernel3_vec: Compute r[k*nj + j] and update a[i*nj + j] */
/* Vectorized version of gramschmidt_kernel3 */
/* Removed unused variable 'tid_mod' to fix warning */
/*------------------------------------------------------------------*/
__global__ void gramschmidt_kernel3_vec(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int gsize = get_group_size();

    /* Thread task division for columns j = k+1 to nj-1 */
    int total_elements = nj - k - 1;
    if (total_elements <= 0) return;

    int elements_per_thread = total_elements / gsize;
    int remainder = total_elements % gsize;
    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) return;

    /* Initialize r[k*nj + j] to 0 */
    for (int j = start_idx; j < end_idx; j += SIMD_LEN) {
        int vec_end = min(j + SIMD_LEN, end_idx);

        if (j + SIMD_LEN <= end_idx) { /* Full vector */
            lvector double zero_vec = (lvector double)vec_svbcast(0.0);
            vector_store(&zero_vec, &r[k * nj + j + k + 1], VEC_BYTES);
        } else { /* Remainder */
            for (int jj = j; jj < vec_end; ++jj) {
                r[k * nj + jj + k + 1] = 0.0;
            }
        }
    }

    /* Compute r[k*nj + j] = sum(q[i*nj + k] * a[i*nj + j]) */
    for (int i = 0; i < ni; ++i) {
        double q_val = q[i * nj + k];
        lvector double q_vec = (lvector double)vec_svbcast(q_val);

        for (int j = start_idx; j < end_idx; j += SIMD_LEN) {
            int vec_end = min(j + SIMD_LEN, end_idx);

            if (j + SIMD_LEN <= end_idx) { /* Full vector */
                lvector double r_vec, a_vec;
                vector_load(&r[k * nj + j + k + 1], &r_vec, VEC_BYTES);
                vector_load(&a[i * nj + j + k + 1], &a_vec, VEC_BYTES);

                /* r_vec += q_vec * a_vec */
                lvector double prod_vec = vec_muli(q_vec, a_vec);
                r_vec = vec_mula(prod_vec, (lvector double)vec_svbcast(1.0), r_vec);
                vector_store(&r_vec, &r[k * nj + j + k + 1], VEC_BYTES);
            } else { /* Remainder */
                for (int jj = j; jj < vec_end; ++jj) {
                    r[k * nj + jj + k + 1] += q[i * nj + k] * a[i * nj + jj + k + 1];
                }
            }
        }
    }

    /* Update a[i*nj + j] -= q[i*nj + k] * r[k*nj + j] */
    for (int i = 0; i < ni; ++i) {
        double q_val = q[i * nj + k];
        lvector double q_vec = (lvector double)vec_svbcast(q_val);

        for (int j = start_idx; j < end_idx; j += SIMD_LEN) {
            int vec_end = min(j + SIMD_LEN, end_idx);

            if (j + SIMD_LEN <= end_idx) { /* Full vector */
                lvector double a_vec, r_vec;
                vector_load(&a[i * nj + j + k + 1], &a_vec, VEC_BYTES);
                vector_load(&r[k * nj + j + k + 1], &r_vec, VEC_BYTES);

                /* a_vec -= q_vec * r_vec */
                lvector double prod_vec = vec_muli(q_vec, r_vec);
                a_vec = vec_mulb(a_vec, (lvector double)vec_svbcast(1.0), prod_vec);
                vector_store(&a_vec, &a[i * nj + j + k + 1], VEC_BYTES);
            } else { /* Remainder */
                for (int jj = j; jj < vec_end; ++jj) {
                    a[i * nj + jj + k + 1] -= q[i * nj + k] * r[k * nj + jj + k + 1];
                }
            }
        }
    }
}
#include "../GRAMSCHM/kernel_vec.h"//大模型生成的存储文件
#include "../GRAMSCHM/kernel_cache_llm.h"//SM缓存优化文件