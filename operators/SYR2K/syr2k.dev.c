#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void syr2k_kernel(int ni, int nj, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;

        c[idx] *= beta;
        for (int k = 0; k < nj; k++) {
            c[idx] += alpha * a[i * nj + k] * b[j * nj + k] + alpha * b[i * nj + k] * a[j * nj + k];
        }
    }
}

#ifdef MINI_DATASET
__global__ void syr2k_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *b,
                                   double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);
    CACHEb_INIT(c, double, &c[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEd_INIT(a, double, a, 9, 5);
    CACHEd_INIT(b, double, b, 9, 5);
    double tmp_c, tmp_a_ik, tmp_b_jk, tmp_b_ik, tmp_a_jk;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;
        CACHEb_RD(c, &c[idx - start_idx], tmp_c);
        tmp_c *= beta;
        for (int k = 0; k < nj; k++) {
            CACHEd_RD(a, &a[i * nj + k], tmp_a_ik);
            CACHEd_RD(b, &b[j * nj + k], tmp_b_jk);
            CACHEd_RD(b, &b[i * nj + k], tmp_b_ik);
            CACHEd_RD(a, &a[j * nj + k], tmp_a_jk);
            tmp_c += alpha * tmp_a_ik * tmp_b_jk + alpha * tmp_b_ik * tmp_a_jk;
        }
        CACHEb_WT(c, &c[idx - start_idx], tmp_c);
    }
    CACHEb_FLUSH(c);
    CACHEd_INVALID(a);
    CACHEd_INVALID(b);
}
#endif

#ifdef SMALL_DATASET
__global__ void syr2k_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *b,
                                   double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);
    CACHEs_INIT(c, double, c, 0, 13);
    CACHEd_INIT(a, double, a, 8, 6);
    CACHEd_INIT(b, double, b, 8, 6);
    double tmp_c, tmp_a_ik, tmp_b_jk, tmp_b_ik, tmp_a_jk;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;
        CACHEs_RD(c, &c[idx], tmp_c);
        tmp_c *= beta;
        for (int k = 0; k < nj; k++) {
            CACHEd_RD(a, &a[i * nj + k], tmp_a_ik);
            CACHEd_RD(b, &b[j * nj + k], tmp_b_jk);
            CACHEd_RD(b, &b[i * nj + k], tmp_b_ik);
            CACHEd_RD(a, &a[j * nj + k], tmp_a_jk);
            tmp_c += alpha * tmp_a_ik * tmp_b_jk + alpha * tmp_b_ik * tmp_a_jk;
        }
        CACHEs_WT(c, &c[idx], tmp_c);
    }
    CACHEs_FLUSH(c);
    CACHEd_INVALID(a);
    CACHEd_INVALID(b);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void syr2k_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *b,
                                   double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);
    CACHEs_INIT(c, double, c, 0, 12);
    CACHEd_INIT(a, double, a, 7, 7);
    CACHEd_INIT(b, double, b, 7, 7);
    double tmp_c, tmp_a_ik, tmp_b_jk, tmp_b_ik, tmp_a_jk;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;
        CACHEs_RD(c, &c[idx], tmp_c);
        tmp_c *= beta;
        for (int k = 0; k < nj; k++) {
            CACHEd_RD(a, &a[i * nj + k], tmp_a_ik);
            CACHEd_RD(b, &b[j * nj + k], tmp_b_jk);
            CACHEd_RD(b, &b[i * nj + k], tmp_b_ik);
            CACHEd_RD(a, &a[j * nj + k], tmp_a_jk);
            tmp_c += alpha * tmp_a_ik * tmp_b_jk + alpha * tmp_b_ik * tmp_a_jk;
        }
        CACHEs_WT(c, &c[idx], tmp_c);
    }
    CACHEs_FLUSH(c);
    CACHEd_INVALID(a);
    CACHEd_INVALID(b);
}
#endif

#ifdef LARGE_DATASET
__global__ void syr2k_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *b,
                                   double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);
    CACHEs_INIT(c, double, c, 0, 11);
    CACHEd_INIT(a, double, a, 7, 7);
    CACHEd_INIT(b, double, b, 7, 7);
    double tmp_c, tmp_a_ik, tmp_b_jk, tmp_b_ik, tmp_a_jk;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;
        CACHEs_RD(c, &c[idx], tmp_c);
        tmp_c *= beta;
        for (int k = 0; k < nj; k++) {
            CACHEd_RD(a, &a[i * nj + k], tmp_a_ik);
            CACHEd_RD(b, &b[i * nj + k], tmp_b_ik);
            CACHEd_RD(b, &b[j * nj + k], tmp_b_jk);
            CACHEd_RD(a, &a[j * nj + k], tmp_a_jk);
            tmp_c += alpha * tmp_a_ik * tmp_b_jk + alpha * tmp_b_ik * tmp_a_jk;
        }
        CACHEs_WT(c, &c[idx], tmp_c);
    }
    CACHEs_FLUSH(c);
    CACHEd_INVALID(a);
    CACHEd_INVALID(b);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void syr2k_kernel_cache(int ni, int nj, double alpha, double beta, double *a, double *b,
                                   double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);
    CACHEs_INIT(c, double, c, 0, 10);
    CACHEd_INIT(a, double, a, 5, 9);
    CACHEd_INIT(b, double, b, 5, 9);
    double tmp_c, tmp_a_ik, tmp_b_jk, tmp_b_ik, tmp_a_jk;
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;
        CACHEs_RD(c, &c[idx], tmp_c);
        tmp_c *= beta;
        for (int k = 0; k < nj; k++) {
            CACHEd_RD(a, &a[i * nj + k], tmp_a_ik);
            CACHEd_RD(b, &b[i * nj + k], tmp_b_ik);
            CACHEd_RD(b, &b[j * nj + k], tmp_b_jk);
            CACHEd_RD(a, &a[j * nj + k], tmp_a_jk);
            tmp_c += alpha * tmp_a_ik * tmp_b_jk + alpha * tmp_b_ik * tmp_a_jk;
        }
        CACHEs_WT(c, &c[idx], tmp_c);
    }
    CACHEs_FLUSH(c);
    CACHEd_INVALID(a);
    CACHEd_INVALID(b);
}
#endif
#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void syr2k_kernel_vec(int ni, int nj, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int tid_mod = thread_id % 24;

    // 分配向量缓冲区
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_a || !buf_b || !buf_c) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_c) vector_free(buf_c);
        return;
    }

    // 计算任务分配
    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;

        c[idx] *= beta;

        double sum_scalar = 0.0;
        lvector double sum_vec = (lvector double)vec_svbcast(0.0);

        for (int k = 0; k < nj; k += SIMD_LEN) {
            int vec_end_k = min(k + SIMD_LEN, nj);

            if (k + SIMD_LEN <= nj) {
                vector_load(&a[i * nj + k], buf_a, VEC_BYTES);
                vector_load(&b[j * nj + k], buf_b, VEC_BYTES);
                lvector double a_ik = vec_ld(0, buf_a);
                lvector double b_jk = vec_ld(0, buf_b);

                vector_load(&b[i * nj + k], buf_c, VEC_BYTES);
                lvector double b_ik = vec_ld(0, buf_c);
                vector_load(&a[j * nj + k], buf_c, VEC_BYTES);
                lvector double a_jk = vec_ld(0, buf_c);

                lvector double tmp1 = vec_muli(a_ik, b_jk);
                tmp1 = vec_muli(tmp1, alpha_vec);

                lvector double tmp2 = vec_muli(b_ik, a_jk);
                tmp2 = vec_muli(tmp2, alpha_vec);

                sum_vec = vec_mula(tmp1, (lvector double)vec_svbcast(1.0), sum_vec);
                sum_vec = vec_mula(tmp2, (lvector double)vec_svbcast(1.0), sum_vec);

            } else {
                for (int kk = k; kk < vec_end_k; ++kk) {
                    sum_scalar += alpha * a[i * nj + kk] * b[j * nj + kk];
                    sum_scalar += alpha * b[i * nj + kk] * a[j * nj + kk];
                }
            }
        }

        /* 用 sum_f64 从寄存器直接取值求和 */
        sum_scalar += sum_f64(sum_vec);

        c[idx] += sum_scalar;
    }

    // 释放缓冲区
    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_c);
}
#include "../SYR2K/kernel_vec.h"//大模型生成的存储文件
#include "../SYR2K/kernel_cache_llm.h"//SM缓存优化文件