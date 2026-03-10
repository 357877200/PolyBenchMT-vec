#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void gesummv_kernel(int n, double alpha, double beta, double *A, double *B, double *tmp,
                               double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);

    for (int i = start_i; i < end_i; i++) {
        tmp[i] = 0;
        y[i] = 0;

        for (int j = 0; j < n; j++) {
            tmp[i] += A[i * n + j] * x[j];
            y[i] += B[i * n + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

#ifdef MINI_DATASET
__global__ void gesummv_kernel_cache(int n, double alpha, double beta, double *A, double *B, double *tmp,
                                     double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEb_INIT(y, double, &y[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    double tmp_a, tmp_b, tmp_x, tmp_y, tmp_tmp;

    for (int i = start_i; i < end_i; i++) {
        tmp_tmp = 0;
        tmp_y = 0;

        for (int j = 0; j < n; j++) {
            CACHEs_RD(A, &A[i * n + j], tmp_a);
            CACHEs_RD(B, &B[i * n + j], tmp_b);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_tmp += +tmp_a * tmp_x;
            tmp_y = tmp_y + tmp_b * tmp_x;
        }
        tmp_y = alpha * tmp_tmp + beta * tmp_y;
        CACHEb_WT(y, &y[i - start_i], tmp_y);
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
    CACHEb_INVALID(x);
}
#endif

#ifdef SMALL_DATASET
__global__ void gesummv_kernel_cache(int n, double alpha, double beta, double *A, double *B, double *tmp,
                                     double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEb_INIT(y, double, &y[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    double tmp_a, tmp_b, tmp_x, tmp_y, tmp_tmp;

    for (int i = start_i; i < end_i; i++) {
        tmp_tmp = 0;
        tmp_y = 0;

        for (int j = 0; j < n; j++) {
            CACHEs_RD(A, &A[i * n + j], tmp_a);
            CACHEs_RD(B, &B[i * n + j], tmp_b);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_tmp += +tmp_a * tmp_x;
            tmp_y = tmp_y + tmp_b * tmp_x;
        }
        tmp_y = alpha * tmp_tmp + beta * tmp_y;
        CACHEb_WT(y, &y[i - start_i], tmp_y);
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
    CACHEb_INVALID(x);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void gesummv_kernel_cache(int n, double alpha, double beta, double *A, double *B, double *tmp,
                                     double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEb_INIT(y, double, &y[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    double tmp_a, tmp_b, tmp_x, tmp_y, tmp_tmp;

    for (int i = start_i; i < end_i; i++) {
        tmp_tmp = 0;
        tmp_y = 0;

        for (int j = 0; j < n; j++) {
            CACHEs_RD(A, &A[i * n + j], tmp_a);
            CACHEs_RD(B, &B[i * n + j], tmp_b);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_tmp += +tmp_a * tmp_x;
            tmp_y = tmp_y + tmp_b * tmp_x;
        }
        tmp_y = alpha * tmp_tmp + beta * tmp_y;
        CACHEb_WT(y, &y[i - start_i], tmp_y);
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
    CACHEb_INVALID(x);
}
#endif

#ifdef LARGE_DATASET
__global__ void gesummv_kernel_cache(int n, double alpha, double beta, double *A, double *B, double *tmp,
                                     double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 13);
    CACHEs_INIT(B, double, B, 0, 13);
    CACHEb_INIT(y, double, &y[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    double tmp_a, tmp_b, tmp_x, tmp_y, tmp_tmp;

    for (int i = start_i; i < end_i; i++) {
        tmp_tmp = 0;
        tmp_y = 0;

        for (int j = 0; j < n; j++) {
            CACHEs_RD(A, &A[i * n + j], tmp_a);
            CACHEs_RD(B, &B[i * n + j], tmp_b);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_tmp += +tmp_a * tmp_x;
            tmp_y = tmp_y + tmp_b * tmp_x;
        }
        tmp_y = alpha * tmp_tmp + beta * tmp_y;
        CACHEb_WT(y, &y[i - start_i], tmp_y);
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
    CACHEb_INVALID(x);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void gesummv_kernel_cache(int n, double alpha, double beta, double *A, double *B, double *tmp,
                                     double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 13);
    CACHEs_INIT(B, double, B, 0, 13);
    CACHEs_INIT(y, double, y, 0, 13);
    CACHEs_INIT(x, double, x, 0, 15);
    double tmp_a, tmp_b, tmp_x, tmp_y, tmp_tmp;

    for (int i = start_i; i < end_i; i++) {
        tmp_tmp = 0;
        tmp_y = 0;

        for (int j = 0; j < n; j++) {
            CACHEs_RD(A, &A[i * n + j], tmp_a);
            CACHEs_RD(B, &B[i * n + j], tmp_b);
            CACHEs_RD(x, &x[j], tmp_x);
            tmp_tmp += +tmp_a * tmp_x;
            tmp_y = tmp_y + tmp_b * tmp_x;
        }
        tmp_y = alpha * tmp_tmp + beta * tmp_y;
        CACHEs_WT(y, &y[i], tmp_y);
    }
    CACHEs_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
    CACHEs_INVALID(x);
}
#endif
#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void gesummv_kernel_vec(int n, double alpha, double beta, double *A, double *B, double *tmp,
                                   double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    /* Check for thread safety */
    if (group_size > 24) return; /* Prevent __gsm__ array overflow */

    /* Task distribution */
    int base = n / group_size;
    int remainder = n % group_size;
    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);

    /* Allocate vector buffers for A, B, and x */
    lvector double *buf = (lvector double *)vector_malloc(sizeof(lvector double) * 3);
    lvector double *v_A = buf + 0;   /* Buffer for A[i,j:j+SIMD_LEN] */
    lvector double *v_B = buf + 1;   /* Buffer for B[i,j:j+SIMD_LEN] */
    lvector double *v_x = buf + 2;   /* Buffer for x[j:j+SIMD_LEN] */

    /* Thread-safe __gsm__ storage for tmp and y accumulators */
    __gsm__ static double tmp_array[24][SIMD_LEN];
    __gsm__ static double y_array[24][SIMD_LEN];

    /* Initialize __gsm__ arrays */
    for (int k = 0; k < SIMD_LEN; k++) {
        tmp_array[tid][k] = 0.0;
        y_array[tid][k] = 0.0;
    }

    for (int i = start_i; i < end_i; i++) {
        lvector double v_tmp = (lvector double)vec_svbcast(0.0); /* Accumulator for tmp[i:i+SIMD_LEN] */
        lvector double v_y = (lvector double)vec_svbcast(0.0);   /* Accumulator for y[i:i+SIMD_LEN] */

        int j;
        /* Vectorized loop: process SIMD_LEN elements */
        for (j = 0; j <= n - SIMD_LEN; j += SIMD_LEN) {
            /* Load A[i,j:j+SIMD_LEN], B[i,j:j+SIMD_LEN], and x[j:j+SIMD_LEN] */
            vector_load(A + i * n + j, v_A, VEC_BYTES);
            vector_load(B + i * n + j, v_B, VEC_BYTES);
            vector_load(x + j, v_x, VEC_BYTES);

            /* Accumulate: v_tmp += A[i,j:j+SIMD_LEN] * x[j:j+SIMD_LEN] */
            v_tmp = vec_mula(*v_A, *v_x, v_tmp);
            /* Accumulate: v_y += B[i,j:j+SIMD_LEN] * x[j:j+SIMD_LEN] */
            v_y = vec_mula(*v_B, *v_x, v_y);
        }

        /* Store vector results to __gsm__ arrays */
        vector_store(&v_tmp, tmp_array[tid], VEC_BYTES);
        vector_store(&v_y, y_array[tid], VEC_BYTES);

        /* Scalar tail: process remaining elements */
        double s_tmp = 0.0, s_y = 0.0;
        for (; j < n; j++) {
            s_tmp += A[i * n + j] * x[j];
            s_y += B[i * n + j] * x[j];
        }

        /* Sum vector results and add scalar tail */
        for (int k = 0; k < SIMD_LEN && (j - k - 1) < n; k++) {
            s_tmp += tmp_array[tid][k];
            s_y += y_array[tid][k];
        }

        /* Compute final y[i] = alpha * tmp[i] + beta * y[i] */
        y[i] = alpha * s_tmp + beta * s_y;
        tmp[i] = s_tmp; /* Store tmp[i] for consistency */
    }

    vector_free(buf);
}
#include "../GESUMMV/kernel_vec.h"//大模型生成的存储文件
#include "../GESUMMV/kernel_cache_llm.h"//SM缓存优化文件