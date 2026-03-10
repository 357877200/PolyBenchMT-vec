#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void gemver_kernel1(int n, double alpha, double beta, double *a, double *v1, double *v2,
                               double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
        }
    }
}

__global__ void gemver_kernel2(int n, double alpha, double beta, double *a, double *x, double *y,
                               double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);

    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            x[i] += beta * a[j * n + i] * y[j];
        }
    }
    for (int i = start_idx; i < end_idx; ++i) {
        x[i] += z[i];
    }
}

__global__ void gemver_kernel3(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            w[i] += alpha * a[i * n + j] * x[j];
        }
    }
}

#ifdef MINI_DATASET
__global__ void gemver_kernel1_cache(int n, double alpha, double beta, double *a, double *v1, double *v2,
                                     double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(u1, double, &u1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(u2, double, &u2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(v1, double, v1, 0, n * sizeof(double));
    CACHEb_INIT(v2, double, v2, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_u1, tmp_u2, tmp_v1, tmp_v2, tmp_a;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; ++j) {
            CACHEb_RD(u1, &u1[i - start_idx], tmp_u1);
            CACHEb_RD(u2, &u2[i - start_idx], tmp_u2);
            CACHEb_RD(v1, &v1[j], tmp_v1);
            CACHEb_RD(v2, &v2[j], tmp_v2);
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            tmp_a += tmp_u1 * tmp_v1 + tmp_u2 * tmp_v2;
            CACHEs_WT(a, &a[i * n + j], tmp_a);
        }
    }
    CACHEs_FLUSH(a);
    CACHEb_INVALID(u1);
    CACHEb_INVALID(u2);
    CACHEb_INVALID(v1);
    CACHEb_INVALID(v2);
}

__global__ void gemver_kernel2_cache(int n, double alpha, double beta, double *a, double *x, double *y,
                                     double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(x, double, &x[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(z, double, &z[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y, double, y, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x, tmp_a, tmp_y, tmp_z;
    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x, &x[i - start_idx], tmp_x);
            CACHEs_RD(a, &a[j * n + i], tmp_a);
            CACHEb_RD(y, &y[j], tmp_y);
            tmp_x += beta * tmp_a * tmp_y;
            CACHEb_WT(x, &x[i - start_idx], tmp_x);
        }
    }
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x, &x[i - start_idx], tmp_x);
        CACHEb_RD(z, &z[i - start_idx], tmp_z);
        tmp_x += tmp_z;
        CACHEb_WT(x, &x[i - start_idx], tmp_x);
    }
    CACHEb_FLUSH(x);
    CACHEs_INVALID(a);
    CACHEb_INVALID(y);
    CACHEb_INVALID(z);
}

__global__ void gemver_kernel3_cache(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEs_INIT(a, double, a, 0, 15);
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    CACHEb_INIT(w, double, &w[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_a, tmp_x, tmp_w;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            CACHEb_RD(w, &w[i - start_idx], tmp_w);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_w += alpha * tmp_a * tmp_x;
            CACHEb_WT(w, &w[i - start_idx], tmp_w);
        }
    }
    CACHEb_FLUSH(w);
    CACHEs_INVALID(a);
    CACHEb_INVALID(x);
}
#endif

#ifdef SMALL_DATASET
__global__ void gemver_kernel1_cache(int n, double alpha, double beta, double *a, double *v1, double *v2,
                                     double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(u1, double, &u1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(u2, double, &u2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(v1, double, v1, 0, n * sizeof(double));
    CACHEb_INIT(v2, double, v2, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 14);
    double tmp_u1, tmp_u2, tmp_v1, tmp_v2, tmp_a;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; ++j) {
            CACHEb_RD(u1, &u1[i - start_idx], tmp_u1);
            CACHEb_RD(u2, &u2[i - start_idx], tmp_u2);
            CACHEb_RD(v1, &v1[j], tmp_v1);
            CACHEb_RD(v2, &v2[j], tmp_v2);
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            tmp_a += tmp_u1 * tmp_v1 + tmp_u2 * tmp_v2;
            CACHEs_WT(a, &a[i * n + j], tmp_a);
        }
    }
    CACHEs_FLUSH(a);
    CACHEb_INVALID(u1);
    CACHEb_INVALID(u2);
    CACHEb_INVALID(v1);
    CACHEb_INVALID(v2);
}

__global__ void gemver_kernel2_cache(int n, double alpha, double beta, double *a, double *x, double *y,
                                     double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(x, double, &x[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(z, double, &z[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y, double, y, 0, n * sizeof(double));
    CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x, tmp_a, tmp_y, tmp_z;
    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x, &x[i - start_idx], tmp_x);
            CACHEs_RD(a, &a[j * n + i], tmp_a);
            CACHEb_RD(y, &y[j], tmp_y);
            tmp_x += beta * tmp_a * tmp_y;
            CACHEb_WT(x, &x[i - start_idx], tmp_x);
        }
    }
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x, &x[i - start_idx], tmp_x);
        CACHEb_RD(z, &z[i - start_idx], tmp_z);
        tmp_x += tmp_z;
        CACHEb_WT(x, &x[i - start_idx], tmp_x);
    }
    CACHEb_FLUSH(x);
    CACHEs_INVALID(a);
    CACHEb_INVALID(y);
    CACHEb_INVALID(z);
}

__global__ void gemver_kernel3_cache(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEs_INIT(a, double, a, 0, 15);
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    CACHEb_INIT(w, double, &w[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_a, tmp_x, tmp_w;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            CACHEb_RD(w, &w[i - start_idx], tmp_w);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_w += alpha * tmp_a * tmp_x;
            CACHEb_WT(w, &w[i - start_idx], tmp_w);
        }
    }
    CACHEb_FLUSH(w);
    CACHEs_INVALID(a);
    CACHEb_INVALID(x);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void gemver_kernel1_cache(int n, double alpha, double beta, double *a, double *v1, double *v2,
                                     double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(u1, double, &u1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(u2, double, &u2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(v1, double, v1, 0, 14);
    CACHEs_INIT(v2, double, v2, 0, 14);
    CACHEs_INIT(a, double, a, 0, 6);
    double tmp_u1, tmp_u2, tmp_v1, tmp_v2, tmp_a;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; ++j) {
            CACHEb_RD(u1, &u1[i - start_idx], tmp_u1);
            CACHEb_RD(u2, &u2[i - start_idx], tmp_u2);
            CACHEs_RD(v1, &v1[j], tmp_v1);
            CACHEs_RD(v2, &v2[j], tmp_v2);
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            tmp_a += tmp_u1 * tmp_v1 + tmp_u2 * tmp_v2;
            CACHEs_WT(a, &a[i * n + j], tmp_a);
        }
    }
    CACHEs_FLUSH(a);
    CACHEb_INVALID(u1);
    CACHEb_INVALID(u2);
    CACHEs_INVALID(v1);
    CACHEs_INVALID(v2);
}

__global__ void gemver_kernel2_cache(int n, double alpha, double beta, double *a, double *x, double *y,
                                     double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(x, double, &x[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(z, double, &z[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(y, double, y, 0, n * sizeof(double));
    // CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x, tmp_a, tmp_y, tmp_z;
    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x, &x[i - start_idx], tmp_x);
            // CACHEs_RD(a, &a[j * n + i], tmp_a);
            CACHEb_RD(y, &y[j], tmp_y);
            tmp_x += beta * a[j * n + i] * tmp_y;
            CACHEb_WT(x, &x[i - start_idx], tmp_x);
        }
    }
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x, &x[i - start_idx], tmp_x);
        CACHEb_RD(z, &z[i - start_idx], tmp_z);
        tmp_x += tmp_z;
        CACHEb_WT(x, &x[i - start_idx], tmp_x);
    }
    CACHEb_FLUSH(x);
    // CACHEs_INVALID(a);
    CACHEb_INVALID(y);
    CACHEb_INVALID(z);
}

__global__ void gemver_kernel3_cache(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEs_INIT(a, double, a, 0, 14);
    CACHEb_INIT(x, double, x, 0, n * sizeof(double));
    CACHEb_INIT(w, double, &w[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_a, tmp_x, tmp_w;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            CACHEb_RD(w, &w[i - start_idx], tmp_w);
            CACHEb_RD(x, &x[j], tmp_x);
            tmp_w += alpha * tmp_a * tmp_x;
            CACHEb_WT(w, &w[i - start_idx], tmp_w);
        }
    }
    CACHEb_FLUSH(w);
    CACHEs_INVALID(a);
    CACHEb_INVALID(x);
}
#endif

#ifdef LARGE_DATASET
__global__ void gemver_kernel1_cache(int n, double alpha, double beta, double *a, double *v1, double *v2,
                                     double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(u1, double, &u1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(u2, double, &u2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(v1, double, v1, 0, 14);
    CACHEs_INIT(v2, double, v2, 0, 14);
    CACHEs_INIT(a, double, a, 0, 7);
    double tmp_u1, tmp_u2, tmp_v1, tmp_v2, tmp_a;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; ++j) {
            CACHEb_RD(u1, &u1[i - start_idx], tmp_u1);
            CACHEb_RD(u2, &u2[i - start_idx], tmp_u2);
            CACHEs_RD(v1, &v1[j], tmp_v1);
            CACHEs_RD(v2, &v2[j], tmp_v2);
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            tmp_a += tmp_u1 * tmp_v1 + tmp_u2 * tmp_v2;
            CACHEs_WT(a, &a[i * n + j], tmp_a);
        }
    }
    CACHEs_FLUSH(a);
    CACHEb_INVALID(u1);
    CACHEb_INVALID(u2);
    CACHEs_INVALID(v1);
    CACHEs_INVALID(v2);
}

__global__ void gemver_kernel2_cache(int n, double alpha, double beta, double *a, double *x, double *y,
                                     double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(x, double, &x[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(z, double, z, 0, 5);
    CACHEs_INIT(y, double, y, 0, 14);
    // CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x, tmp_a, tmp_y, tmp_z;
    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x, &x[i - start_idx], tmp_x);
            // CACHEs_RD(a, &a[j * n + i], tmp_a);
            CACHEb_RD(y, &y[j], tmp_y);
            tmp_x += beta * a[j * n + i] * tmp_y;
            CACHEb_WT(x, &x[i - start_idx], tmp_x);
        }
    }
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x, &x[i - start_idx], tmp_x);
        CACHEs_RD(z, &z[i], tmp_z);
        tmp_x += tmp_z;
        CACHEb_WT(x, &x[i - start_idx], tmp_x);
    }
    CACHEb_FLUSH(x);
    // CACHEs_INVALID(a);
    CACHEs_INVALID(y);
    CACHEs_INVALID(z);
}

__global__ void gemver_kernel3_cache(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEs_INIT(a, double, a, 0, 7);
    CACHEs_INIT(x, double, x, 0, 15);
    CACHEs_INIT(w, double, w, 0, 9);
    double tmp_a, tmp_x, tmp_w;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            CACHEs_RD(w, &w[i], tmp_w);
            CACHEs_RD(x, &x[j], tmp_x);
            tmp_w += alpha * tmp_a * tmp_x;
            CACHEs_WT(w, &w[i], tmp_w);
        }
    }
    CACHEs_FLUSH(w);
    CACHEs_INVALID(a);
    CACHEs_INVALID(x);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void gemver_kernel1_cache(int n, double alpha, double beta, double *a, double *v1, double *v2,
                                     double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(u1, double, &u1[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(u2, double, &u2[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(v1, double, v1, 0, 14);
    CACHEs_INIT(v2, double, v2, 0, 14);
    CACHEs_INIT(a, double, a, 0, 7);
    double tmp_u1, tmp_u2, tmp_v1, tmp_v2, tmp_a;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; ++j) {
            CACHEb_RD(u1, &u1[i - start_idx], tmp_u1);
            CACHEb_RD(u2, &u2[i - start_idx], tmp_u2);
            CACHEs_RD(v1, &v1[j], tmp_v1);
            CACHEs_RD(v2, &v2[j], tmp_v2);
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            tmp_a += tmp_u1 * tmp_v1 + tmp_u2 * tmp_v2;
            CACHEs_WT(a, &a[i * n + j], tmp_a);
        }
    }
    CACHEs_FLUSH(a);
    CACHEb_INVALID(u1);
    CACHEb_INVALID(u2);
    CACHEs_INVALID(v1);
    CACHEs_INVALID(v2);
}

__global__ void gemver_kernel2_cache(int n, double alpha, double beta, double *a, double *x, double *y,
                                     double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEb_INIT(x, double, &x[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(z, double, z, 0, 4);
    CACHEs_INIT(y, double, y, 0, 15);
    // CACHEs_INIT(a, double, a, 0, 15);
    double tmp_x, tmp_a, tmp_y, tmp_z;
    for (int j = 0; j < n; j++) {
        for (int i = start_idx; i < end_idx; ++i) {
            CACHEb_RD(x, &x[i - start_idx], tmp_x);
            // CACHEs_RD(a, &a[j * n + i], tmp_a);
            CACHEb_RD(y, &y[j], tmp_y);
            tmp_x += beta * a[j * n + i] * tmp_y;
            CACHEb_WT(x, &x[i - start_idx], tmp_x);
        }
    }
    for (int i = start_idx; i < end_idx; ++i) {
        CACHEb_RD(x, &x[i - start_idx], tmp_x);
        CACHEs_RD(z, &z[i], tmp_z);
        tmp_x += tmp_z;
        CACHEb_WT(x, &x[i - start_idx], tmp_x);
    }
    CACHEb_FLUSH(x);
    // CACHEs_INVALID(a);
    CACHEs_INVALID(y);
    CACHEs_INVALID(z);
}

__global__ void gemver_kernel3_cache(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);
    CACHEs_INIT(a, double, a, 0, 6);
    CACHEs_INIT(x, double, x, 0, 15);
    CACHEs_INIT(w, double, w, 0, 8);
    double tmp_a, tmp_x, tmp_w;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < n; j++) {
            CACHEs_RD(a, &a[i * n + j], tmp_a);
            CACHEs_RD(w, &w[i], tmp_w);
            CACHEs_RD(x, &x[j], tmp_x);
            tmp_w += alpha * tmp_a * tmp_x;
            CACHEs_WT(w, &w[i], tmp_w);
        }
    }
    CACHEs_FLUSH(w);
    CACHEs_INVALID(a);
    CACHEs_INVALID(x);
}
#endif

#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double tmp_buf[24][SIMD_LEN];

/*------------------------------------------------------------------*/
/* gemver_kernel1_vec: Update a[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j] */
/* Vectorized version of gemver_kernel1 */
/*------------------------------------------------------------------*/
__global__ void gemver_kernel1_vec(int n, double alpha, double beta, double *a, double *v1, double *v2, double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    for (int i = start_idx; i < end_idx; ++i) {
        lvector double u1_vec = (lvector double)vec_svbcast(u1[i]);
        lvector double u2_vec = (lvector double)vec_svbcast(u2[i]);

        for (int j = 0; j < n; j += SIMD_LEN) {
            int vec_end = min(j + SIMD_LEN, n);

            if (j + SIMD_LEN <= n) { /* Full vector */
                lvector double a_vec, v1_vec, v2_vec;
                vector_load(&a[i * n + j], &a_vec, VEC_BYTES);
                vector_load(&v1[j], &v1_vec, VEC_BYTES);
                vector_load(&v2[j], &v2_vec, VEC_BYTES);

                /* Compute u1[i] * v1[j] */
                lvector double prod1_vec = vec_muli(u1_vec, v1_vec);
                /* Compute u2[i] * v2[j] */
                lvector double prod2_vec = vec_muli(u2_vec, v2_vec);
                /* a_vec += prod1_vec + prod2_vec */
                a_vec = vec_mula(prod1_vec, (lvector double)vec_svbcast(1.0), a_vec);
                a_vec = vec_mula(prod2_vec, (lvector double)vec_svbcast(1.0), a_vec);
                vector_store(&a_vec, &a[i * n + j], VEC_BYTES);
            } else { /* Remainder */
                for (int jj = j; jj < vec_end; ++jj) {
                    a[i * n + jj] += u1[i] * v1[jj] + u2[i] * v2[jj];
                }
            }
        }
    }
}

/*------------------------------------------------------------------*/
/* gemver_kernel2_vec: Update x[i] += beta * a[j * n + i] * y[j] and x[i] += z[i] */
/* Vectorized version of gemver_kernel2 */
/*------------------------------------------------------------------*/
__global__ void gemver_kernel2_vec(int n, double alpha, double beta, double *a, double *x, double *y, double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    /* Update x[i] += beta * a[j * n + i] * y[j] */
    for (int j = 0; j < n; ++j) {
        lvector double y_vec = (lvector double)vec_svbcast(y[j]);
        lvector double beta_vec = (lvector double)vec_svbcast(beta);

        for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
            int vec_end = min(i + SIMD_LEN, end_idx);

            if (i + SIMD_LEN <= end_idx) { /* Full vector */
                lvector double x_vec, a_vec;
                vector_load(&x[i], &x_vec, VEC_BYTES);
                vector_load(&a[j * n + i], &a_vec, VEC_BYTES);

                /* Compute beta * a[j * n + i] * y[j] */
                lvector double prod_vec = vec_muli(a_vec, y_vec);
                prod_vec = vec_muli(prod_vec, beta_vec);
                /* x_vec += prod_vec */
                x_vec = vec_mula(prod_vec, (lvector double)vec_svbcast(1.0), x_vec);
                vector_store(&x_vec, &x[i], VEC_BYTES);
            } else { /* Remainder */
                for (int ii = i; ii < vec_end; ++ii) {
                    x[ii] += beta * a[j * n + ii] * y[j];
                }
            }
        }
    }

    /* Update x[i] += z[i] */
    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end = min(i + SIMD_LEN, end_idx);

        if (i + SIMD_LEN <= end_idx) { /* Full vector */
            lvector double x_vec, z_vec;
            vector_load(&x[i], &x_vec, VEC_BYTES);
            vector_load(&z[i], &z_vec, VEC_BYTES);

            /* x_vec += z_vec */
            x_vec = vec_mula(z_vec, (lvector double)vec_svbcast(1.0), x_vec);
            vector_store(&x_vec, &x[i], VEC_BYTES);
        } else { /* Remainder */
            for (int ii = i; ii < vec_end; ++ii) {
                x[ii] += z[ii];
            }
        }
    }
}

/*------------------------------------------------------------------*/
/* gemver_kernel3_vec: Update w[i] += alpha * a[i * n + j] * x[j] */
/* Vectorized version of gemver_kernel3 */
/*------------------------------------------------------------------*/
__global__ void gemver_kernel3_vec(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int tid_mod = tid % 24;
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    for (int i = start_idx; i < end_idx; ++i) {
        lvector double sum_vec = (lvector double)vec_svbcast(0.0);

        for (int j = 0; j < n; j += SIMD_LEN) {
            int vec_end = min(j + SIMD_LEN, n);

            if (j + SIMD_LEN <= n) { /* Full vector */
                lvector double a_vec, x_vec;
                vector_load(&a[i * n + j], &a_vec, VEC_BYTES);
                vector_load(&x[j], &x_vec, VEC_BYTES);

                /* Compute alpha * a[i * n + j] * x[j] */
                lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
                lvector double prod_vec = vec_muli(a_vec, x_vec);
                prod_vec = vec_muli(prod_vec, alpha_vec);
                /* Accumulate to sum_vec */
                sum_vec = vec_mula(prod_vec, (lvector double)vec_svbcast(1.0), sum_vec);
            } else { /* Remainder */
                for (int jj = j; jj < vec_end; ++jj) {
                    w[i] += alpha * a[i * n + jj] * x[jj];
                }
            }
        }

        /* Sum across vector lanes to update w[i] */
        vector_store(&sum_vec, tmp_buf[tid_mod], VEC_BYTES);
        double sum = 0.0;
        for (int k = 0; k < SIMD_LEN; ++k) {
            sum += tmp_buf[tid_mod][k];
        }
        w[i] += sum;
    }
}
#include "../GEMVER/kernel_vec.h"//大模型生成的存储文件
#include "../GEMVER/kernel_cache_llm.h"//SM缓存优化文件