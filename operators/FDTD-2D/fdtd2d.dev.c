#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void fdtd2d_kernel1(int nx, int ny, int t, double *_fict_, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start == 0) {
        for (int j = 0; j < ny; j++) {
            ey[j] = _fict_[t];
        }
        start += ny;
    }
    for (int idx = start; idx < end; idx++) {
        ey[idx] = ey[idx] - 0.5f * (hz[idx] - hz[idx - ny]);
    }
}

__global__ void fdtd2d_kernel2(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    if (start / ny == 0) {
        start++;
    }
    for (int idx = start; idx < end; idx++) {
        int j = idx % ny;

        if (j > 0) {
            ex[idx] = ex[idx] - 0.5f * (hz[idx] - hz[idx - 1]);
        }
    }
}

__global__ void fdtd2d_kernel3(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        if (i < nx - 1 && j < ny - 1) {
            hz[i * ny + j] =
                hz[i * ny + j] - 0.7f * (ex[i * ny + (j + 1)] - ex[i * ny + j] + ey[(i + 1) * ny + j] - ey[i * ny + j]);
        }
    }
}

#ifdef MINI_DATASET
__global__ void fdtd2d_kernel1_cache(int nx, int ny, int t, double *_fict_, double *ex, double *ey,
                                        double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start == 0) {
        CACHEb_INIT(_fict_, double, &_fict_[t], 0, sizeof(double));
        CACHEb_INIT(ey, double, ey, 0, ny * sizeof(double));
        double tmp_fict_;
        CACHEb_RD(_fict_, &_fict_[0], tmp_fict_);
        for (int j = 0; j < ny; j++) {
            CACHEb_WT(ey, &ey[j], tmp_fict_);
        }
        start += ny;
        CACHEb_FLUSH(ey);
        CACHEb_INVALID(_fict_);
    }
    CACHEb_INIT(ey, double, &ey[start], 0, (end - start) * sizeof(double));
    CACHEb_INIT(hz, double, &hz[start - ny], 0, (end - start + ny) * sizeof(double));
    double tmp_ey, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        CACHEb_RD(ey, &ey[idx - start], tmp_ey);
        CACHEb_RD(hz, &hz[idx - start + ny], tmp_hz1);
        CACHEb_RD(hz, &hz[idx - start], tmp_hz2);
        tmp_ey = tmp_ey - 0.5f * (tmp_hz1 - tmp_hz2);
        CACHEb_WT(ey, &ey[idx - start], tmp_ey);
    }
    CACHEb_FLUSH(ey);
    CACHEb_INVALID(hz);
}

__global__ void fdtd2d_kernel2_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    if (start / ny == 0) {
        start++;
    }
    CACHEb_INIT(ex, double, &ex[start], 0, (end - start) * sizeof(double));
    CACHEb_INIT(hz, double, &hz[start - 1], 0, (end - start + 1) * sizeof(double));
    double tmp_ex, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int j = idx % ny;
        if (j > 0) {
            CACHEb_RD(hz, &hz[idx - start + 1], tmp_hz1);
            CACHEb_RD(hz, &hz[idx - start], tmp_hz2);
            CACHEb_RD(ex, &ex[idx - start], tmp_ex);
            tmp_ex = tmp_ex - 0.5f * (tmp_hz1 - tmp_hz2);
            CACHEb_WT(ex, &ex[idx - start], tmp_ex);
        }
    }
    CACHEb_FLUSH(ex);
    CACHEb_INVALID(hz);
}

__global__ void fdtd2d_kernel3_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    CACHEb_INIT(hz, double, &hz[start], 0, (end - start) * sizeof(double));
    CACHEb_INIT(ex, double, &ex[start], 0, (end - start + 1) * sizeof(double));
    CACHEb_INIT(ey, double, &ey[start], 0, (end - start + ny) * sizeof(double));
    double tmp_ex1, tmp_ex2, tmp_ey1, tmp_ey2, tmp_hz;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        if (i < nx - 1 && j < ny - 1) {
            CACHEb_RD(hz, &hz[i * ny + j - start], tmp_hz);
            CACHEb_RD(ex, &ex[i * ny + j + 1 - start], tmp_ex1);
            CACHEb_RD(ex, &ex[i * ny + j - start], tmp_ex2);
            CACHEb_RD(ey, &ey[(i + 1) * ny + j - start], tmp_ey1);
            CACHEb_RD(ey, &ey[i * ny + j - start], tmp_ey2);
            tmp_hz = tmp_hz - 0.7f * (tmp_ex1 - tmp_ex2 + tmp_ey1 - tmp_ey2);
            CACHEb_WT(hz, &hz[i * ny + j - start], tmp_hz);
        }
    }
    CACHEb_FLUSH(hz);
    CACHEb_INVALID(ex);
    CACHEb_INVALID(ey);
}
#endif

#ifdef SMALL_DATASET
__global__ void fdtd2d_kernel1_cache(int nx, int ny, int t, double *_fict_, double *ex, double *ey,
                                        double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start == 0) {
        CACHEb_INIT(_fict_, double, &_fict_[t], 0, sizeof(double));
        CACHEb_INIT(ey, double, ey, 0, ny * sizeof(double));
        double tmp_fict_;
        CACHEb_RD(_fict_, &_fict_[0], tmp_fict_);
        for (int j = 0; j < ny; j++) {
            CACHEb_WT(ey, &ey[j], tmp_fict_);
        }
        start += ny;
        CACHEb_FLUSH(ey);
        CACHEb_INVALID(_fict_);
    }
    CACHEb_INIT(ey, double, &ey[start], 0, (end - start) * sizeof(double));
    CACHEd_INIT(hz, double, hz, 5, 9);
    double tmp_ey, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        CACHEb_RD(ey, &ey[idx - start], tmp_ey);
        CACHEd_RD(hz, &hz[idx], tmp_hz1);
        CACHEd_RD(hz, &hz[idx - ny], tmp_hz2);
        tmp_ey = tmp_ey - 0.5f * (tmp_hz1 - tmp_hz2);
        CACHEb_WT(ey, &ey[idx - start], tmp_ey);
    }
    CACHEb_FLUSH(ey);
    CACHEd_INVALID(hz);
}

__global__ void fdtd2d_kernel2_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    if (start / ny == 0) {
        start++;
    }
    CACHEb_INIT(ex, double, &ex[start], 0, (end - start) * sizeof(double));
    CACHEs_INIT(hz, double, hz, 0, 14);
    double tmp_ex, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int j = idx % ny;
        if (j > 0) {
            CACHEs_RD(hz, &hz[idx], tmp_hz1);
            CACHEs_RD(hz, &hz[idx - 1], tmp_hz2);
            CACHEb_RD(ex, &ex[idx - start], tmp_ex);
            tmp_ex = tmp_ex - 0.5f * (tmp_hz1 - tmp_hz2);
            CACHEb_WT(ex, &ex[idx - start], tmp_ex);
        }
    }
    CACHEb_FLUSH(ex);
    CACHEs_INVALID(hz);
}

__global__ void fdtd2d_kernel3_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    CACHEb_INIT(hz, double, &hz[start], 0, (end - start) * sizeof(double));
    CACHEs_INIT(ex, double, ex, 0, 14);
    CACHEd_INIT(ey, double, ey, 3, 7);

    double tmp_ex1, tmp_ex2, tmp_ey1, tmp_ey2, tmp_hz;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        if (i < nx - 1 && j < ny - 1) {
            CACHEb_RD(hz, &hz[i * ny + j - start], tmp_hz);
            CACHEs_RD(ex, &ex[i * ny + j + 1], tmp_ex1);
            CACHEs_RD(ex, &ex[i * ny + j], tmp_ex2);
            CACHEd_RD(ey, &ey[(i + 1) * ny + j], tmp_ey1);
            CACHEd_RD(ey, &ey[i * ny + j], tmp_ey2);
            tmp_hz = tmp_hz - 0.7f * (tmp_ex1 - tmp_ex2 + tmp_ey1 - tmp_ey2);
            CACHEb_WT(hz, &hz[i * ny + j - start], tmp_hz);
        }
    }
    CACHEb_FLUSH(hz);
    CACHEs_INVALID(ex);
    CACHEd_INVALID(ey);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void fdtd2d_kernel1_cache(int nx, int ny, int t, double *_fict_, double *ex, double *ey,
                                        double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start == 0) {
        CACHEb_INIT(_fict_, double, &_fict_[t], 0, sizeof(double));
        CACHEs_INIT(ey, double, ey, 0, 15);
        double tmp_fict_;
        CACHEb_RD(_fict_, &_fict_[0], tmp_fict_);
        for (int j = 0; j < ny; j++) {
            CACHEs_WT(ey, &ey[j], tmp_fict_);
        }
        start += ny;
        CACHEs_FLUSH(ey);
        CACHEb_INVALID(_fict_);
    }
    CACHEs_INIT(ey, double, ey, 0, 15);
    double tmp_ey;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        CACHEs_RD(ey, &ey[idx], tmp_ey);
        tmp_ey = tmp_ey - 0.5f * (hz[idx] - hz[idx - ny]);
        CACHEs_WT(ey, &ey[idx], tmp_ey);
    }
    CACHEs_FLUSH(ey);
}

__global__ void fdtd2d_kernel2_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    if (start / ny == 0) {
        start++;
    }
    CACHEs_INIT(ex, double, ex, 0, 15);
    CACHEs_INIT(hz, double, hz, 0, 14);
    double tmp_ex, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int j = idx % ny;
        if (j > 0) {
            CACHEs_RD(hz, &hz[idx], tmp_hz1);
            CACHEs_RD(hz, &hz[idx - 1], tmp_hz2);
            CACHEs_RD(ex, &ex[idx], tmp_ex);
            tmp_ex = tmp_ex - 0.5f * (tmp_hz1 - tmp_hz2);
            CACHEs_WT(ex, &ex[idx], tmp_ex);
        }
    }
    CACHEs_FLUSH(ex);
    CACHEs_INVALID(hz);
}

__global__ void fdtd2d_kernel3_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    CACHEs_INIT(hz, double, hz, 0, 15);
    CACHEs_INIT(ex, double, ex, 0, 14);

    double tmp_ex1, tmp_ex2, tmp_ey1, tmp_ey2, tmp_hz;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        if (i < nx - 1 && j < ny - 1) {
            CACHEs_RD(hz, &hz[i * ny + j], tmp_hz);
            CACHEs_RD(ex, &ex[i * ny + j], tmp_ex2);
            CACHEs_RD(ex, &ex[i * ny + j + 1], tmp_ex1);
            tmp_hz = tmp_hz - 0.7f * (tmp_ex1 - tmp_ex2 + ey[(i + 1) * ny + j] - ey[i * ny + j]);
            CACHEs_WT(hz, &hz[i * ny + j], tmp_hz);
        }
    }
    CACHEs_FLUSH(hz);
    CACHEs_INVALID(ex);
}
#endif

#ifdef LARGE_DATASET
__global__ void fdtd2d_kernel1_cache(int nx, int ny, int t, double *_fict_, double *ex, double *ey,
                                        double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start == 0) {
        CACHEb_INIT(_fict_, double, &_fict_[t], 0, sizeof(double));
        CACHEs_INIT(ey, double, ey, 0, 15);
        double tmp_fict_;
        CACHEb_RD(_fict_, &_fict_[0], tmp_fict_);
        for (int j = 0; j < ny; j++) {
            CACHEs_WT(ey, &ey[j], tmp_fict_);
        }
        start += ny;
        CACHEs_FLUSH(ey);
        CACHEb_INVALID(_fict_);
    }
    CACHEs_INIT(ey, double, ey, 0, 15);
    double tmp_ey;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        CACHEs_RD(ey, &ey[idx], tmp_ey);
        tmp_ey = tmp_ey - 0.5f * (hz[idx] - hz[idx - ny]);
        CACHEs_WT(ey, &ey[idx], tmp_ey);
    }
    CACHEs_FLUSH(ey);
}

__global__ void fdtd2d_kernel2_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    if (start / ny == 0) {
        start++;
    }
    CACHEs_INIT(ex, double, ex, 0, 15);
    CACHEs_INIT(hz, double, hz, 0, 14);
    double tmp_ex, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int j = idx % ny;
        if (j > 0) {
            CACHEs_RD(hz, &hz[idx], tmp_hz1);
            CACHEs_RD(hz, &hz[idx - 1], tmp_hz2);
            CACHEs_RD(ex, &ex[idx], tmp_ex);
            tmp_ex = tmp_ex - 0.5f * (tmp_hz1 - tmp_hz2);
            CACHEs_WT(ex, &ex[idx], tmp_ex);
        }
    }
    CACHEs_FLUSH(ex);
    CACHEs_INVALID(hz);
}

__global__ void fdtd2d_kernel3_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    CACHEs_INIT(hz, double, hz, 0, 15);
    CACHEs_INIT(ex, double, ex, 0, 14);

    double tmp_ex1, tmp_ex2, tmp_ey1, tmp_ey2, tmp_hz;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        if (i < nx - 1 && j < ny - 1) {
            CACHEs_RD(hz, &hz[i * ny + j], tmp_hz);
            CACHEs_RD(ex, &ex[i * ny + j], tmp_ex2);
            CACHEs_RD(ex, &ex[i * ny + j + 1], tmp_ex1);
            tmp_hz = tmp_hz - 0.7f * (tmp_ex1 - tmp_ex2 + ey[(i + 1) * ny + j] - ey[i * ny + j]);
            CACHEs_WT(hz, &hz[i * ny + j], tmp_hz);
        }
    }
    CACHEs_FLUSH(hz);
    CACHEs_INVALID(ex);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void fdtd2d_kernel1_cache(int nx, int ny, int t, double *_fict_, double *ex, double *ey,
                                        double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start == 0) {
        CACHEb_INIT(_fict_, double, &_fict_[t], 0, sizeof(double));
        CACHEs_INIT(ey, double, ey, 0, 15);
        double tmp_fict_;
        CACHEb_RD(_fict_, &_fict_[0], tmp_fict_);
        for (int j = 0; j < ny; j++) {
            CACHEs_WT(ey, &ey[j], tmp_fict_);
        }
        start += ny;
        CACHEs_FLUSH(ey);
        CACHEb_INVALID(_fict_);
    }
    CACHEs_INIT(ey, double, ey, 0, 15);
    double tmp_ey;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        CACHEs_RD(ey, &ey[idx], tmp_ey);
        tmp_ey = tmp_ey - 0.5f * (hz[idx] - hz[idx - ny]);
        CACHEs_WT(ey, &ey[idx], tmp_ey);
    }
    CACHEs_FLUSH(ey);
}

__global__ void fdtd2d_kernel2_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    if (start / ny == 0) {
        start++;
    }
    CACHEs_INIT(ex, double, ex, 0, 15);
    CACHEs_INIT(hz, double, hz, 0, 14);
    double tmp_ex, tmp_hz1, tmp_hz2;
    for (int idx = start; idx < end; idx++) {
        int j = idx % ny;
        if (j > 0) {
            CACHEs_RD(hz, &hz[idx], tmp_hz1);
            CACHEs_RD(hz, &hz[idx - 1], tmp_hz2);
            CACHEs_RD(ex, &ex[idx], tmp_ex);
            tmp_ex = tmp_ex - 0.5f * (tmp_hz1 - tmp_hz2);
            CACHEs_WT(ex, &ex[idx], tmp_ex);
        }
    }
    CACHEs_FLUSH(ex);
    CACHEs_INVALID(hz);
}

__global__ void fdtd2d_kernel3_cache(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;

    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);
    CACHEs_INIT(hz, double, hz, 0, 15);
    CACHEs_INIT(ex, double, ex, 0, 14);

    double tmp_ex1, tmp_ex2, tmp_ey1, tmp_ey2, tmp_hz;
    for (int idx = start; idx < end; idx++) {
        int i = idx / ny;
        int j = idx % ny;
        if (i < nx - 1 && j < ny - 1) {
            CACHEs_RD(hz, &hz[i * ny + j], tmp_hz);
            CACHEs_RD(ex, &ex[i * ny + j], tmp_ex2);
            CACHEs_RD(ex, &ex[i * ny + j + 1], tmp_ex1);
            tmp_hz = tmp_hz - 0.7f * (tmp_ex1 - tmp_ex2 + ey[(i + 1) * ny + j] - ey[i * ny + j]);
            CACHEs_WT(hz, &hz[i * ny + j], tmp_hz);
        }
    }
    CACHEs_FLUSH(hz);
    CACHEs_INVALID(ex);
}
#endif
#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double tmp_buf[24][SIMD_LEN];

/*------------------------------------------------------------------*/
/* fdtd2d_kernel1_vec: Initialize ey[0:ny] = _fict_[t] and update ey */
/* Vectorized version of fdtd2d_kernel1 */
/*------------------------------------------------------------------*/
__global__ void fdtd2d_kernel1_vec(int nx, int ny, int t, double *_fict_, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int tid_mod = thread_id % 24;

    int total_elements = nx * ny;
    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    /* Initialize ey[j] = _fict_[t] for j = 0 to ny-1 (thread 0 only) */
    if (thread_id == 0) {
        lvector double fict_vec = (lvector double)vec_svbcast(_fict_[t]);
        for (int j = 0; j < ny; j += SIMD_LEN) {
            int vec_end = min(j + SIMD_LEN, ny);

            if (j + SIMD_LEN <= ny) { /* Full vector */
                vector_store(&fict_vec, &ey[j], VEC_BYTES);
            } else { /* Remainder */
                vector_store(&fict_vec, tmp_buf[tid_mod], VEC_BYTES);
                for (int jj = j; jj < vec_end; ++jj) {
                    ey[jj] = tmp_buf[tid_mod][jj - j];
                }
            }
        }
        start += ny;
    }

    /* Update ey[idx] -= 0.5 * (hz[idx] - hz[idx - ny]) */
    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end = min(idx + SIMD_LEN, end);

        if (idx + SIMD_LEN <= end && (idx % ny) + SIMD_LEN <= ny) { /* Full vector, aligned */
            lvector double ey_vec, hz_vec, hz_prev_vec;
            vector_load(&ey[idx], &ey_vec, VEC_BYTES);
            vector_load(&hz[idx], &hz_vec, VEC_BYTES);
            vector_load(&hz[idx - ny], &hz_prev_vec, VEC_BYTES);

            /* Compute hz[idx] - hz[idx - ny] */
            lvector double diff_vec = vec_mulb(hz_vec, (lvector double)vec_svbcast(1.0), hz_prev_vec);
            /* Multiply by 0.5 */
            lvector double half_vec = (lvector double)vec_svbcast(0.5);
            lvector double prod_vec = vec_muli(diff_vec, half_vec);
            /* ey_vec -= prod_vec */
            ey_vec = vec_mulb(ey_vec, (lvector double)vec_svbcast(1.0), prod_vec);
            vector_store(&ey_vec, &ey[idx], VEC_BYTES);
        } else { /* Remainder or non-aligned */
            for (int ii = idx; ii < vec_end; ++ii) {
                ey[ii] -= 0.5 * (hz[ii] - hz[ii - ny]);
            }
        }
    }
}

/*------------------------------------------------------------------*/
/* fdtd2d_kernel2_vec: Update ex[idx] for j > 0 */
/* Vectorized version of fdtd2d_kernel2 */
/*------------------------------------------------------------------*/
__global__ void fdtd2d_kernel2_vec(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;
    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    if (start / ny == 0) {
        start++;
    }

    /* Update ex[idx] -= 0.5 * (hz[idx] - hz[idx - 1]) for j > 0 */
    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end = min(idx + SIMD_LEN, end);

        if (idx + SIMD_LEN <= end && (idx % ny) > 0 && (idx % ny) + SIMD_LEN <= ny) { /* Full vector, j > 0, aligned */
            lvector double ex_vec, hz_vec, hz_prev_vec;
            vector_load(&ex[idx], &ex_vec, VEC_BYTES);
            vector_load(&hz[idx], &hz_vec, VEC_BYTES);
            vector_load(&hz[idx - 1], &hz_prev_vec, VEC_BYTES);

            /* Compute hz[idx] - hz[idx - 1] */
            lvector double diff_vec = vec_mulb(hz_vec, (lvector double)vec_svbcast(1.0), hz_prev_vec);
            /* Multiply by 0.5 */
            lvector double half_vec = (lvector double)vec_svbcast(0.5);
            lvector double prod_vec = vec_muli(diff_vec, half_vec);
            /* ex_vec -= prod_vec */
            ex_vec = vec_mulb(ex_vec, (lvector double)vec_svbcast(1.0), prod_vec);
            vector_store(&ex_vec, &ex[idx], VEC_BYTES);
        } else { /* Remainder or j == 0 */
            for (int ii = idx; ii < vec_end; ++ii) {
                int j = ii % ny;
                if (j > 0) {
                    ex[ii] -= 0.5 * (hz[ii] - hz[ii - 1]);
                }
            }
        }
    }
}

/*------------------------------------------------------------------*/
/* fdtd2d_kernel3_vec: Update hz[i * ny + j] */
/* Vectorized version of fdtd2d_kernel3 */
/*------------------------------------------------------------------*/
__global__ void fdtd2d_kernel3_vec(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;
    int start = thread_id * (total_elements / num_threads);
    int end = (thread_id == num_threads - 1) ? total_elements : (thread_id + 1) * (total_elements / num_threads);

    /* Update hz[i * ny + j] -= 0.7 * (ex[i * ny + (j + 1)] - ex[i * ny + j] + ey[(i + 1) * ny + j] - ey[i * ny + j]) */
    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end = min(idx + SIMD_LEN, end);

        if (idx + SIMD_LEN <= end && (idx % ny) + SIMD_LEN <= ny - 1 && (idx / ny) < nx - 1) { /* Full vector, within bounds */
            lvector double hz_vec, ex_vec, ex_next_vec, ey_vec, ey_next_vec;
            vector_load(&hz[idx], &hz_vec, VEC_BYTES);
            vector_load(&ex[idx], &ex_vec, VEC_BYTES);
            vector_load(&ex[idx + 1], &ex_next_vec, VEC_BYTES);
            vector_load(&ey[idx], &ey_vec, VEC_BYTES);
            vector_load(&ey[idx + ny], &ey_next_vec, VEC_BYTES);

            /* Compute ex[i * ny + (j + 1)] - ex[i * ny + j] */
            lvector double ex_diff_vec = vec_mulb(ex_next_vec, (lvector double)vec_svbcast(1.0), ex_vec);
            /* Compute ey[(i + 1) * ny + j] - ey[i * ny + j] */
            lvector double ey_diff_vec = vec_mulb(ey_next_vec, (lvector double)vec_svbcast(1.0), ey_vec);
            /* Sum differences */
            lvector double sum_diff_vec = vec_mula(ex_diff_vec, (lvector double)vec_svbcast(1.0), ey_diff_vec);
            /* Multiply by 0.7 */
            lvector double const_vec = (lvector double)vec_svbcast(0.7);
            lvector double prod_vec = vec_muli(sum_diff_vec, const_vec);
            /* hz_vec -= prod_vec */
            hz_vec = vec_mulb(hz_vec, (lvector double)vec_svbcast(1.0), prod_vec);
            vector_store(&hz_vec, &hz[idx], VEC_BYTES);
        } else { /* Remainder or out of bounds */
            for (int ii = idx; ii < vec_end; ++ii) {
                int i = ii / ny;
                int j = ii % ny;
                if (i < nx - 1 && j < ny - 1) {
                    hz[i * ny + j] -= 0.7 * (ex[i * ny + (j + 1)] - ex[i * ny + j] + ey[(i + 1) * ny + j] - ey[i * ny + j]);
                }
            }
        }
    }
}
#include "../FDTD-2D/kernel_vec.h"//大模型生成的存储文件
#include "../FDTD-2D/kernel_cache_llm.h"//SM缓存优化文件