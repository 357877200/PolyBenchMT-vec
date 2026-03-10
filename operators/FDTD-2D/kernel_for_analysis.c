#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

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