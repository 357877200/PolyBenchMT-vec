#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }
__global__ void deriche_kernel1(int w, int h, double a1, double a2, double b1, double b2,
                            double *imgIn, double *y1)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按列分块
    int i_start = (w * thread_id) / group_size;
    int i_end   = (w * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) i_end = w;

    for (int i = i_start; i < i_end; i++) {
        double ym1 = 0.0;
        double ym2 = 0.0;
        double xm1 = 0.0;
        for (int j = 0; j < h; j++) {
            int idx = i*h + j;
            y1[idx] = a1 * imgIn[idx] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = imgIn[idx];
            ym2 = ym1;
            ym1 = y1[idx];
        }
    }
}

__global__ void deriche_kernel2(int w, int h, double a3, double a4, double b1, double b2,
                            double *imgIn, double *y2)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按列分块
    int i_start = (w * thread_id) / group_size;
    int i_end   = (w * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) i_end = w;

    for (int i = i_start; i < i_end; i++) {
        double yp1 = 0.0, yp2 = 0.0;
        double xp1 = 0.0, xp2 = 0.0;
        for (int j = h-1; j >= 0; j--) {
            int idx = i*h + j;
            y2[idx] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = imgIn[idx];
            yp2 = yp1;
            yp1 = y2[idx];
        }
    }
}

__global__ void deriche_kernel3(int w, int h, double c1,
                            double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按总元素分块
    int total_elements = w * h;
    int elems_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elems_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx   = (thread_id + 1) * elems_per_thread + ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        imgOut[idx] = c1 * (y1[idx] + y2[idx]);
    }
}
