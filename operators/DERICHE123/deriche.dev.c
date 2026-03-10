#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

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

__global__ void deriche_kernel4(int w, int h, double a5, double a6, double b1, double b2,
                            double *imgOut, double *y1)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按行分块
    int j_start = (h * thread_id) / group_size;
    int j_end   = (h * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) j_end = h;

    for (int j = j_start; j < j_end; j++) {
        double tm1 = 0.0;
        double ym1 = 0.0, ym2 = 0.0;
        for (int i = 0; i < w; i++) {
            int idx = i*h + j;
            y1[idx] = a5 * imgOut[idx] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = imgOut[idx];
            ym2 = ym1;
            ym1 = y1[idx];
        }
    }
}

__global__ void deriche_kernel5(int w, int h, double a7, double a8,
                             double b1, double b2,
                             double *imgOut, double *y2)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按行分块
    int j_start = (h * thread_id) / group_size;
    int j_end   = (h * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) j_end = h;

    for (int j = j_start; j < j_end; j++) {
        double tp1 = 0.0, tp2 = 0.0;
        double yp1 = 0.0, yp2 = 0.0;
        for (int i = w-1; i >= 0; i--) {
            int idx = i*h + j;
            y2[idx] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;  
            tp1 = imgOut[idx];  
            yp2 = yp1;  
            yp1 = y2[idx];  
        }
    }
}
__global__ void deriche_kernel6(int w, int h, double c2,
                             double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按总元素分块
    int total_elements = w * h;
    int elems_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elems_per_thread +
                    (thread_id < remainder ? thread_id : remainder);
    int end_idx   = (thread_id + 1) * elems_per_thread +
                    ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        imgOut[idx] = c2 * (y1[idx] + y2[idx]);
    }
}

#define SIMD_LEN    16
#define VEC_BYTES   128

/* kernel3 向量实现 */
__global__ void deriche_kernel3_vec(int w, int h, double c1, double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    lvector double *buf_y1   = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2   = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out  = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y1 || !buf_y2 || !buf_out) {
        if (buf_y1) vector_free(buf_y1);
        if (buf_y2) vector_free(buf_y2);
        if (buf_out) vector_free(buf_out);
        return;
    }

    int total_elements = w * h;  
    int elems_per_thread = total_elements / group_size;  
    int remainder = total_elements % group_size;  
    
    int start_idx = thread_id * elems_per_thread + (thread_id < remainder ? thread_id : remainder);  
    int end_idx   = (thread_id + 1) * elems_per_thread + ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);  

    lvector double c1_vec = (lvector double)vec_svbcast(c1);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    int vec_start = start_idx;
    int vec_end   = (end_idx / SIMD_LEN) * SIMD_LEN;

    for (int idx = vec_start; idx < vec_end; idx += SIMD_LEN) {
        vector_load(&y1[idx], buf_y1, VEC_BYTES);
        vector_load(&y2[idx], buf_y2, VEC_BYTES);
        lvector double vy1 = vec_ld(0, buf_y1);
        lvector double vy2 = vec_ld(0, buf_y2);

        lvector double vsum = vec_mula(vy1, one_vec, vy2);
        lvector double vout = vec_muli(c1_vec, vsum);

        vec_st(vout, 0, buf_out);
        vector_store(buf_out, &imgOut[idx], VEC_BYTES);
    }

    for (int idx = vec_end; idx < end_idx; ++idx) {
        imgOut[idx] = c1 * (y1[idx] + y2[idx]);
    }

    vector_free(buf_y1);
    vector_free(buf_y2);
    vector_free(buf_out);
}

/* kernel6 向量实现 */
__global__ void deriche_kernel6_vec(int w, int h, double c2, double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    lvector double *buf_y1   = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2   = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out  = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y1 || !buf_y2 || !buf_out) {
        if (buf_y1) vector_free(buf_y1);
        if (buf_y2) vector_free(buf_y2);
        if (buf_out) vector_free(buf_out);
        return;
    }

    int total_elements = w * h;  
    int elems_per_thread = total_elements / group_size;  
    int remainder = total_elements % group_size;  
    
    int start_idx = thread_id * elems_per_thread + (thread_id < remainder ? thread_id : remainder);  
    int end_idx   = (thread_id + 1) * elems_per_thread + ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);  

    lvector double c2_vec = (lvector double)vec_svbcast(c2);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    int vec_start = start_idx;
    int vec_end   = (end_idx / SIMD_LEN) * SIMD_LEN;

    for (int idx = vec_start; idx < vec_end; idx += SIMD_LEN) {
        vector_load(&y1[idx], buf_y1, VEC_BYTES);
        vector_load(&y2[idx], buf_y2, VEC_BYTES);
        lvector double vy1 = vec_ld(0, buf_y1);
        lvector double vy2 = vec_ld(0, buf_y2);

        lvector double vsum = vec_mula(vy1, one_vec, vy2);
        lvector double vout = vec_muli(c2_vec, vsum);

        vec_st(vout, 0, buf_out);
        vector_store(buf_out, &imgOut[idx], VEC_BYTES);
    }

    for (int idx = vec_end; idx < end_idx; ++idx) {
        imgOut[idx] = c2 * (y1[idx] + y2[idx]);
    }

    vector_free(buf_y1);
    vector_free(buf_y2);
    vector_free(buf_out);
}
#include "../DERICHE123/kernel_vec.h"
#include "../DERICHE123/kernel_cache_llm.h"//SM缓存优化文件