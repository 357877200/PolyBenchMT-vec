#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void durbin_kernel1(int k, int barrier_id, int n ,double *r, double *y, double *z, double *alpha, double *beta)
{
    // 只用一个线程执行所有操作
    int tid = get_thread_id();
    if (tid != 0) return; // 其他线程直接退出

    // 更新 beta
    *beta = (1.0 - (*alpha) * (*alpha)) * (*beta);

    // 局部 sum（顺序计算）
    double sum_total = 0.0;
    for (int i = 0; i < k; i++) {
        sum_total += r[k - i - 1] * y[i];
    }

    // 更新 alpha
    *alpha = -(r[k] + sum_total) / (*beta);

    // 更新 z
    double alpha_val = *alpha;
    for (int i = 0; i < k; i++) {
        z[i] = y[i] + alpha_val * y[k - i - 1];
    }
}

__global__ void durbin_kernel2(int k, int barrier_id,double *y, double *z,double *alpha)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int chunk_size = (k + num_threads - 1) / num_threads;
    int start_i = tid * chunk_size;
    int end_i   = (start_i + chunk_size > k) ? k : start_i + chunk_size;

    // 阶段1：并行拷贝 z → y
    for (int i = start_i; i < end_i; i++) {
        y[i] = z[i];
    }

    // 阶段2：thread0 更新 y[k]
    if (tid == 0) {
        y[k] = *alpha;
    }
}


__global__ void durbin_kernel1_thread(int k, int barrier_id, int n,double *r, double *y, double *z,double *alpha, double *beta)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    // 共享数组，每线程一个槽，用于存局部和
    __gsm__ static double sum_buf[24];

    int chunk_size = (k + num_threads - 1) / num_threads;
    int start_i = tid * chunk_size;
    int end_i   = (start_i + chunk_size > k) ? k : start_i + chunk_size;

    // 更新 beta
    if (tid == 0) {
        *beta = (1.0 - (*alpha) * (*alpha)) * (*beta);
    }
    group_barrier(barrier_id);

    // 局部 sum （线程私有）
    double local_sum = 0.0;
    for (int i = start_i; i < end_i; i++) {
        local_sum += r[k - i - 1] * y[i];
    }

    // 将本线程结果放到共享 sum_buf
    sum_buf[tid] = local_sum;
    group_barrier(barrier_id);

    // 汇总 + 更新 alpha
    if (tid == 0) {
        double sum_total = 0.0;
        for (int t = 0; t < num_threads; t++)
            sum_total += sum_buf[t];
        *alpha = -(r[k] + sum_total) / (*beta);
    }
    group_barrier(barrier_id);

    // 更新 z
    double alpha_val = *alpha;
    for (int i = start_i; i < end_i; i++) {
        z[i] = y[i] + alpha_val * y[k - i - 1];
    }

}



#define SIMD_LEN    16
#define VEC_BYTES   128
__gsm__ static double sum_buf[24];
__gsm__ static double temp_buf_r[24][SIMD_LEN];
__gsm__ static double temp_buf_y[24][SIMD_LEN];

__global__ void durbin_kernel1_vec(int k, int barrier_id, int n,double *r, double *y, double *z,double *alpha, double *beta)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int chunk_size = (k + num_threads - 1) / num_threads;
    int start_i = tid * chunk_size;
    int end_i   = (start_i + chunk_size > k) ? k : start_i + chunk_size;

    // 更新 beta
    if (tid == 0) {
        *beta = (1.0 - (*alpha) * (*alpha)) * (*beta);
    }
    group_barrier(barrier_id);

    double local_sum = 0.0;
    // 局部 sum
    for (int i = start_i; i < end_i; i++) {
        local_sum += r[k - i - 1] * y[i];
    }

    sum_buf[tid] = local_sum;
    group_barrier(barrier_id);

    if (tid == 0) {
        double sum_total = 0.0;
        for (int t = 0; t < num_threads; t++)
            sum_total += sum_buf[t];
        *alpha = -(r[k] + sum_total) / (*beta);
    }
    group_barrier(barrier_id);

    // 更新 z 向量化实现，反序访问通过临时缓冲
    lvector double *buf_y_fwd = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y_rev = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y_fwd || !buf_y_rev) {
        if (buf_y_fwd) vector_free(buf_y_fwd);
        if (buf_y_rev) vector_free(buf_y_rev);
        return;
    }

    double alpha_val = *alpha;
    lvector double alpha_vec = (lvector double)vec_svbcast(alpha_val);
    lvector double one_vec   = (lvector double)vec_svbcast(1.0);

    for (int i = start_i; i + SIMD_LEN <= end_i; i += SIMD_LEN) {
        // forward y[i]
        vector_load(&y[i], buf_y_fwd, VEC_BYTES);

        // reverse y[k - i - 1]
        for (int jj = 0; jj < SIMD_LEN; ++jj) {
            temp_buf_y[tid][jj] = y[k - (i + jj) - 1];
        }
        vector_load(temp_buf_y[tid], buf_y_rev, VEC_BYTES);

        lvector double vy_fwd = vec_ld(0, buf_y_fwd);
        lvector double vy_rev = vec_ld(0, buf_y_rev);

        // alpha * y_rev + y_fwd
        lvector double vprod = vec_muli(alpha_vec, vy_rev);
        lvector double vz = vec_mula(vy_fwd, one_vec, vprod);

        vec_st(vz, 0, buf_y_fwd);
        vector_store(buf_y_fwd, &z[i], VEC_BYTES);
    }

    // 尾部标量
    for (int i = end_i - ((end_i-start_i)%SIMD_LEN); i < end_i; ++i) {
        z[i] = y[i] + alpha_val * y[k - i - 1];
    }

    vector_free(buf_y_fwd);
    vector_free(buf_y_rev);
}

__global__ void durbin_kernel2_vec(int k, int barrier_id,double *y, double *z,double *alpha)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int chunk_size = (k + num_threads - 1) / num_threads;
    int start_i = tid * chunk_size;
    int end_i   = (start_i + chunk_size > k) ? k : start_i + chunk_size;

    lvector double *buf_y = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_z = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y || !buf_z) {
        if (buf_y) vector_free(buf_y);
        if (buf_z) vector_free(buf_z);
        return;
    }

    for (int i = start_i; i + SIMD_LEN <= end_i; i += SIMD_LEN) {
        vector_load(&z[i], buf_z, VEC_BYTES);
        vec_st(vec_ld(0, buf_z), 0, buf_y);
        vector_store(buf_y, &y[i], VEC_BYTES);
    }
    for (int i = end_i - ((end_i-start_i)%SIMD_LEN); i < end_i; ++i) {
        y[i] = z[i];
    }

    if (tid == 0) {
        y[k] = *alpha;
    }

    vector_free(buf_y);
    vector_free(buf_z);
}
#include "../DURBIN/kernel_vec.h"//大模型生成的存储文件
#include "../DURBIN/kernel_cache_llm.h"//SM缓存优化文件