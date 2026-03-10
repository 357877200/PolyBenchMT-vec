#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void jacobi1D_kernel1(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    // 计算结束索引
    int end_idx;
    if (thread_id < remainder) {
        end_idx = start_idx + elements_per_thread;
    } else {
        end_idx = start_idx + elements_per_thread - 1;
    }

    // 确保不超过数组边界
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);
    // 执行计算
    for (int i = start_idx; i <= end_idx; i++) {
        B[i] = 0.33333f * (A[i - 1] + A[i] + A[i + 1]);
    }
    return;
}

__global__ void jacobi1D_kernel2(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    // 计算结束索引
    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;

    // 确保不超过数组边界
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    // 执行计算
    for (int i = start_idx; i <= end_idx; i++) {
        A[i] = B[i];
    }

    return;
}

__global__ void jacobi1D_kernel1_cache(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    // 计算结束索引
    int end_idx;
    if (thread_id < remainder) {
        end_idx = start_idx + elements_per_thread;
    } else {
        end_idx = start_idx + elements_per_thread - 1;
    }

    // 确保不超过数组边界
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    // CACHEb_INIT(A, double, &A[start_idx - 1], 0, 692);
    // CACHEb_INIT(B, double, &B[start_idx], 0, 684);
    CACHEb_INIT(A, double, &A[start_idx - 1], 0, (end_idx - start_idx + 3) * sizeof(double));
    CACHEb_INIT(B, double, &B[start_idx], 0, (end_idx - start_idx + 1) * sizeof(double));
    double tmp_A1, tmp_A2, tmp_A3, tmp_B;
    for (int i = start_idx; i <= end_idx; i++) {
        CACHEb_RD(A, &A[i - start_idx], tmp_A1);
        CACHEb_RD(A, &A[i - start_idx + 1], tmp_A2);
        CACHEb_RD(A, &A[i - start_idx + 2], tmp_A3);
        tmp_B = 0.3333333f * (tmp_A1 + tmp_A2 + tmp_A3);
        CACHEb_WT(B, &B[i - start_idx], tmp_B);
    }
    CACHEb_INVALID(A);
    CACHEb_FLUSH(B);
    return;
}

__global__ void jacobi1D_kernel2_cache(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    // 计算结束索引
    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;

    // 确保不超过数组边界
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    CACHEb_INIT(A, double, &A[start_idx], 0, (end_idx - start_idx + 1) * sizeof(double));
    CACHEb_INIT(B, double, &B[start_idx], 0, (end_idx - start_idx + 1) * sizeof(double));
    double tmp_A, tmp_B;
    for (int i = start_idx; i <= end_idx; i++) {
        CACHEb_RD(B, &B[i - start_idx], tmp_B);
        tmp_A = tmp_B;
        CACHEb_WT(A, &A[i - start_idx], tmp_A);
    }
    CACHEb_INVALID(B);
    CACHEb_FLUSH(A);
    return;
}

#define SIMD_LEN 16
#define VEC_BYTES 128
// 使用乘加代替加法
__global__ void jacobi1D_kernel1_vec(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    /* 检查线程安全性 */
    if (total_threads > 24) return; /* 防止 __gsm__ 数组溢出 */

    /* 计算每个线程的任务分配 */
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    /* 广播系数 0.33333 为向量 */
    lvector double v_coeff = (lvector double)vec_svbcast(0.33333);
    lvector double v_one = (lvector double)vec_svbcast(1.0);

    /* 分配向量缓冲区：左、中、右三个向量 */
    lvector double *buf = (lvector double *)vector_malloc(sizeof(lvector double) * 3);
    lvector double *v_left = buf + 0;  /* A[i-1:i-1+SIMD_LEN] */
    lvector double *v_mid = buf + 1;   /* A[i:i+SIMD_LEN] */
    lvector double *v_right = buf + 2; /* A[i+1:i+1+SIMD_LEN] */


    /* 主循环：处理每个 i */
    for (int i = start_idx; i <= end_idx; ) {
        int remain_elements = end_idx - i + 1;

        if (remain_elements >= SIMD_LEN) {
            /* 向量化处理 SIMD_LEN 个元素 */
            size_t off_left = i - 1;
            size_t off_mid = i;
            size_t off_right = i + 1;

            /* 加载 A[i-1:i-1+SIMD_LEN], A[i:i+SIMD_LEN], A[i+1:i+1+SIMD_LEN] */
            vector_load(A + off_left, v_left, VEC_BYTES);
            vector_load(A + off_mid, v_mid, VEC_BYTES);
            vector_load(A + off_right, v_right, VEC_BYTES);

            /* 计算 B[i:i+SIMD_LEN] = 0.33333 * (A[i-1:i-1+SIMD_LEN] + A[i:i+SIMD_LEN] + A[i+1:i+1+SIMD_LEN]) */
            lvector double v_sum = (lvector double)vec_svbcast(0.0);
            v_sum = vec_mula(*v_left, v_one, v_sum);  /* v_sum += v_left */
            v_sum = vec_mula(*v_mid, v_one, v_sum);   /* v_sum += v_mid */
            v_sum = vec_mula(*v_right, v_one, v_sum); /* v_sum += v_right */
            v_sum = vec_muli(v_coeff, v_sum);         /* v_sum *= 0.33333 */

            /* 写回 B[i:i+SIMD_LEN] */
            vector_store(&v_sum, B + i, VEC_BYTES);

            i += SIMD_LEN; /* 跳过 SIMD_LEN 个元素 */
        } else {
            /* 标量尾部处理 */
            for (; i <= end_idx; i++) {
                B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
            }
        }
    }

    vector_free(buf);
}
// 使用vsip库,提供double类型的向量加法
__global__ void jacobi1D_kernel1_vec_vsip(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    /* 检查线程安全性 */
    if (total_threads > 24) return; /* 防止 __gsm__ 数组溢出 */

    /* 计算每个线程的任务分配 */
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    /* 广播系数 0.33333 为向量 */
    lvector double v_coeff = (lvector double)vec_svbcast(0.33333);

    /* 分配向量缓冲区：左、中、右三个向量 */
    lvector double *buf = (lvector double *)vector_malloc(sizeof(lvector double) * 3);
    lvector double *v_left = buf + 0;  /* A[i-1:i-1+SIMD_LEN] */
    lvector double *v_mid = buf + 1;   /* A[i:i+SIMD_LEN] */
    lvector double *v_right = buf + 2; /* A[i+1:i+1+SIMD_LEN] */

    lvector double * v_sum1 = vector_malloc(sizeof(lvector double));
    /* 主循环：处理每个 i */
    for (int i = start_idx; i <= end_idx; ) {
        int remain_elements = end_idx - i + 1;

        if (remain_elements >= SIMD_LEN) {
            /* 向量化处理 SIMD_LEN 个元素 */
            size_t off_left = i - 1;
            size_t off_mid = i;
            size_t off_right = i + 1;

            /* 加载 A[i-1:i-1+SIMD_LEN], A[i:i+SIMD_LEN], A[i+1:i+1+SIMD_LEN] */
            vector_load(A + off_left, v_left, VEC_BYTES);
            vector_load(A + off_mid, v_mid, VEC_BYTES);
            vector_load(A + off_right, v_right, VEC_BYTES);

            /* 计算 B[i:i+SIMD_LEN] = 0.33333 * (A[i-1:i-1+SIMD_LEN] + A[i:i+SIMD_LEN] + A[i+1:i+1+SIMD_LEN]) */
            
            vec_st((lvector double)vec_svbcast(0.0), 0, v_sum1);
            vsip_vadd_d_v(v_sum1,v_left,v_sum1,16);
            vsip_vadd_d_v(v_sum1,v_mid,v_sum1,16);
            vsip_vadd_d_v(v_sum1,v_right,v_sum1,16);
            lvector double v_sum = vec_muli(v_coeff, *v_sum1);         /* v_sum *= 0.33333 */

            /* 写回 B[i:i+SIMD_LEN] */
            vector_store(&v_sum, B + i, VEC_BYTES);

            i += SIMD_LEN; /* 跳过 SIMD_LEN 个元素 */
        } else {
            /* 标量尾部处理 */
            for (; i <= end_idx; i++) {
                B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
            }
        }
    }

    vector_free(buf);
}

__global__ void jacobi1D_kernel2_vec(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    /* 检查线程安全性 */
    if (total_threads > 24) return; /* 防止 __gsm__ 数组溢出 */

    /* 计算每个线程的任务分配 */
    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    /* 分配向量缓冲区 */
    lvector double *v_buf = (lvector double *)vector_malloc(sizeof(lvector double));

    /* 主循环：处理每个 i */
    for (int i = start_idx; i <= end_idx; ) {
        int remain_elements = end_idx - i + 1;

        if (remain_elements >= SIMD_LEN) {
            /* 向量化复制 SIMD_LEN 个元素 */
            vector_load(B + i, v_buf, VEC_BYTES);
            vector_store(v_buf, A + i, VEC_BYTES);

            i += SIMD_LEN; /* 跳过 SIMD_LEN 个元素 */
        } else {
            /* 标量尾部处理 */
            for (; i <= end_idx; i++) {
                A[i] = B[i];
            }
        }
    }

    vector_free(v_buf);
}
#include "../JACOBI1D/kernel_vec.h"//大模型生成的存储文件
#include "../JACOBI1D/kernel_cache_llm.h"//SM缓存优化文件