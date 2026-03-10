#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void atax_kernel1(int nx, int ny, double *A, double *x, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = nx;                               // 任务总数（每个线程处理一行的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    // 处理分配给当前线程的任务范围
    for (int i = start_idx; i < end_idx; i++) {
        tmp[i] = 0;
        for (int j = 0; j < ny; j++) {
            tmp[i] += A[i * ny + j] * x[j];
        }
    }
}

__global__ void atax_kernel2(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    for (int j = start_idx; j < end_idx; j++) {
        y[j] = 0;
    }

    for (int i = 0; i < nx; i++) {
        for (int j = start_idx; j < end_idx; j++) {
            y[j] += A[i * ny + j] * tmp[i];
        }
    }
}

#ifdef MINI_DATASET
__global__ void atax_kernel1_cache(int nx, int ny, double *A, double *x, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = nx;                               // 任务总数（每个线程处理一行的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(tmp, double, &tmp[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(x, double, x, 0, ny * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_x, tmp_tmp;
    // 处理分配给当前线程的任务范围
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(tmp, &tmp[i - start_idx], tmp_tmp);
        tmp_tmp = 0;
        for (int j = 0; j < ny; j++) {
            CACHEb_RD(x, &x[j], tmp_x);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_tmp += tmp_A * tmp_x;
        }
        CACHEb_WT(tmp, &tmp[i - start_idx], tmp_tmp);
    }
    CACHEb_FLUSH(tmp);
    CACHEs_INVALID(A);
    CACHEb_INVALID(x);
}

__global__ void atax_kernel2_cache(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(y, double, &y[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(tmp, double, tmp, 0, nx * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_A, tmp_tmp, tmp_y;
    for (int j = start_idx; j < end_idx; j++) {
        CACHEb_WT(y, &y[j - start_idx], 0);
    }

    for (int i = 0; i < nx; i++) {
        CACHEb_RD(tmp, &tmp[i], tmp_tmp);
        for (int j = start_idx; j < end_idx; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(y, &y[j - start_idx], tmp_y);
            tmp_y += tmp_A * tmp_tmp;
            CACHEb_WT(y, &y[j - start_idx], tmp_y);
        }
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEb_INVALID(tmp);
}
#endif

#ifdef SMALL_DATASET
__global__ void atax_kernel1_cache(int nx, int ny, double *A, double *x, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = nx;                               // 任务总数（每个线程处理一行的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(tmp, double, &tmp[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(x, double, x, 0, ny * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_A, tmp_x, tmp_tmp;
    // 处理分配给当前线程的任务范围
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(tmp, &tmp[i - start_idx], tmp_tmp);
        tmp_tmp = 0;
        for (int j = 0; j < ny; j++) {
            CACHEb_RD(x, &x[j], tmp_x);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_tmp += tmp_A * tmp_x;
        }
        CACHEb_WT(tmp, &tmp[i - start_idx], tmp_tmp);
    }
    CACHEb_FLUSH(tmp);
    CACHEs_INVALID(A);
    CACHEb_INVALID(x);
}

__global__ void atax_kernel2_cache(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(y, double, &y[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEb_INIT(tmp, double, tmp, 0, nx * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_A, tmp_tmp, tmp_y;
    for (int j = start_idx; j < end_idx; j++) {
        CACHEb_WT(y, &y[j - start_idx], 0);
    }

    for (int i = 0; i < nx; i++) {
        CACHEb_RD(tmp, &tmp[i], tmp_tmp);
        for (int j = start_idx; j < end_idx; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(y, &y[j - start_idx], tmp_y);
            tmp_y += tmp_A * tmp_tmp;
            CACHEb_WT(y, &y[j - start_idx], tmp_y);
        }
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEb_INVALID(tmp);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void atax_kernel1_cache(int nx, int ny, double *A, double *x, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = nx;                               // 任务总数（每个线程处理一行的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(tmp, double, &tmp[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(x, double, x, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_A, tmp_x, tmp_tmp;
    // 处理分配给当前线程的任务范围
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(tmp, &tmp[i - start_idx], tmp_tmp);
        tmp_tmp = 0;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(x, &x[j], tmp_x);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_tmp += tmp_A * tmp_x;
        }
        CACHEb_WT(tmp, &tmp[i - start_idx], tmp_tmp);
    }
    CACHEb_FLUSH(tmp);
    CACHEs_INVALID(A);
    CACHEs_INVALID(x);
}

__global__ void atax_kernel2_cache(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(y, double, &y[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(tmp, double, tmp, 0, 15);
    CACHEs_INIT(A, double, A, 0, 12);
    double tmp_A, tmp_tmp, tmp_y;
    for (int j = start_idx; j < end_idx; j++) {
        CACHEb_WT(y, &y[j - start_idx], 0);
    }

    for (int i = 0; i < nx; i++) {
        CACHEs_RD(tmp, &tmp[i], tmp_tmp);
        for (int j = start_idx; j < end_idx; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(y, &y[j - start_idx], tmp_y);
            tmp_y += tmp_A * tmp_tmp;
            CACHEb_WT(y, &y[j - start_idx], tmp_y);
        }
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(tmp);
}
#endif

#ifdef LARGE_DATASET
__global__ void atax_kernel1_cache(int nx, int ny, double *A, double *x, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = nx;                               // 任务总数（每个线程处理一行的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(tmp, double, &tmp[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(x, double, x, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_A, tmp_x, tmp_tmp;
    // 处理分配给当前线程的任务范围
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(tmp, &tmp[i - start_idx], tmp_tmp);
        tmp_tmp = 0;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(x, &x[j], tmp_x);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_tmp += tmp_A * tmp_x;
        }
        CACHEb_WT(tmp, &tmp[i - start_idx], tmp_tmp);
    }
    CACHEb_FLUSH(tmp);
    CACHEs_INVALID(A);
    CACHEs_INVALID(x);
}

__global__ void atax_kernel2_cache(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(y, double, &y[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(tmp, double, tmp, 0, 15);
    CACHEs_INIT(A, double, A, 0, 12);
    double tmp_A, tmp_tmp, tmp_y;
    for (int j = start_idx; j < end_idx; j++) {
        CACHEb_WT(y, &y[j - start_idx], 0);
    }

    for (int i = 0; i < nx; i++) {
        CACHEs_RD(tmp, &tmp[i], tmp_tmp);
        for (int j = start_idx; j < end_idx; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(y, &y[j - start_idx], tmp_y);
            tmp_y += tmp_A * tmp_tmp;
            CACHEb_WT(y, &y[j - start_idx], tmp_y);
        }
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(tmp);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void atax_kernel1_cache(int nx, int ny, double *A, double *x, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = nx;                               // 任务总数（每个线程处理一行的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(tmp, double, &tmp[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(x, double, x, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_A, tmp_x, tmp_tmp;
    // 处理分配给当前线程的任务范围
    for (int i = start_idx; i < end_idx; i++) {
        CACHEb_RD(tmp, &tmp[i - start_idx], tmp_tmp);
        tmp_tmp = 0;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(x, &x[j], tmp_x);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_tmp += tmp_A * tmp_x;
        }
        CACHEb_WT(tmp, &tmp[i - start_idx], tmp_tmp);
    }
    CACHEb_FLUSH(tmp);
    CACHEs_INVALID(A);
    CACHEs_INVALID(x);
}

__global__ void atax_kernel2_cache(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEb_INIT(y, double, &y[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    CACHEs_INIT(tmp, double, tmp, 0, 15);
    CACHEs_INIT(A, double, A, 0, 12);
    double tmp_A, tmp_tmp, tmp_y;
    for (int j = start_idx; j < end_idx; j++) {
        CACHEb_WT(y, &y[j - start_idx], 0);
    }

    for (int i = 0; i < nx; i++) {
        CACHEs_RD(tmp, &tmp[i], tmp_tmp);
        for (int j = start_idx; j < end_idx; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(y, &y[j - start_idx], tmp_y);
            tmp_y += tmp_A * tmp_tmp;
            CACHEb_WT(y, &y[j - start_idx], tmp_y);
        }
    }
    CACHEb_FLUSH(y);
    CACHEs_INVALID(A);
    CACHEs_INVALID(tmp);
}
#endif

/* 一条向量 16 个 double --------------------------------------------------*/
#define SIMD_LEN  16
#define VEC_BYTES 128
__gsm__ static double temp_array[24][SIMD_LEN];
/*---------------------------------------------------------------------*/
__global__ void atax_kernel1_vec(int nx, int ny,
                                 double *A,      /* A[nx][ny]  */
                                 double *x,      /* x[ny]      */
                                 double *tmp,    /* tmp[nx]    */
                                 uint64_t *before_hot_data,
                                 uint64_t *after_hot_data)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nx;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }
    start_idx = min(start_idx, total_elements);
    end_idx = min(end_idx, total_elements);

    lvector double vzero = (lvector double)vec_svbcast(0.0);
    lvector double vone = (lvector double)vec_svbcast(1.0);
 
    for (int i = start_idx; i < end_idx; i++) {
        int row_offset = i * ny;
        tmp[i] = 0.0;

        int vec_iterations = ny / SIMD_LEN;
        int remainder_cols = ny % SIMD_LEN;

        lvector double sum_vec = vzero;
        for (int v = 0; v < vec_iterations; ++v) {
            int col_offset = v * SIMD_LEN;
            if (col_offset + SIMD_LEN <= ny) {
                lvector double a_vec, x_vec;
                vector_load(&A[row_offset + col_offset], &a_vec, VEC_BYTES);
                vector_load(&x[col_offset], &x_vec, VEC_BYTES);

                lvector double prod_vec = vec_muli(a_vec, x_vec);
                sum_vec = vec_mula(prod_vec, vone, sum_vec);
            }
        }  
        double local_sum = 0.0;
        if (vec_iterations > 0) {
// temp_array数组要用gsm标识符声明，这样才会生成到核外空间，vector_store才能将数据从am缓存搬运过去
            
            vector_store(&sum_vec, temp_array[thread_id], VEC_BYTES);
            for (int k = 0; k < SIMD_LEN; ++k) {
                local_sum += temp_array[thread_id][k];
            }
        }
        for (int j = vec_iterations * SIMD_LEN; j < ny; ++j) {
            local_sum += A[row_offset + j] * x[j];
        }
        tmp[i] = local_sum;
    }
}


__global__ void atax_kernel2_vec(int nx, int ny, double *A, double *y, double *tmp)
{
    // 获取当前线程的线程 ID 和线程总数
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 计算每个线程需要处理的任务范围
    int total_elements = ny;                               // 任务总数（每个线程处理一列的计算）
    int elements_per_thread = total_elements / group_size; // 每个线程处理的元素数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 处理余下的部分
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    // 计算向量化迭代次数和尾部剩余部分
    int total_cols = end_idx - start_idx;
    int vec_iterations = total_cols / SIMD_LEN;
    int remainder_cols = total_cols % SIMD_LEN;

    // 创建一个向量化常量 -1.0 和 0.0
    lvector double vneg1 = (lvector double)vec_svbcast(-1.0);
    lvector double vzero = (lvector double)vec_svbcast(0.0);

    // 先将 y 的值初始化为 0，采用向量化方式
    for (int v = 0; v < vec_iterations; ++v) {
        int col_offset = start_idx + v * SIMD_LEN;
        vector_store(&vzero, &y[col_offset], VEC_BYTES);
    }
    // 处理尾部标量初始化
    for (int j = start_idx + vec_iterations * SIMD_LEN; j < end_idx; ++j) {
        y[j] = 0.0;
    }

    // 外层循环遍历每一行 i
    for (int i = 0; i < nx; ++i) {
        // 加载 tmp[i] 作为广播值
        double tmp_val = tmp[i];
        lvector double tmp_vec = (lvector double)vec_svbcast(tmp_val);

        // 向量化处理
        for (int v = 0; v < vec_iterations; ++v) {
            int col_offset = start_idx + v * SIMD_LEN;

            // 加载 A 和 y 的当前向量
            lvector double a_vec, y_vec;
            vector_load(&A[i * ny + col_offset], &a_vec, VEC_BYTES);
            vector_load(&y[col_offset], &y_vec, VEC_BYTES);

            // 计算 A[i*ny+j] * tmp[i]
            lvector double prod_vec = vec_muli(a_vec, tmp_vec);

            // 计算 y[j] += prod_vec
            // 使用 vec_mula 实现加法: y_new = y_old + prod_vec * 1.0
            lvector double one_vec = (lvector double)vec_svbcast(1.0);
            lvector double new_y_vec = vec_mula(prod_vec, one_vec, y_vec);

            // 存储结果回 y
            vector_store(&new_y_vec, &y[col_offset], VEC_BYTES);
        }

        // 处理尾部标量部分
        for (int j = start_idx + vec_iterations * SIMD_LEN; j < end_idx; ++j) {
            y[j] += A[i * ny + j] * tmp[i];
        }
    }
}
#include "../ATAX/kernel_vec.h"//大模型生成的存储文件
#include "../ATAX/kernel_cache_llm.h"//SM缓存优化文件





