#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void covar_kernel1(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int j_start = (m * thread_id) / group_size;     // 计算每个线程负责的起始数据
    int j_end = (m * (thread_id + 1)) / group_size; // 计算每个线程负责的结束数据

    // 保证线程任务在数据范围内均匀分配，处理多余的部分
    if (thread_id == group_size - 1) {
        j_end = m; // 最后一个线程处理剩余的所有数据
    }

    // 计算每个线程负责的任务
    for (int j = j_start; j < j_end; ++j) {
        mean[j] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = j_start; j < j_end; ++j) {
            mean[j] += data[i * m + j];
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        mean[j] /= (double)n;
    }
}

__global__ void covar_kernel2(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算该线程的任务范围
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    // 遍历该线程分配的任务
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m; // 列索引 (对应原始代码中的 j)
        int i = idx / m; // 行索引 (对应原始代码中的 i)

        if (i < n && j < m) {
            data[i * m + j] -= mean[j]; // 原始计算逻辑
        }
    }
}

__global__ void covar_kernel3(int m, int n, double *symmat, double *data)
{
    // 获取线程ID和线程总数
    int thread_id = get_thread_id();   // 当前线程ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的元素数量
    int total_elements = m * m; // 总的协方差矩阵元素数量
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算当前线程需要处理的起始和结束元素索引
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    // 遍历该线程负责的协方差矩阵元素
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m; // 行索引
        int j2 = idx / m; // 列索引

        if (j1 <= j2) // 确保只计算对称矩阵的一半
        {
            symmat[j1 * m + j2] = 0.0;
            for (int i = 0; i < n; ++i) {
                symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
            }
            // 对称矩阵的另一半元素赋值
            symmat[j2 * m + j1] = symmat[j1 * m + j2];
        }
    }
}

#ifdef MINI_DATASET
__global__ void covar_kernel1_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int j_start = (m * thread_id) / group_size;
    int j_end = (thread_id == group_size - 1) ? m : (m * (thread_id + 1)) / group_size;

    CACHEb_INIT(mean, double, &mean[j_start], 0, (j_end - j_start) * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_data;
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_WT(mean, &mean[j - j_start], 0.0);
    }
    for (int i = 0; i < n; i++) {
        for (int j = j_start; j < j_end; ++j) {
            CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
            CACHEb_RD(data, &data[i * m + j], tmp_data);
            tmp_mean += tmp_data;
            CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
        tmp_mean /= (double)n;
        CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
    }
    CACHEb_FLUSH(mean);
    CACHEs_INVALID(data);
}

__global__ void covar_kernel2_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算该线程的任务范围
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEb_INIT(mean, double, mean, 0, _PB_M * sizeof(double));
    CACHEb_INIT(data, double, &data[start_idx], 0, (end_idx - start_idx) * sizeof(double));
    double tmp_mean, tmp_data;

    // 遍历该线程分配的任务
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m; // 列索引 (对应原始代码中的 j)

        CACHEb_RD(mean, &mean[j], tmp_mean);
        CACHEb_RD(data, &data[idx - start_idx], tmp_data);
        tmp_data -= tmp_mean;
        CACHEb_WT(data, &data[idx - start_idx], tmp_data);
    }
    CACHEb_FLUSH(data);
    CACHEb_INVALID(mean);
}

__global__ void covar_kernel3_cache(int m, int n, double *symmat, double *data)
{
    // 获取线程ID和线程总数
    int thread_id = get_thread_id();   // 当前线程ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的元素数量
    int total_elements = m * m; // 总的协方差矩阵元素数量
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算当前线程需要处理的起始和结束元素索引
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(symmat, double, symmat, 0, 15);
    double tmp_symmat1;
    // 遍历该线程负责的协方差矩阵元素
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m; // 行索引
        int j2 = idx / m; // 列索引
        if (j1 <= j2)     // 确保只计算对称矩阵的一半
        {
            tmp_symmat1 = 0.0;
            for (int i = 0; i < n; ++i) {
                tmp_symmat1 += data[i * m + j1] * data[i * m + j2];
            }
            // 对称矩阵的另一半元素赋值
            symmat[j2 * m + j1] = tmp_symmat1;
            CACHEs_WT(symmat, &symmat[j1 * m + j2], tmp_symmat1);
        }
    }
    CACHEs_FLUSH(symmat);
}
#endif

#ifdef SMALL_DATASET
__global__ void covar_kernel1_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int j_start = (m * thread_id) / group_size;
    int j_end = (thread_id == group_size - 1) ? m : (m * (thread_id + 1)) / group_size;

    CACHEb_INIT(mean, double, &mean[j_start], 0, (j_end - j_start) * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 10);
    double tmp_mean, tmp_data;
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_WT(mean, &mean[j - j_start], 0.0);
    }
    for (int i = 0; i < n; i++) {
        for (int j = j_start; j < j_end; ++j) {
            CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
            CACHEb_RD(data, &data[i * m + j], tmp_data);
            tmp_mean += tmp_data;
            CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
        tmp_mean /= (double)n;
        CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
    }
    CACHEb_FLUSH(mean);
    CACHEs_INVALID(data);
}

__global__ void covar_kernel2_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算该线程的任务范围
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEb_INIT(mean, double, mean, 0, _PB_M * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_data;

    // 遍历该线程分配的任务
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m; // 列索引 (对应原始代码中的 j)

        CACHEb_RD(mean, &mean[j], tmp_mean);
        CACHEs_RD(data, &data[idx], tmp_data);
        tmp_data -= tmp_mean;
        CACHEs_WT(data, &data[idx], tmp_data);
    }
    CACHEs_FLUSH(data);
    CACHEb_INVALID(mean);
}

__global__ void covar_kernel3_cache(int m, int n, double *symmat, double *data)
{
    // 获取线程ID和线程总数
    int thread_id = get_thread_id();   // 当前线程ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的元素数量
    int total_elements = m * m; // 总的协方差矩阵元素数量
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算当前线程需要处理的起始和结束元素索引
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(symmat, double, symmat, 0, 15);
    double tmp_symmat1;
    // 遍历该线程负责的协方差矩阵元素
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m; // 行索引
        int j2 = idx / m; // 列索引
        if (j1 <= j2)     // 确保只计算对称矩阵的一半
        {
            tmp_symmat1 = 0.0;
            for (int i = 0; i < n; ++i) {
                tmp_symmat1 += data[i * m + j1] * data[i * m + j2];
            }
            // 对称矩阵的另一半元素赋值
            symmat[j2 * m + j1] = tmp_symmat1;
            CACHEs_WT(symmat, &symmat[j1 * m + j2], tmp_symmat1);
        }
    }
    CACHEs_FLUSH(symmat);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void covar_kernel1_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int j_start = (m * thread_id) / group_size;
    int j_end = (thread_id == group_size - 1) ? m : (m * (thread_id + 1)) / group_size;

    CACHEb_INIT(mean, double, &mean[j_start], 0, (j_end - j_start) * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 10);
    double tmp_mean, tmp_data;
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_WT(mean, &mean[j - j_start], 0.0);
    }
    for (int i = 0; i < n; i++) {
        for (int j = j_start; j < j_end; ++j) {
            CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
            CACHEb_RD(data, &data[i * m + j], tmp_data);
            tmp_mean += tmp_data;
            CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
        tmp_mean /= (double)n;
        CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
    }
    CACHEb_FLUSH(mean);
    CACHEs_INVALID(data);
}
__global__ void covar_kernel2_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算该线程的任务范围
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEb_INIT(mean, double, mean, 0, _PB_M * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_data;

    // 遍历该线程分配的任务
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m; // 列索引 (对应原始代码中的 j)

        CACHEb_RD(mean, &mean[j], tmp_mean);
        CACHEs_RD(data, &data[idx], tmp_data);
        tmp_data -= tmp_mean;
        CACHEs_WT(data, &data[idx], tmp_data);
    }
    CACHEs_FLUSH(data);
    CACHEb_INVALID(mean);
}

__global__ void covar_kernel3_cache(int m, int n, double *symmat, double *data)
{
    // 获取线程ID和线程总数
    int thread_id = get_thread_id();   // 当前线程ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的元素数量
    int total_elements = m * m; // 总的协方差矩阵元素数量
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算当前线程需要处理的起始和结束元素索引
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(symmat, double, symmat, 0, 15);
    double tmp_symmat1;
    // 遍历该线程负责的协方差矩阵元素
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m; // 行索引
        int j2 = idx / m; // 列索引
        if (j1 <= j2)     // 确保只计算对称矩阵的一半
        {
            tmp_symmat1 = 0.0;
            for (int i = 0; i < n; ++i) {
                tmp_symmat1 += data[i * m + j1] * data[i * m + j2];
            }
            // 对称矩阵的另一半元素赋值
            symmat[j2 * m + j1] = tmp_symmat1;
            CACHEs_WT(symmat, &symmat[j1 * m + j2], tmp_symmat1);
        }
    }
    CACHEs_FLUSH(symmat);
}
#endif

#ifdef LARGE_DATASET
__global__ void covar_kernel1_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int j_start = (m * thread_id) / group_size;
    int j_end = (thread_id == group_size - 1) ? m : (m * (thread_id + 1)) / group_size;

    CACHEb_INIT(mean, double, &mean[j_start], 0, (j_end - j_start) * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 10);
    double tmp_mean, tmp_data;
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_WT(mean, &mean[j - j_start], 0.0);
    }
    for (int i = 0; i < n; i++) {
        for (int j = j_start; j < j_end; ++j) {
            CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
            CACHEb_RD(data, &data[i * m + j], tmp_data);
            tmp_mean += tmp_data;
            CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
        tmp_mean /= (double)n;
        CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
    }
    CACHEb_FLUSH(mean);
    CACHEs_INVALID(data);
}
__global__ void covar_kernel2_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算该线程的任务范围
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEb_INIT(mean, double, mean, 0, _PB_M * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_data;

    // 遍历该线程分配的任务
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m; // 列索引 (对应原始代码中的 j)

        CACHEb_RD(mean, &mean[j], tmp_mean);
        CACHEs_RD(data, &data[idx], tmp_data);
        tmp_data -= tmp_mean;
        CACHEs_WT(data, &data[idx], tmp_data);
    }
    CACHEs_FLUSH(data);
    CACHEb_INVALID(mean);
}

__global__ void covar_kernel3_cache(int m, int n, double *symmat, double *data)
{
    // 获取线程ID和线程总数
    int thread_id = get_thread_id();   // 当前线程ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的元素数量
    int total_elements = m * m; // 总的协方差矩阵元素数量
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算当前线程需要处理的起始和结束元素索引
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(symmat, double, symmat, 0, 15);
    double tmp_symmat1;
    // 遍历该线程负责的协方差矩阵元素
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m; // 行索引
        int j2 = idx / m; // 列索引
        if (j1 <= j2)     // 确保只计算对称矩阵的一半
        {
            tmp_symmat1 = 0.0;
            for (int i = 0; i < n; ++i) {
                tmp_symmat1 += data[i * m + j1] * data[i * m + j2];
            }
            // 对称矩阵的另一半元素赋值
            symmat[j2 * m + j1] = tmp_symmat1;
            CACHEs_WT(symmat, &symmat[j1 * m + j2], tmp_symmat1);
        }
    }
    CACHEs_FLUSH(symmat);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void covar_kernel1_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int j_start = (m * thread_id) / group_size;
    int j_end = (thread_id == group_size - 1) ? m : (m * (thread_id + 1)) / group_size;

    CACHEb_INIT(mean, double, &mean[j_start], 0, (j_end - j_start) * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 10);
    double tmp_mean, tmp_data;
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_WT(mean, &mean[j - j_start], 0.0);
    }
    for (int i = 0; i < n; i++) {
        for (int j = j_start; j < j_end; ++j) {
            CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
            CACHEb_RD(data, &data[i * m + j], tmp_data);
            tmp_mean += tmp_data;
            CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
        }
    }
    for (int j = j_start; j < j_end; ++j) {
        CACHEb_RD(mean, &mean[j - j_start], tmp_mean);
        tmp_mean /= (double)n;
        CACHEb_WT(mean, &mean[j - j_start], tmp_mean);
    }
    CACHEb_FLUSH(mean);
    CACHEs_INVALID(data);
}
__global__ void covar_kernel2_cache(int m, int n, double *mean, double *data)
{
    // 获取当前线程的 ID 和线程组大小
    int thread_id = get_thread_id();   // 线程 ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的数据范围
    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算该线程的任务范围
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEb_INIT(mean, double, mean, 0, _PB_M * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_data;

    // 遍历该线程分配的任务
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m; // 列索引 (对应原始代码中的 j)

        CACHEb_RD(mean, &mean[j], tmp_mean);
        CACHEs_RD(data, &data[idx], tmp_data);
        tmp_data -= tmp_mean;
        CACHEs_WT(data, &data[idx], tmp_data);
    }
    CACHEs_FLUSH(data);
    CACHEb_INVALID(mean);
}

__global__ void covar_kernel3_cache(int m, int n, double *symmat, double *data)
{
    // 获取线程ID和线程总数
    int thread_id = get_thread_id();   // 当前线程ID
    int group_size = get_group_size(); // 线程总数

    // 每个线程负责的元素数量
    int total_elements = m * m; // 总的协方差矩阵元素数量
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算当前线程需要处理的起始和结束元素索引
    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);
    CACHEs_INIT(symmat, double, symmat, 0, 15);
    double tmp_symmat1;
    // 遍历该线程负责的协方差矩阵元素
    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m; // 行索引
        int j2 = idx / m; // 列索引
        if (j1 <= j2)     // 确保只计算对称矩阵的一半
        {
            tmp_symmat1 = 0.0;
            for (int i = 0; i < n; ++i) {
                tmp_symmat1 += data[i * m + j1] * data[i * m + j2];
            }
            // 对称矩阵的另一半元素赋值
            symmat[j2 * m + j1] = tmp_symmat1;
            CACHEs_WT(symmat, &symmat[j1 * m + j2], tmp_symmat1);
        }
    }
    CACHEs_FLUSH(symmat);
}
#endif

#define SIMD_LEN  16
#define VEC_BYTES 128

/*------------------------------------------------------------------*/
/*  mean : 与原 scalar covar_kernel1 等价                              */
/*------------------------------------------------------------------*/
__global__ void covar_kernel1_vec(int m, int n,
                                 double * __restrict mean,
                                 const double * __restrict data)
{
    int tid          = get_thread_id();
    int gsize        = get_group_size();

    /* 每个线程负责 [j_start , j_end) 这一段列 */
    int j_start = (m *  tid    ) / gsize;
    int j_end   = (m * (tid+1)) / gsize;
    if (tid == gsize-1) j_end = m;

    /*--------------------------------------------------------------*/
    /* 1. 清零                                                      */
    /*--------------------------------------------------------------*/
    for (int j = j_start; j < j_end; j += SIMD_LEN)
    {
        int vec_end = min(j+SIMD_LEN , j_end);

        if (j + SIMD_LEN <= j_end)                 /* 整块 */
        {
            lvector double zero_vec =
                (lvector double)vec_svbcast(0.0);
            vector_store(&zero_vec , &mean[j] , VEC_BYTES);
        }
        else                                        /* 尾块 */
        {
            for (int jj=j; jj<vec_end; ++jj)
                mean[jj] = 0.0;
        }
    }

    /*--------------------------------------------------------------*/
    /* 2. Σ 累加                                                    */
    /*--------------------------------------------------------------*/
    for (int i = 0; i < n; ++i)
    {
        for (int j = j_start; j < j_end; j += SIMD_LEN)
        {
            int vec_end = min(j+SIMD_LEN , j_end);

            if (j + SIMD_LEN <= j_end)
            {
                lvector double mean_vec , data_vec;
                vector_load(&mean[j]          , &mean_vec , VEC_BYTES);
                vector_load(&data[i*m + j]    , &data_vec , VEC_BYTES);

                /* mean_vec += data_vec */
                mean_vec = vec_mula(data_vec ,
                                    (lvector double)vec_svbcast(1.0),
                                    mean_vec);
                vector_store(&mean_vec , &mean[j] , VEC_BYTES);
            }
            else
            {
                for (int jj=j; jj<vec_end; ++jj)
                    mean[jj] += data[i*m + jj];
            }
        }
    }

    /*--------------------------------------------------------------*/
    /* 3. /= n                                                      */
    /*--------------------------------------------------------------*/
    lvector double n_vec =
        (lvector double)vec_svbcast((double)n);

    for (int j = j_start; j < j_end; j += SIMD_LEN)
    {
        int vec_end = min(j+SIMD_LEN , j_end);

        if (j + SIMD_LEN <= j_end)
        {
            lvector double mean_vec;
            vector_load(&mean[j] , &mean_vec , VEC_BYTES);

            mean_vec = vm_fdivd16(mean_vec , n_vec);
            vector_store(&mean_vec , &mean[j] , VEC_BYTES);
        }
        else
        {
            for (int jj=j; jj<vec_end; ++jj)
                mean[jj] /= (double)n;
        }
    }
}

/*------------------------------------------------------------------*/
/*  reduce : 仅做 data -= mean                                       */
/*------------------------------------------------------------------*/
__global__ void covar_kernel2_vec(int m, int n,
                                   const double * __restrict mean,
                                   double * __restrict data)
{
    int tid          = get_thread_id();
    int gsize        = get_group_size();

    long tot_elem    = (long)m * n;

    long per_thread  = (tot_elem + gsize - 1) / gsize;
    long start_idx   = tid * per_thread;
    long end_idx     = min(start_idx + per_thread , tot_elem);

    for (long idx = start_idx; idx < end_idx; idx += SIMD_LEN)
    {
        long remain   = end_idx - idx;
        int  valid    = (remain >= SIMD_LEN);
        int  j_start  = idx % m;

        if (valid && (j_start + SIMD_LEN <= m))
        {
            lvector double data_vec , mean_vec;

            vector_load(&data[idx] , &data_vec , VEC_BYTES);
            vector_load(&mean[j_start] , &mean_vec , VEC_BYTES);

            /* data_vec -= mean_vec */
            data_vec = vec_mulb(data_vec ,
                                (lvector double)vec_svbcast(1.0),
                                mean_vec);

            vector_store(&data_vec , &data[idx] , VEC_BYTES);
        }
        else          /* 跨行或尾块，用标量处理 */
        {
            long vec_end = idx + min((long)SIMD_LEN , remain);
            for (long t = idx; t < vec_end; ++t)
            {
                int j = t %  m;
                int i = t /  m;
                if (i < n && j < m)
                    data[t] -= mean[j];
            }
        }
    }
}

/*------------------------------------------------------------------*/
/*  covar : Σ data[i][j1] * data[i][j2]  (对称矩阵)                 */
/*------------------------------------------------------------------*/
__gsm__ static double tmp_buf[24][SIMD_LEN];

/*====================================================================*/
/*  向量化协方差计算（最终版，已修复末行漏算）                         */
/*====================================================================*/
__global__ void covar_kernel3_vec(int m, int n,
                                 double *  symmat,
                                 double *  data)
{
    int tid     = get_thread_id();
    int gsize   = get_group_size();
    int tid_mod = tid % 24;

    /* ① 线程任务划分 —— 现在覆盖 0 … m-1 全部行号                   */
    int items_per_th = (m + gsize - 1) / gsize;          /* 修正处 #1 */
    int j1_start     = tid * items_per_th;
    int j1_end       = min(j1_start + items_per_th , m); /* 修正处 #2 */

    for (int j1 = j1_start; j1 < j1_end; ++j1)
    {
        /************* 1. 对角线 (j1 , j1) = Σ data[i][j1]^2 *************/
        double diag = 0.0;
        for (int i = 0; i < n; ++i)
            diag += data[i * m + j1] * data[i * m + j1];
        symmat[j1 * m + j1] = diag;      /* 对角线属自对称 */

        /************* 2. 只在 j1 < m-1 时才有上三角需要计算 *************/
        for (int j2 = j1 + 1; j2 < m; j2 += SIMD_LEN)
        {
            int vec_end = min(j2 + SIMD_LEN , m);

            if (j2 + SIMD_LEN <= m)                    /* 整 16 个 */
            {
                lvector double sum_vec =
                    (lvector double)vec_svbcast(0.0);

                for (int i = 0; i < n; ++i)
                {
                    double a  = data[i * m + j1];
                    lvector double a_vec =
                        (lvector double)vec_svbcast(a);

                    lvector double b_vec;
                    vector_load(&data[i * m + j2] , &b_vec , VEC_BYTES);

                    lvector double prod_vec = vec_muli(a_vec , b_vec);
                    sum_vec = vec_mula(prod_vec,
                                       (lvector double)vec_svbcast(1.0),
                                       sum_vec);
                }
                /* 写上三角 */
                vector_store(&sum_vec , &symmat[j1 * m + j2] , VEC_BYTES);

                /* 写下三角 */
                vector_store(&sum_vec , tmp_buf[tid_mod] , VEC_BYTES);
                for (int k = 0; k < SIMD_LEN; ++k)
                    symmat[(j2 + k) * m + j1] = tmp_buf[tid_mod][k];
            }
            else                                         /* 尾块 */
            {
                for (int jj = j2; jj < vec_end; ++jj)
                {
                    double acc = 0.0;
                    for (int i = 0; i < n; ++i)
                        acc += data[i * m + j1] * data[i * m + jj];

                    symmat[j1 * m + jj] = acc;
                    symmat[jj * m + j1] = acc;
                }
            }
        }
    }
}
#include "../COVAR/kernel_vec.h"//大模型生成的存储文件
#include "../COVAR/kernel_cache_llm.h"//SM缓存优化文件