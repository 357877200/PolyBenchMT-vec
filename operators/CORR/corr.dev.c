#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
#define FLOAT_N 3214212.01
#define EPS 0.005

__global__ void corr_kernel1(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 为每个线程分配任务，确保覆盖所有数据
    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);
    for (int j = start_j; j < end_j; j++) {
        mean[j] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = start_j; j < end_j; j++) {
            mean[j] += data[i * m + j];
        }
    }
    for (int j = start_j; j < end_j; j++) {
        mean[j] /= (double)FLOAT_N;
    }
}

__global__ void corr_kernel2(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 为每个线程分配任务
    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);

    for (int j = start_j; j < end_j; j++) {
        std[j] = 0.0;
        for (int i = 0; i < n; i++) {
            std[j] += (data[i * m + j] - mean[j]) * (data[i * m + j] - mean[j]);
        }
        std[j] /= (FLOAT_N);
        std[j] = sqrt(std[j]);
        if (std[j] <= EPS) {
            std[j] = 1.0;
        }
    }
}

__global__ void corr_kernel3(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 此核函数处理二维数据，需要重新分配工作
    // 计算总元素数
    int total_elements = n * m;
    int items_per_thread = (total_elements + total_threads - 1) / total_threads;
    int start_idx = thread_id * items_per_thread;
    int end_idx = min(start_idx + items_per_thread, total_elements);

    for (int idx = start_idx; idx < end_idx; idx++) {
        // 将一维索引转换为二维索引
        int i = idx / m;
        int j = idx % m;

        if ((i < n) && (j < m)) {
            data[i * m + j] -= mean[j];
            data[i * m + j] /= (sqrt(FLOAT_N) * std[j]);
        }
    }
}

__global__ void corr_kernel4(int m, int n, double *symmat, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 注意: 这个核函数处理上三角矩阵，需要特殊处理任务分配
    // 需要计算的元素数量为 (m-1)*m/2
    int j1;

    // 每个线程处理一部分行
    int items_per_thread = ((m - 1) + total_threads - 1) / total_threads;
    int start_j1 = thread_id * items_per_thread;
    int end_j1 = min(start_j1 + items_per_thread, m - 1);

    for (j1 = start_j1; j1 < end_j1; j1++) {
        symmat[j1 * m + j1] = 1.0;

        for (int j2 = (j1 + 1); j2 < m; j2++) {
            symmat[j1 * m + j2] = 0.0;
            for (int i = 0; i < n; i++) {
                symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
            }
            symmat[j2 * m + j1] = symmat[j1 * m + j2];
        }
    }
}

__global__ void corr_kernel1_cache(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 为每个线程分配任务，确保覆盖所有数据
    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);
    CACHEb_INIT(mean, double, &mean[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_data;
    for (int j = start_j; j < end_j; j++) {
        CACHEb_WT(mean, &mean[j - start_j], 0.0);
    }
    for (int i = 0; i < n; i++) {
        for (int j = start_j; j < end_j; j++) {
            CACHEb_RD(mean, &mean[j - start_j], tmp_mean);
            CACHEs_RD(data, &data[i * m + j], tmp_data);
            tmp_mean += tmp_data;
            CACHEb_WT(mean, &mean[j - start_j], tmp_mean);
        }
    }
    for (int j = start_j; j < end_j; j++) {
        CACHEb_RD(mean, &mean[j - start_j], tmp_mean);
        tmp_mean /= (double)FLOAT_N;
        CACHEb_WT(mean, &mean[j - start_j], tmp_mean);
    }
    CACHEb_FLUSH(mean);
    CACHEs_INVALID(data);
}

__global__ void corr_kernel2_cache(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 为每个线程分配任务
    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);
    CACHEb_INIT(std, double, &std[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEb_INIT(mean, double, &mean[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEd_INIT(data, double, data, 5, 10);
    double tmp_std, tmp_mean, tmp_data;
    for (int j = start_j; j < end_j; j++) {
        CACHEb_RD(mean, &mean[j - start_j], tmp_mean);
        tmp_std = 0.0;
        for (int i = 0; i < n; i++) {
            CACHEd_RD(data, &data[i * m + j], tmp_data);
            tmp_std += (tmp_data - tmp_mean) * (tmp_data - tmp_mean);
        }
        tmp_std /= (FLOAT_N);
        tmp_std = sqrt(tmp_std);
        if (tmp_std <= EPS) {
            tmp_std = 1.0;
        }
        CACHEb_WT(std, &std[j - start_j], tmp_std);
    }
    CACHEb_FLUSH(std);
    CACHEb_INVALID(mean);
    CACHEd_INVALID(data);
}

__global__ void corr_kernel3_cache(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 此核函数处理二维数据，需要重新分配工作
    // 计算总元素数
    int total_elements = n * m;
    int items_per_thread = (total_elements + total_threads - 1) / total_threads;
    int start_idx = thread_id * items_per_thread;
    int end_idx = min(start_idx + items_per_thread, total_elements);

    CACHEb_INIT(mean, double, mean, 0, m * sizeof(double));
    CACHEb_INIT(std, double, std, 0, m * sizeof(double));
    CACHEs_INIT(data, double, data, 0, 15);
    double tmp_mean, tmp_std, tmp_data;
    for (int idx = start_idx; idx < end_idx; idx++) {
        // 将一维索引转换为二维索引
        int i = idx / m;
        int j = idx % m;
        CACHEb_RD(mean, &mean[j], tmp_mean);
        CACHEb_RD(std, &std[j], tmp_std);
        CACHEs_RD(data, &data[i * n + j], tmp_data);
        tmp_data -= tmp_mean;
        tmp_data /= (sqrt(FLOAT_N) * tmp_std);
        CACHEs_WT(data, &data[i * n + j], tmp_data);
    }
    CACHEs_FLUSH(data);
    CACHEb_INVALID(std);
    CACHEb_INVALID(mean);
}

__global__ void corr_kernel4_cache(int m, int n, double *symmat, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    // 注意: 这个核函数处理上三角矩阵，需要特殊处理任务分配
    // 需要计算的元素数量为 (m-1)*m/2
    int j1;

    // 每个线程处理一部分行
    int items_per_thread = ((m - 1) + total_threads - 1) / total_threads;
    int start_j1 = thread_id * items_per_thread;
    int end_j1 = min(start_j1 + items_per_thread, m - 1);
    CACHEs_INIT(symmat, double, symmat, 0, 15);
    double tmp_symmat1;
    for (j1 = start_j1; j1 < end_j1; j1++) {
        CACHEs_WT(symmat, &symmat[j1 * m + j1], 1.0);

        for (int j2 = (j1 + 1); j2 < m; j2++) {
            tmp_symmat1 = 0.0;
            for (int i = 0; i < n; i++) {
                tmp_symmat1 += data[i * m + j1] * data[i * m + j2];
            }
            CACHEs_WT(symmat, &symmat[j1 * m + j2], tmp_symmat1);
            symmat[j2 * m + j1] = tmp_symmat1;
        }
    }
    CACHEs_FLUSH(symmat);
}


#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double temp_std[24][SIMD_LEN];

__global__ void corr_kernel1_vec(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);

    for (int j = start_j; j < end_j; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, end_j);
        if (j + SIMD_LEN <= end_j) {
            lvector double zero_vec = (lvector double)vec_svbcast(0.0);
            vector_store(&zero_vec, mean + j, VEC_BYTES);
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                mean[jj] = 0.0;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, end_j);
            if (j + SIMD_LEN <= end_j) {
                lvector double data_vec, mean_vec;
                vector_load(&data[i * m + j], &data_vec, VEC_BYTES);
                vector_load(&mean[j], &mean_vec, VEC_BYTES);
                lvector double new_mean_vec = vec_mula(data_vec, (lvector double)vec_svbcast(1.0), mean_vec);
                vector_store(&new_mean_vec, &mean[j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    mean[jj] += data[i * m + jj];
                }
            }
        }
    }

    lvector double float_n_vec = (lvector double)vec_svbcast((double)FLOAT_N);
    for (int j = start_j; j < end_j; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, end_j);
        if (j + SIMD_LEN <= end_j) {
            lvector double mean_vec;
            vector_load(&mean[j], &mean_vec, VEC_BYTES);
            lvector double result_vec = vm_fdivd16(mean_vec, float_n_vec);
            vector_store(&result_vec, &mean[j], VEC_BYTES);
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                 mean[jj] /= FLOAT_N;
            }
        }
    }
}

__global__ void corr_kernel2_vec(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int tid_mod = thread_id % 24;

    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);

    lvector double float_n_vec = (lvector double)vec_svbcast((double)FLOAT_N);

    for (int j = start_j; j < end_j; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, end_j);

        if (j + SIMD_LEN <= end_j) {
            lvector double zero_vec = (lvector double)vec_svbcast(0.0);
            vector_store(&zero_vec, std + j, VEC_BYTES);

            lvector double mean_vec;
            vector_load(&mean[j], &mean_vec, VEC_BYTES);
            lvector double sum_vec = zero_vec;

            for (int i = 0; i < n; i++) {
                lvector double data_vec, diff_vec, sq_diff_vec;
                vector_load(&data[i * m + j], &data_vec, VEC_BYTES);
                diff_vec = vec_mulb(data_vec, (lvector double)vec_svbcast(1.0), mean_vec);
                sq_diff_vec = vec_muli(diff_vec, diff_vec);
                sum_vec = vec_mula(sq_diff_vec, (lvector double)vec_svbcast(1.0), sum_vec);
            }
            vector_store(&sum_vec, temp_std[tid_mod], VEC_BYTES);

            lvector double var_vec = vm_fdivd16(sum_vec, float_n_vec);
            vector_store(&var_vec, temp_std[tid_mod], VEC_BYTES);

            lvector double std_vec = vm_sqrtd16(var_vec);
            vector_store(&std_vec, temp_std[tid_mod], VEC_BYTES);

            vector_store(&std_vec, &std[j], VEC_BYTES);

        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                std[jj] = 0.0;
                double mean_val = mean[jj];
                for (int i = 0; i < n; i++) {
                    double diff = data[i * m + jj] - mean_val;
                    std[jj] += diff * diff;
                }
                std[jj] /= FLOAT_N;
                std[jj] = sqrt(std[jj]);
                if (std[jj] <= EPS) {
                    std[jj] = 1.0;
                }
            }
        }
    }

     for (int j = start_j; j < end_j && j + SIMD_LEN <= end_j; j += SIMD_LEN) {
         lvector double std_vec;
         vector_load(&std[j], &std_vec, VEC_BYTES);

         vector_store(&std_vec, temp_std[tid_mod], VEC_BYTES);
         int need_rewrite = 0;
         for(int k=0; k < SIMD_LEN; k++){
             if(temp_std[tid_mod][k] <= EPS){
                 temp_std[tid_mod][k] = 1.0;
                 need_rewrite = 1;
             }
         }
         if(need_rewrite){
             lvector double new_std_vec;
             vector_load(temp_std[tid_mod], &new_std_vec, VEC_BYTES);
             vector_store(&new_std_vec, &std[j], VEC_BYTES);
         }
    }
}

__global__ void corr_kernel3_vec(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int total_elements = n * m;
    int items_per_thread = (total_elements + total_threads - 1) / total_threads;
    int start_idx = thread_id * items_per_thread;
    int end_idx = min(start_idx + items_per_thread, total_elements);

    lvector double sqrt_float_n_vec = (lvector double)vec_svbcast(sqrt(FLOAT_N));

    for (int idx = start_idx; idx < end_idx; idx += SIMD_LEN) {
        int vec_end_idx = min(idx + SIMD_LEN, end_idx);
        int j_start = idx % m;
        if (idx + SIMD_LEN <= end_idx && j_start + SIMD_LEN <= m) {
            lvector double data_vec, mean_vec, std_vec;
            vector_load(&data[idx], &data_vec, VEC_BYTES);
            vector_load(&mean[j_start], &mean_vec, VEC_BYTES);
            vector_load(&std[j_start], &std_vec, VEC_BYTES);

            lvector double diff_vec = vec_mulb(data_vec, (lvector double)vec_svbcast(1.0), mean_vec);
            lvector double denom_vec = vec_muli(std_vec, sqrt_float_n_vec);
            lvector double result_vec = vm_fdivd16(diff_vec, denom_vec);

            vector_store(&result_vec, &data[idx], VEC_BYTES);

        } else {
            for (int t = idx; t < vec_end_idx; ++t) {
                int i = t / m;
                int j = t % m;
                if ((i < n) && (j < m)) {
                    data[t] -= mean[j];
                    data[t] /= (sqrt(FLOAT_N) * std[j]);
                }
            }
        }
    }
}

__global__ void corr_kernel4_vec(int m, int n, double *symmat, double *data)
{
    /* 线程与任务划分 ******************************************************/
    int thread_id      = get_thread_id();
    int total_threads  = get_group_size();

    /* j1: 行号, 只到 m-1 (最后一行不用再算) */
    int items_per_thread = ((m - 1) + total_threads - 1) / total_threads;
    int start_j1         = thread_id * items_per_thread;
    int end_j1           = min(start_j1 + items_per_thread, m - 1);

    /* ================================================================== */
    for (int j1 = start_j1; j1 < end_j1; ++j1)
    {
        /* 对角线恒为 1 */
        symmat[j1 * m + j1] = 1.0;

        /******************************************************************
         * 1. 对 j2（列号, 只算上三角, 即 j2>j1）做块状初始化  ---> 置 0
         ******************************************************************/
        for (int j2 = j1 + 1; j2 < m; j2 += SIMD_LEN)
        {
            int vec_end_j2 = min(j2 + SIMD_LEN, m);

            if (j2 + SIMD_LEN <= m)
            {
                /* 整块，直接向量清零 */
                lvector double zero_vec = (lvector double)vec_svbcast(0.0);
                vector_store(&zero_vec, &symmat[j1 * m + j2], VEC_BYTES);
            }
            else                /* 尾块 用标量 */
            {
                for (int jj = j2; jj < vec_end_j2; ++jj)
                    symmat[j1 * m + jj] = 0.0;
            }
        }

        /******************************************************************
         * 2. 遍历 n 行数据，做 Σ data[i][j1] * data[i][j2] 累加
         ******************************************************************/
        for (int j2 = j1 + 1; j2 < m; j2 += SIMD_LEN)
        {
            int vec_end_j2 = min(j2 + SIMD_LEN, m);

            if (j2 + SIMD_LEN <= m)
            {
                /* ========= 16 维整块 ========= */
                lvector double sum_vec = (lvector double)vec_svbcast(0.0);

                for (int i = 0; i < n; ++i)
                {
                    /* 把第 j1 列该行的一个标量广播成向量 */
                    double a_scalar = data[i * m + j1];
                    lvector double a_vec = (lvector double)vec_svbcast(a_scalar);

                    /* 取出当前行 (j2 … j2+15) 这一段 */
                    lvector double b_vec;
                    vector_load(&data[i * m + j2], &b_vec, VEC_BYTES);

                    /* 累加 */
                    lvector double prod_vec = vec_muli(b_vec, a_vec);
                    sum_vec = vec_mula(prod_vec,
                                       (lvector double)vec_svbcast(1.0),
                                       sum_vec);
                }

                /* 把结果写到上三角矩阵 */
                vector_store(&sum_vec, &symmat[j1 * m + j2], VEC_BYTES);
            }
            else
            {
                /* ========= 尾块 (不足 16) ========= */
                for (int jj = j2; jj < vec_end_j2; ++jj)
                {
                    double acc = 0.0;
                    for (int i = 0; i < n; ++i)
                        acc += data[i * m + j1] * data[i * m + jj];
                    symmat[j1 * m + jj] = acc;
                }
            }
        }

        /******************************************************************
         * 3. 把上三角结果复制到下三角 (保持对称)
         *    为简单起见用标量循环，benign race(写相同值) 可接受
         ******************************************************************/
        for (int j2 = j1 + 1; j2 < m; ++j2)
            symmat[j2 * m + j1] = symmat[j1 * m + j2];
    }
}
#include "../CORR/kernel_vec.h"//大模型生成的存储文件
#include "../CORR/kernel_cache_llm.h"//SM缓存优化文件