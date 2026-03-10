#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/compute_tool.h"
#include "../common/prof_event.h"


__global__ void mm2_kernel1(int ni, int nj, int nk, double alpha, double *tmp,
                            double *A, double *B)
{
    // 获取当前线程的线程 ID和总数
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 计算每个线程负责处理的数据范围
    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算起始和结束索引
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

    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / nj;
        int j = idx % nj;
        tmp[i * nj + j] = 0;
    }

    for (int k = 0; k < nk; k++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / nj;
            int j = idx % nj;

            tmp[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
        }
    }
}

__global__ void mm2_kernel2(int ni, int nj, int nl, double beta, double *tmp,
                            double *C, double *D)
{

    int thread_id = get_thread_id();

    int group_size = get_group_size();

    // 计算每个线程负责的任务范围
    int total_elements = ni * nl;                          // 任务总数（D矩阵的元素个数）
    int elements_per_thread = total_elements / group_size; // 每个线程负责的元素个数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 分配余下的元素给前面的一些线程
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / nl; // 计算 i（行索引）
        int j = idx % nl; // 计算 j（列索引）
        D[i * nl + j] *= beta;
    }

    for (int k = 0; k < nj; k++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / nl; // 计算 i（行索引）
            int j = idx % nl; // 计算 j（列索引）
            D[i * nl + j] += tmp[i * nj + k] * C[k * nl + j];
        }
    }
}

__global__ void mm2_kernel1_cache(int ni, int nj, int nk, int nl, double alpha, double beta, double *tmp,
                            double *A, double *B)
{
    // 获取当前线程的线程 ID和总数
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 计算每个线程负责处理的数据范围
    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    // 计算起始和结束索引
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
    CACHEs_INIT(tmp, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 9);
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_tmp;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / nj;
        int j = idx % nj;
        tmp_tmp = 0;
        CACHEs_WT(tmp, &tmp[i * nj + j], tmp_tmp);
    }
    double tmp_A, tmp_B;
    for (int k = 0; k < nk; k++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / nj;
            int j = idx % nj;
            CACHEs_RD(A, &A[i * nk + k], tmp_A);
            CACHEs_RD(B, &B[k * nj + j], tmp_B);    
            CACHEs_RD(tmp, &tmp[i * nj + j], tmp_tmp);
            tmp_tmp += alpha * tmp_A * tmp_B;
            CACHEs_WT(tmp, &tmp[i * nj + j], tmp_tmp);
        }
    }
    CACHEs_FLUSH(tmp);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void mm2_kernel2_cache(int ni, int nj, int nk, int nl, double alpha, double beta, double *tmp,
                            double *C, double *D)
{
    // 获取当前线程的线程 ID
    int thread_id = get_thread_id();
    // 获取线程的总数
    int group_size = get_group_size();

    // 计算每个线程负责的任务范围
    int total_elements = ni * nl;                          // 任务总数（D矩阵的元素个数）
    int elements_per_thread = total_elements / group_size; // 每个线程负责的元素个数
    int remainder = total_elements % group_size;           // 余下的元素

    // 每个线程的任务范围
    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    // 分配余下的元素给前面的一些线程
    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    CACHEs_INIT(C, double, C, 0, 9);
    CACHEs_INIT(D, double, D, 0, 14);
    CACHEs_INIT(tmp, double, tmp, 0, 15);
    double tmp_D, tmp_C, tmp_tmp;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / nl; // 计算 i（行索引）
        int j = idx % nl; // 计算 j（列索引）
        CACHEs_RD(D, &D[i * nl + j], tmp_D);
        tmp_D *= beta;
        CACHEs_WT(D, &D[i * nl + j], tmp_D);
    }

    for (int k = 0; k < nj; k++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / nl; // 计算 i（行索引）
            int j = idx % nl; // 计算 j（列索引）
            CACHEs_RD(C, &C[k * nl + j], tmp_C);
            CACHEs_RD(tmp, &tmp[i * nj + k], tmp_tmp);
            CACHEs_RD(D, &D[i * nl + j], tmp_D);
            tmp_D += tmp_C * tmp_tmp;
            CACHEs_WT(D, &D[i * nl + j], tmp_D);
        }
    }
    CACHEs_FLUSH(D);
    CACHEs_INVALID(tmp);
    CACHEs_INVALID(C);
}


#define ELEMS_PER_PART 1024  // 分批的最大元素数，每次缓存处理的最大任务数

__global__ void mm2_kernel1_cache_fast(
    int ni, int nj, int nk, double alpha, double *tmp,
    double *A, double *B)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nj;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_A_row = (double*)scalar_malloc(nk * sizeof(double));
    double* cache_B_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_tmp   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nj;
        int j_start = idx % nj;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        // 如果当前行剩余列不足batch_tasks，则取最小
        batch_tasks = min(batch_tasks, nj - j_start);

        // 初始化 cache_tmp
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_tmp[bj] = 0.0;
        }
        // 缓存当前行A
        scalar_load(&A[i * nk], cache_A_row, nk * sizeof(double));

        // 进行矩阵乘法累加
        for (int k = 0; k < nk; ++k) {
            // 缓存B[k行]
            scalar_load(&B[k * nj + j_start], cache_B_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_tmp[bj] += alpha * cache_A_row[k] * cache_B_row[bj];
            }
        }
        // 写回
        scalar_store(cache_tmp, &tmp[i * nj + j_start], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_A_row);
    scalar_free(cache_B_row);
    scalar_free(cache_tmp);
}

__global__ void mm2_kernel2_cache_fast(
    int ni, int nj, int nl, double beta, double *tmp,
    double *C, double *D)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nl;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_tmp_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_C_row   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_D       = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nl;
        int j_start = idx % nl;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nl - j_start);

        // 缓存 D 当前行块
        scalar_load(&D[i * nl + j_start], cache_D, batch_tasks * sizeof(double));

        // 先乘 beta
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_D[bj] *= beta;
        }

        // 缓存 tmp 当前行
        scalar_load(&tmp[i * nj], cache_tmp_row, nj * sizeof(double));

        for (int k = 0; k < nj; ++k) {
            // 缓存 C[k行]
            scalar_load(&C[k * nl + j_start], cache_C_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_D[bj] += cache_tmp_row[k] * cache_C_row[bj];
            }
        }

        // 写回结果
        scalar_store(cache_D, &D[i * nl + j_start], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_tmp_row);
    scalar_free(cache_C_row);
    scalar_free(cache_D);
}


#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void mm2_kernel1_vec(int ni, int nj, int nk, double alpha, double *tmp,
                                double *A, double *B)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
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

    // 确保不越界
    end_idx = min(end_idx, total_elements);

    // 初始化临时数组为零（向量化和标量结合）
    for (int idx = start_idx; idx < end_idx; idx += SIMD_LEN) {
        int i = idx / nj;
        int j = idx % nj;
        // 计算可向量化的终止位置
        int vec_end = min(idx + SIMD_LEN, end_idx);

        if (j + SIMD_LEN <= nj && idx + SIMD_LEN <= end_idx) {
            lvector double zero = (lvector double)vec_svbcast(0.0);
            vector_store(&zero, tmp + i * nj + j, VEC_BYTES);
        } else {
            // 标量处理尾部
            for (int t = idx; t < vec_end; ++t) {
                int ti = t / nj;
                int tj = t % nj;
                tmp[ti * nj + tj] = 0.0;
            }
        }
    }

    // 矩阵乘法计算
    for (int k = 0; k < nk; k++) {
        for (int idx = start_idx; idx < end_idx; idx += SIMD_LEN) {
            int i = idx / nj;
            int j = idx % nj;
            int vec_end = min(idx + SIMD_LEN, end_idx);

            if (j + SIMD_LEN <= nj && idx + SIMD_LEN <= end_idx) {
                lvector double b;
                lvector double c;

                vector_load(B + k * nj + j, &b, VEC_BYTES);
                vector_load(tmp + i * nj + j, &c, VEC_BYTES);

                lvector double res = vec_muli((lvector double)vec_svbcast(A[i * nk + k]), (lvector double)vec_svbcast(alpha));
                res = vec_mula(b, res, c);
                vector_store(&res, tmp + i * nj + j, VEC_BYTES);
            } else {
                // 标量处理尾部
                for (int t = idx; t < vec_end; ++t) {
                    int ti = t / nj;
                    int tj = t % nj;
                    tmp[ti * nj + tj] += alpha * A[ti * nk + k] * B[k * nj + tj];
                }
            }
        }
    }
}

__global__ void mm2_kernel2_vec(int ni, int nj, int nl, double beta, double *tmp,
                                double *C, double *D)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
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

    // 确保不越界
    end_idx = min(end_idx, total_elements);

    // 将 D 矩阵乘以 beta（向量化和标量结合）
    for (int idx = start_idx; idx < end_idx; idx += SIMD_LEN) {
        int i = idx / nl;
        int j = idx % nl;
        int vec_end = min(idx + SIMD_LEN, end_idx);

        if (j + SIMD_LEN <= nl && idx + SIMD_LEN <= end_idx) {
            lvector double d;
            vector_load(D + i * nl + j, &d, VEC_BYTES);
            lvector double res = vec_muli(d, (lvector double)vec_svbcast(beta));
            vector_store(&res, D + i * nl + j, VEC_BYTES);
        } else {
            // 标量处理尾部
            for (int t = idx; t < vec_end; ++t) {
                int ti = t / nl;
                int tj = t % nl;
                D[ti * nl + tj] *= beta;
            }
        }
    }

    // 更新 D 矩阵
    for (int k = 0; k < nj; k++) {
        for (int idx = start_idx; idx < end_idx; idx += SIMD_LEN) {
            int i = idx / nl;
            int j = idx % nl;
            int vec_end = min(idx + SIMD_LEN, end_idx);

            if (j + SIMD_LEN <= nl && idx + SIMD_LEN <= end_idx) {
                lvector double c;
                lvector double d;

                vector_load(C + k * nl + j, &c, VEC_BYTES);
                vector_load(D + i * nl + j, &d, VEC_BYTES);

                lvector double res = vec_mula((lvector double)vec_svbcast(tmp[i * nj + k]), c, d);
                vector_store(&res, D + i * nl + j, VEC_BYTES);
            } else {
                // 标量处理尾部
                for (int t = idx; t < vec_end; ++t) {
                    int ti = t / nl;
                    int tj = t % nl;
                    D[ti * nl + tj] += tmp[ti * nj + k] * C[k * nl + tj];
                }
            }
        }
    }
}
#include "../2MM/kernel_vec.h"//大模型生成的存储文件
#include "../2MM/kernel_cache_llm.h"//SM缓存优化文件