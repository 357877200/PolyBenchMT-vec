#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
__global__ void bicg_kernel1(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        s[j] = 0.0f;
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            s[j] += r[i] * A[i * ny + j];
        }
    }
}

__global__ void bicg_kernel2(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    for (int i = start_i; i < end_i; i++) {
        q[i] = 0.0f;

        for (int j = 0; j < ny; j++) {
            q[i] += A[i * ny + j] * p[j];
        }
    }
}

#ifdef MINI_DATASET
__global__ void bicg_kernel1_cache(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);
    CACHEb_INIT(s, double, &s[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEb_INIT(r, double, r, 0, nx * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_s, tmp_r, tmp_A;
    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        CACHEb_WT(s, &s[j - start_j], 0);
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            CACHEb_RD(s, &s[j - start_j], tmp_s);
            CACHEb_RD(r, &r[i], tmp_r);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_s += tmp_r * tmp_A;
            CACHEb_WT(s, &s[j - start_j], tmp_s);
        }
    }
    CACHEb_FLUSH(s);
    CACHEb_INVALID(r);
    CACHEs_INVALID(A);
}

__global__ void bicg_kernel2_cache(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    CACHEb_INIT(q, double, &q[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(p, double, p, 0, ny * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_q, tmp_p, tmp_A;

    for (int i = start_i; i < end_i; i++) {
        tmp_q = 0.0f;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(p, &p[j], tmp_p);
            tmp_q += tmp_A * tmp_p;
        }
        CACHEb_WT(q, &q[i - start_i], tmp_q);
    }
    CACHEb_FLUSH(q);
    CACHEb_INVALID(p);
    CACHEs_INVALID(A);
}
#endif

#ifdef SMALL_DATASET
__global__ void bicg_kernel1_cache(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);
    CACHEb_INIT(s, double, &s[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEb_INIT(r, double, r, 0, nx * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_s, tmp_r, tmp_A;
    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        CACHEb_WT(s, &s[j - start_j], 0);
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            CACHEb_RD(s, &s[j - start_j], tmp_s);
            CACHEb_RD(r, &r[i], tmp_r);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_s += tmp_r * tmp_A;
            CACHEb_WT(s, &s[j - start_j], tmp_s);
        }
    }
    CACHEb_FLUSH(s);
    CACHEb_INVALID(r);
    CACHEs_INVALID(A);
}

__global__ void bicg_kernel2_cache(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    CACHEb_INIT(q, double, &q[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(p, double, p, 0, ny * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 15);
    double tmp_q, tmp_p, tmp_A;

    for (int i = start_i; i < end_i; i++) {
        tmp_q = 0.0f;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(p, &p[j], tmp_p);
            tmp_q += tmp_A * tmp_p;
        }
        CACHEb_WT(q, &q[i - start_i], tmp_q);
    }
    CACHEb_FLUSH(q);
    CACHEb_INVALID(p);
    CACHEs_INVALID(A);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void bicg_kernel1_cache(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);
    CACHEb_INIT(s, double, &s[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEb_INIT(r, double, r, 0, nx * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_s, tmp_r, tmp_A;
    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        CACHEb_WT(s, &s[j - start_j], 0);
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            CACHEb_RD(s, &s[j - start_j], tmp_s);
            CACHEb_RD(r, &r[i], tmp_r);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_s += tmp_r * tmp_A;
            CACHEb_WT(s, &s[j - start_j], tmp_s);
        }
    }
    CACHEb_FLUSH(s);
    CACHEb_INVALID(r);
    CACHEs_INVALID(A);
}

__global__ void bicg_kernel2_cache(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    CACHEb_INIT(q, double, &q[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEb_INIT(p, double, p, 0, ny * sizeof(double));
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_q, tmp_p, tmp_A;

    for (int i = start_i; i < end_i; i++) {
        tmp_q = 0.0f;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEb_RD(p, &p[j], tmp_p);
            tmp_q += tmp_A * tmp_p;
        }
        CACHEb_WT(q, &q[i - start_i], tmp_q);
    }
    CACHEb_FLUSH(q);
    CACHEb_INVALID(p);
    CACHEs_INVALID(A);
}
#endif

#ifdef LARGE_DATASET
__global__ void bicg_kernel1_cache(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);
    CACHEb_INIT(s, double, &s[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEs_INIT(r, double, r, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_s, tmp_r, tmp_A;
    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        CACHEb_WT(s, &s[j - start_j], 0);
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            CACHEb_RD(s, &s[j - start_j], tmp_s);
            CACHEs_RD(r, &r[i], tmp_r);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_s += tmp_r * tmp_A;
            CACHEb_WT(s, &s[j - start_j], tmp_s);
        }
    }
    CACHEb_FLUSH(s);
    CACHEs_INVALID(r);
    CACHEs_INVALID(A);
}

__global__ void bicg_kernel2_cache(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    CACHEb_INIT(q, double, &q[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEs_INIT(p, double, p, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_q, tmp_p, tmp_A;

    for (int i = start_i; i < end_i; i++) {
        tmp_q = 0.0f;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEs_RD(p, &p[j], tmp_p);
            tmp_q += tmp_A * tmp_p;
        }
        CACHEb_WT(q, &q[i - start_i], tmp_q);
    }
    CACHEb_FLUSH(q);
    CACHEs_INVALID(p);
    CACHEs_INVALID(A);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void bicg_kernel1_cache(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    // 计算当前线程的起始和结束位置
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);
    CACHEb_INIT(s, double, &s[start_j], 0, (end_j - start_j) * sizeof(double));
    CACHEs_INIT(r, double, r, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_s, tmp_r, tmp_A;
    // 先初始化输出数组
    for (int j = start_j; j < end_j; j++) {
        CACHEb_WT(s, &s[j - start_j], 0);
    }

    // 交换循环顺序，提高内存访问效率
    for (int i = 0; i < nx; i++) {
        for (int j = start_j; j < end_j; j++) {
            CACHEb_RD(s, &s[j - start_j], tmp_s);
            CACHEs_RD(r, &r[i], tmp_r);
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            tmp_s += tmp_r * tmp_A;
            CACHEb_WT(s, &s[j - start_j], tmp_s);
        }
    }
    CACHEb_FLUSH(s);
    CACHEs_INVALID(r);
    CACHEs_INVALID(A);
}

__global__ void bicg_kernel2_cache(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    // 计算每个线程需要处理的元素数量
    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    // 计算当前线程的起始和结束位置
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    // 处理分配给当前线程的所有元素
    CACHEb_INIT(q, double, &q[start_i], 0, (end_i - start_i) * sizeof(double));
    CACHEs_INIT(p, double, p, 0, 15);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_q, tmp_p, tmp_A;

    for (int i = start_i; i < end_i; i++) {
        tmp_q = 0.0f;
        for (int j = 0; j < ny; j++) {
            CACHEs_RD(A, &A[i * ny + j], tmp_A);
            CACHEs_RD(p, &p[j], tmp_p);
            tmp_q += tmp_A * tmp_p;
        }
        CACHEb_WT(q, &q[i - start_i], tmp_q);
    }
    CACHEb_FLUSH(q);
    CACHEs_INVALID(p);
    CACHEs_INVALID(A);
}
#endif

#define SIMD_LEN 16
#define VEC_BYTES 128

__gsm__ static double temp_q[24][SIMD_LEN];

__global__ void bicg_kernel1_vec(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;
    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);

    for (int j = start_j; j < end_j; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, end_j);

        if (j + SIMD_LEN <= end_j) {
            lvector double zero_vec = (lvector double)vec_svbcast(0.0);
            vector_store(&zero_vec, s + j, VEC_BYTES);
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                s[jj] = 0.0;
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        double r_val = r[i];
        lvector double r_vec = (lvector double)vec_svbcast(r_val);

        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, end_j);

            if (j + SIMD_LEN <= end_j) {
                lvector double a_vec, s_vec;
                vector_load(&A[i * ny + j], &a_vec, VEC_BYTES);
                vector_load(&s[j], &s_vec, VEC_BYTES);

                lvector double mul_res = vec_muli(r_vec, a_vec);
                lvector double new_s_vec = vec_mula(mul_res, (lvector double)vec_svbcast(1.0), s_vec);

                vector_store(&new_s_vec, &s[j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    s[jj] += r_val * A[i * ny + jj];
                }
            }
        }
    }
}


__global__ void bicg_kernel2_vec(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;
    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    for (int i = start_i; i < end_i; i++) {
        double sum_scalar = 0.0;
        lvector double sum_vec = (lvector double)vec_svbcast(0.0);
        int tid_mod = thread_id % 24;

        for (int j = 0; j < ny; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, ny);

            if (j + SIMD_LEN <= ny) {
                lvector double a_vec, p_vec;
                vector_load(&A[i * ny + j], &a_vec, VEC_BYTES);
                vector_load(&p[j], &p_vec, VEC_BYTES);

                lvector double mul_res = vec_muli(a_vec, p_vec);
                sum_vec = vec_mula(mul_res, (lvector double)vec_svbcast(1.0), sum_vec);

            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                     sum_scalar += A[i * ny + jj] * p[jj];
                }
            }
        }

        vector_store(&sum_vec, temp_q[tid_mod], VEC_BYTES);
        for(int k = 0; k < SIMD_LEN; k++) {
            sum_scalar += temp_q[tid_mod][k];
        }

        q[i] = sum_scalar;
    }
}
#include "../BICG/kernel_vec.h"//大模型生成的存储文件
#include "../BICG/kernel_cache_llm.h"//SM缓存优化文件
