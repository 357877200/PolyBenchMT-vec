#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void convolution2D_kernel(int ni, int nj, double *A, double *B, uint64_t *before_hot_data, uint64_t *after_hot_data)
{
    int group_size = get_group_size();
    int thread_id  = get_thread_id();

    // 卷积核系数
    double c11 = +0.2, c21 = +0.5, c31 = -0.8;
    double c12 = -0.3, c22 = +0.6, c32 = -0.9;
    double c13 = +0.4, c23 = +0.7, c33 = +0.10;

    const int total_tasks = (ni - 2) * (nj - 2);
    if (total_tasks <= 0) return;

    const int base_tasks = total_tasks / group_size;
    const int remainder  = total_tasks % group_size;

    int start = (thread_id < remainder)
        ? thread_id * (base_tasks + 1)
        : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);

    // 单循环完成全部卷积计算
    for (int t = start; t < end; ++t)
    {
        const int i = 1 + t / (nj - 2); // i ∈ [1, ni-2]
        const int j = 1 + t % (nj - 2); // j ∈ [1, nj-2]

        double val =
            c11 * A[(i - 1) * nj + (j - 1)] +
            c21 * A[(i - 1) * nj + j]       +
            c31 * A[(i - 1) * nj + (j + 1)] +
            c12 * A[i * nj + (j - 1)]      +
            c22 * A[i * nj + j]            +
            c32 * A[i * nj + (j + 1)]      +
            c13 * A[(i + 1) * nj + (j - 1)] +
            c23 * A[(i + 1) * nj + j]       +
            c33 * A[(i + 1) * nj + (j + 1)];

        B[i * nj + j] = val;
    }
}


__global__ void convolution2D_kernel_cache(int ni, int nj, double *A, double *B, uint64_t *before_hot_data, uint64_t *after_hot_data)
{
    int group_size = get_group_size();
    int thread_id = get_thread_id();

    double c11 = +0.2, c21 = +0.5, c31 = -0.8;
    double c12 = -0.3, c22 = +0.6, c32 = -0.9;
    double c13 = +0.4, c23 = +0.7, c33 = +0.10;

    const int total_tasks = (ni - 2) * (nj - 2);
    if (total_tasks <= 0)
        return;

    // 任务分配策略：前remainder个线程多处理1个任务
    const int base_tasks = total_tasks / group_size;
    const int remainder = total_tasks % group_size;

    int start = (thread_id < remainder) ? thread_id * (base_tasks + 1)
                                        : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);
    // ！！！CACHEs_INIT第三个参数要随着输入规模改变
    CACHEs_INIT(A, double, A, 0, 7);
    CACHEs_INIT(B, double, B, 0, 6);
    double tmp_A1, tmp_A2, tmp_A3, tmp_A4, tmp_A5, tmp_A6, tmp_A7, tmp_A8, tmp_A9, tmp_B;
    for (int t = start; t < end; ++t) {
        const int i = 1 + t / (nj - 2); // i ∈ [1, ni-2]
        const int j = 1 + t % (nj - 2); // j ∈ [1, nj-2]
        CACHEs_RD(A, &A[(i - 1) * nj + (j - 1)], tmp_A1);
        CACHEs_RD(A, &A[(i - 1) * nj + j], tmp_A2);
        CACHEs_RD(A, &A[(i - 1) * nj + (j + 1)], tmp_A3);
        tmp_B = c11 * tmp_A1 + c21 * tmp_A2 + c31 * tmp_A3;
        CACHEs_WT(B, &B[i * nj + j], tmp_B);
    }
    for (int t = start; t < end; ++t) {
        const int i = 1 + t / (nj - 2); // i ∈ [1, ni-2]
        const int j = 1 + t % (nj - 2); // j ∈ [1, nj-2]
        CACHEs_RD(A, &A[i * nj + (j - 1)], tmp_A4);
        CACHEs_RD(A, &A[i * nj + j], tmp_A5);
        CACHEs_RD(A, &A[i * nj + (j + 1)], tmp_A6);
        CACHEs_RD(B, &B[i * nj + j], tmp_B);
        tmp_B += c12 * tmp_A4 + c22 * tmp_A5 + c32 * tmp_A6;
        CACHEs_WT(B, &B[i * nj + j], tmp_B);
    }
    for (int t = start; t < end; ++t) {
        const int i = 1 + t / (nj - 2); // i ∈ [1, ni-2]
        const int j = 1 + t % (nj - 2); // j ∈ [1, nj-2]
        CACHEs_RD(A, &A[(i + 1) * nj + (j - 1)], tmp_A7);
        CACHEs_RD(A, &A[(i + 1) * nj + j], tmp_A8);
        CACHEs_RD(A, &A[(i + 1) * nj + (j + 1)], tmp_A9);
        CACHEs_RD(B, &B[i * nj + j], tmp_B);
        tmp_B += c13 * tmp_A7 + c23 * tmp_A8 + c33 * tmp_A9;
        CACHEs_WT(B, &B[i * nj + j], tmp_B);
    }
    CACHEs_INVALID(A);
    CACHEs_FLUSH(B);
}


#define ELEMS_PER_PART 1024

__global__ void convolution2D_kernel_cache_fast(
    int ni, int nj, double *A, double *B)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    // 卷积系数
    double c11=+0.2, c12=-0.3, c13=+0.4;
    double c21=+0.5, c22=+0.6, c23=+0.7;
    double c31=-0.8, c32=-0.9, c33=+0.10;

    if (ni < 3 || nj < 3) return;
    const int total_tasks = (ni - 2) * (nj - 2);
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_above = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_curr  = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_below = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_out   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int t = start; t < end; )
    {
        int first_i = 1 + t / (nj - 2);
        int first_j = 1 + t % (nj - 2);
        int i = first_i;

        // 当前行剩余可算列数、批量大小
        int remain_in_row = (nj - 2) - first_j + 1;
        int batch_tasks = min(ELEMS_PER_PART, min(end - t, remain_in_row));

        // halo
        int load_start_j = (first_j > 1) ? (first_j - 1) : 0;
        int load_end_j   = min(first_j + batch_tasks, nj - 1);
        int load_len = load_end_j - load_start_j + 1;

        // 批量加载三行数据
        scalar_load(&A[(i-1)*nj + load_start_j], cache_above, load_len * sizeof(double));
        scalar_load(&A[i*nj     + load_start_j], cache_curr,  load_len * sizeof(double));
        scalar_load(&A[(i+1)*nj + load_start_j], cache_below, load_len * sizeof(double));

        // 卷积计算
        for (int bi = 0; bi < batch_tasks; ++bi) {
            int j = first_j + bi;
            int off_m1 = (j - 1) - load_start_j;
            int off_0  = j - load_start_j;
            int off_p1 = (j + 1) - load_start_j;
            double res =
                c11*cache_above[off_m1] + c21*cache_above[off_0] + c31*cache_above[off_p1] +
                c12*cache_curr [off_m1] + c22*cache_curr [off_0] + c32*cache_curr [off_p1] +
                c13*cache_below[off_m1] + c23*cache_below[off_0] + c33*cache_below[off_p1];
            cache_out[bi] = res;
        }

        // 批量写回当前行的 batch_tasks 个结果
        scalar_store(cache_out, &B[i * nj + first_j], batch_tasks * sizeof(double));

        t += batch_tasks;
        // 若这一批正好扫完整行，则自动进入下一行
    }

    scalar_free(cache_above);
    scalar_free(cache_curr);
    scalar_free(cache_below);
    scalar_free(cache_out);
}
/* 一条向量 16 个 double */
#define SIMD_LEN 16
#define VEC_BYTES 128
/*--------------------------------------------------------------*/
__global__ void convolution2D_kernel_vec(int ni, int nj,
                                     double *A,
                                     double *B,
                                     uint64_t *before_hot_data,
                                     uint64_t *after_hot_data)
{
    int thread_id  = get_thread_id();
    int group_size = get_group_size();

    /* 把 3×3 卷积核常量广播成向量 ------------------------------ */
    lvector double vc11 = (lvector double)vec_svbcast(0.2); // 0.2f
    lvector double vc21 = (lvector double)vec_svbcast(0.5); // 0.5f
    lvector double vc31 = (lvector double)vec_svbcast(-0.8); // -0.8f
 
    lvector double vc12 = (lvector double)vec_svbcast(-0.3); // -0.3f
    lvector double vc22 = (lvector double)vec_svbcast(0.6); // 0.6f
    lvector double vc32 = (lvector double)vec_svbcast(-0.9); // -0.9f
 
    lvector double vc13 = (lvector double)vec_svbcast(0.4); // 0.4f
    lvector double vc23 = (lvector double)vec_svbcast(0.7); // 0.7f
    lvector double vc33 = (lvector double)vec_svbcast(0.1); // 0.1f

    /* 需要计算的 B 区域宽高 ------------------------------------ */
    const int nj_inner    = nj - 2;            /* 每行 task 数       */
    const int total_tasks = (ni - 2) * nj_inner;
    if (total_tasks <= 0) return;

    /* 与原版相同的任务分配 -------------------------------------- */
    const int base_tasks = total_tasks / group_size;
    const int remainder  = total_tasks % group_size;

    int start = (thread_id < remainder)
              ? thread_id * (base_tasks + 1)
              : remainder * (base_tasks + 1) +
                (thread_id - remainder) * base_tasks;
    int end   = start + ((thread_id < remainder) ? (base_tasks + 1)
                                                 :  base_tasks);

    /* 申请 9 个向量缓冲，最多一次要读 3×3 位置 ------------------- */
    lvector double *buf =
        (lvector double *)vector_malloc(sizeof(lvector double) * 9);

    lvector double *up_l  = buf + 0, *up_m  = buf + 1, *up_r  = buf + 2;
    lvector double *mid_l = buf + 3, *mid_m = buf + 4, *mid_r = buf + 5;
    lvector double *lo_l  = buf + 6, *lo_m  = buf + 7, *lo_r  = buf + 8;

/* ---------------- 主循环：t 线性遍历所有 task -------------- */
for (int t = start; t < end; )
{
    int i = 1 + t / nj_inner;   /* 行号 */
    int j = 1 + t % nj_inner;   /* 列号 */
 
    int remain_row = nj_inner - (j - 1);    /* 同一行剩余 task */


    if (remain_row >= SIMD_LEN && (end - t) >= SIMD_LEN )
    {
        /* ========  向量化处理 16 个像素  =================== */
        size_t off_up  = (size_t)(i - 1) * nj + (j - 1);
        size_t off_md  = (size_t) i       * nj + (j - 1);
        size_t off_dn  = (size_t)(i + 1) * nj + (j - 1);
 
        /* vector_load( src_global , dst_buffer , bytes ) */
        vector_load(A + off_up ,  up_l ,  VEC_BYTES);
        vector_load(A + off_up + 1, up_m , VEC_BYTES);
        vector_load(A + off_up + 2, up_r , VEC_BYTES);
 
        vector_load(A + off_md ,  mid_l, VEC_BYTES);
        vector_load(A + off_md + 1, mid_m, VEC_BYTES);
        vector_load(A + off_md + 2, mid_r, VEC_BYTES);
 
        vector_load(A + off_dn ,  lo_l , VEC_BYTES);
        vector_load(A + off_dn + 1, lo_m , VEC_BYTES);
        vector_load(A + off_dn + 2, lo_r , VEC_BYTES);
 
        /* 重置 res_vec 为 0.0，防止累加错误 */
        lvector double res_vec = (lvector double)vec_svbcast(0.0);
 
        /* res_vec = Σ cij * aij  */
        res_vec = vec_mula(*up_l ,  vc11, res_vec);
        res_vec = vec_mula(*up_m ,  vc21, res_vec);
        res_vec = vec_mula(*up_r ,  vc31, res_vec);
 
        res_vec = vec_mula(*mid_l, vc12, res_vec);
        res_vec = vec_mula(*mid_m, vc22, res_vec);
        res_vec = vec_mula(*mid_r, vc32, res_vec);
 
        res_vec = vec_mula(*lo_l , vc13, res_vec);
        res_vec = vec_mula(*lo_m , vc23, res_vec);
        res_vec = vec_mula(*lo_r , vc33, res_vec);
 
        /* 写回结果：vector_store( src_buf , dst_global , bytes ) */
        vector_store(&res_vec, B + i * nj + j, VEC_BYTES);
 
        t += SIMD_LEN;      /* 吞掉 16 个 task */
    }
    else
    {
        /* ========== 标量尾部 ================================ */
        double v11 = A[(i - 1) * nj + (j - 1)];
        double v12 = A[(i - 1) * nj +  j     ];
        double v13 = A[(i - 1) * nj + (j + 1)];
 
        double v21 = A[ i      * nj + (j - 1)];
        double v22 = A[ i      * nj +  j     ];
        double v23 = A[ i      * nj + (j + 1)];
 
        double v31 = A[(i + 1) * nj + (j - 1)];
        double v32 = A[(i + 1) * nj +  j     ];
        double v33 = A[(i + 1) * nj + (j + 1)];
 
        B[i * nj + j] =
            0.2 * v11 + 0.5 * v12 - 0.8 * v13
            - 0.3 * v21 + 0.6 * v22 - 0.9 * v23
            + 0.4 * v31 + 0.7 * v32 + 0.10 * v33;
 
        ++t;                 /* 只解决 1 个 task */
    }
}

    vector_free(buf);
}

/* 2025.7.29 float是32位，而vector_load和vector_store的主机端传入地址要64位对齐，这个核函数目前不能正常工作 */
/* 一条向量 32 个 float */
#define SIMD_32_LEN 32

/*--------------------------------------------------------------*/
__global__ void convolution2D_kernel_vec_f32(int ni, int nj,
                                     float *A,
                                     float *B,
                                     uint64_t *before_hot_data,
                                     uint64_t *after_hot_data)
{
    int thread_id  = get_thread_id();
    int group_size = get_group_size();

    /* 把 3×3 卷积核常量广播成向量 ------------------------------ */
    lvector float vc11 = (lvector float)vec_svbcast(0x3E4CCCCD3E4CCCCD); // 0.2f
    lvector float vc21 = (lvector float)vec_svbcast(0x3F0000003F000000); // 0.5f
    lvector float vc31 = (lvector float)vec_svbcast(0xBF4CCCCDBF4CCCCD); // -0.8f
 
    lvector float vc12 = (lvector float)vec_svbcast(0xBE99999ABE99999A); // -0.3f
    lvector float vc22 = (lvector float)vec_svbcast(0x3F19999A3F19999A); // 0.6f
    lvector float vc32 = (lvector float)vec_svbcast(0xBF666666BF666666); // -0.9f
 
    lvector float vc13 = (lvector float)vec_svbcast(0x3ECCCCCD3ECCCCCD); // 0.4f
    lvector float vc23 = (lvector float)vec_svbcast(0x3F3333333F333333); // 0.7f
    lvector float vc33 = (lvector float)vec_svbcast(0x3DCCCCCD3DCCCCCD); // 0.1f

    /* 需要计算的 B 区域宽高 ------------------------------------ */
    const int nj_inner    = nj - 2;            /* 每行 task 数       */
    const int total_tasks = (ni - 2) * nj_inner;
    if (total_tasks <= 0) return;

    /* 与原版相同的任务分配 -------------------------------------- */
    const int base_tasks = total_tasks / group_size;
    const int remainder  = total_tasks % group_size;

    int start = (thread_id < remainder)
              ? thread_id * (base_tasks + 1)
              : remainder * (base_tasks + 1) +
                (thread_id - remainder) * base_tasks;
    int end   = start + ((thread_id < remainder) ? (base_tasks + 1)
                                                 :  base_tasks);

 /* 申请缓冲区：3行，每行 SIMD_32_LEN + 2 个元素 */
    int row_size = SIMD_32_LEN + 2;  // 需要额外2个元素用于卷积窗口
    lvector float *buf = (lvector float *)vector_malloc(sizeof(lvector float) * 3 * row_size);

/* ---------------- 主循环：t 线性遍历所有 task -------------- */
for (int t = start; t < end; )
{
    int i = 1 + t / nj_inner;   /* 行号 */
    int j = 1 + t % nj_inner;   /* 列号 */
 
    int remain_row = nj_inner - (j - 1);    /* 同一行剩余 task */
    /* 检查所有相关地址是否64字节对齐 */

    size_t off_up  = (size_t)(i - 1) * nj + (j - 1);
    size_t off_md  = (size_t) i       * nj + (j - 1);
    size_t off_dn  = (size_t)(i + 1) * nj + (j - 1);
    
    uintptr_t addr_load_up = (uintptr_t)(A + off_up);
    uintptr_t addr_load_md = (uintptr_t)(A + off_md);
    uintptr_t addr_load_dn = (uintptr_t)(A + off_dn);
    uintptr_t addr_store   = (uintptr_t)(B + i * nj + j);
    
    int all_aligned = (addr_load_up % 64 == 0) && 
                      (addr_load_md % 64 == 0) && 
                      (addr_load_dn % 64 == 0) && 
                      (addr_store % 64 == 0);

    if (remain_row >= SIMD_32_LEN && (end - t) >= SIMD_32_LEN && all_aligned)
    {
        hthread_printf("vector\n");
        size_t row_bytes = row_size * sizeof(float);
 
        size_t off_up  = (size_t)(i - 1) * nj + (j - 1);
        size_t off_md  = (size_t) i       * nj + (j - 1);
        size_t off_dn  = (size_t)(i + 1) * nj + (j - 1);

        /* 使用三条 vector_load 加载三行数据到缓冲区 */
        vector_load(A + off_up, buf, row_bytes);                    // 上行
        vector_load(A + off_md, buf + row_size, row_bytes);         // 中行  
        vector_load(A + off_dn, buf + 2 * row_size, row_bytes);     // 下行

        /* 重置 res_vec 为 0.0，防止累加错误 */
        lvector float res_vec = (lvector float)vec_svbcast(0.0);
 
        /* 使用 vec_ld 从缓冲区读取九个向量并进行计算 */
        res_vec = vec_mula(vec_ld(0, buf + 0),                vc11, res_vec);  // up_l
        res_vec = vec_mula(vec_ld(0, buf + 1),                vc21, res_vec);  // up_m
        res_vec = vec_mula(vec_ld(0, buf + 2),                vc31, res_vec);  // up_r

        res_vec = vec_mula(vec_ld(0, buf + row_size + 0),     vc12, res_vec);  // mid_l
        res_vec = vec_mula(vec_ld(0, buf + row_size + 1),     vc22, res_vec);  // mid_m
        res_vec = vec_mula(vec_ld(0, buf + row_size + 2),     vc32, res_vec);  // mid_r
 
        res_vec = vec_mula(vec_ld(0, buf + 2 * row_size + 0), vc13, res_vec);  // lo_l
        res_vec = vec_mula(vec_ld(0, buf + 2 * row_size + 1), vc23, res_vec);  // lo_m
        res_vec = vec_mula(vec_ld(0, buf + 2 * row_size + 2), vc33, res_vec);  // lo_r

        /* 写回结果：vector_store( src_buf , dst_global , bytes ) */
        vector_store(&res_vec, B + i * nj + j, sizeof(float) * SIMD_32_LEN);
        
        t += SIMD_32_LEN;      /* 吞掉 16 个 task */
    }
    else
    {
        /* ========== 标量尾部 ================================ */
        float v11 = A[(i - 1) * nj + (j - 1)];
        float v12 = A[(i - 1) * nj +  j     ];
        float v13 = A[(i - 1) * nj + (j + 1)];
 
        float v21 = A[ i      * nj + (j - 1)];
        float v22 = A[ i      * nj +  j     ];
        float v23 = A[ i      * nj + (j + 1)];
 
        float v31 = A[(i + 1) * nj + (j - 1)];
        float v32 = A[(i + 1) * nj +  j     ];
        float v33 = A[(i + 1) * nj + (j + 1)];
 
        B[i * nj + j] =
            0.2 * v11 + 0.5 * v12 - 0.8 * v13
            - 0.3 * v21 + 0.6 * v22 - 0.9 * v23
            + 0.4 * v31 + 0.7 * v32 + 0.10 * v33;
 
        ++t;                 /* 只解决 1 个 task */
    }
}
    vector_free(buf);
}

#include "../2DCONV/kernel_vec.h"//向量化后存储文件
#include "../2DCONV/kernel_cache_llm.h"//SM缓存优化文件