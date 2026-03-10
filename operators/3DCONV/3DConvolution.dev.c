#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/compute_tool.h"
#include "../common/prof_event.h"
__global__ void convolution3D_kernel(int ni, int nj, int nk, int i, double *A, double *B) {
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    int total_elements = (nj - 2) * (nk - 2); // 修正为有效区域
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 2) + 1;
        int k = idx % (nk - 2) + 1;
        int idx_B = i * (nk * nj) + j * nk + k;
        B[idx_B] = 
            c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + 
            c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] +
            c21 * A[(i - 1) * (nk * nj) + j * nk + (k - 1)] + 
            c23 * A[(i + 1) * (nk * nj) + j * nk + (k - 1)] +
            c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (k - 1)] + 
            c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (k - 1)] +
            c12 * A[i * (nk * nj) + (j - 1) * nk + k] + 
            c22 * A[i * (nk * nj) + j * nk + k] +
            c32 * A[i * (nk * nj) + (j + 1) * nk + k] + 
            c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)] +
            c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)] + 
            c21 * A[(i - 1) * (nk * nj) + j * nk + (k + 1)] +
            c23 * A[(i + 1) * (nk * nj) + j * nk + (k + 1)] + 
            c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)] +
            c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)];
    }
}

#ifdef MIni_DATASET
__global__ void convolution3D_kernel_cache(int ni, int nj, int nk, int i, double *A, double *B)
{
    // 加速器架构参数
    int total_threads = get_group_size();
    int thread_id = get_thread_id();

    // 计算当前线程处理的(j,k)范围
    int total_elements = (nj - 1) * (nk - 1);
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    // 关键边界计算
    int j_start = start_idx / (nk - 1) + 1;
    int k_start = start_idx % (nk - 1) + 1;
    int i_min = (i - 1 >= 0) ? (i - 1) : 0;
    int i_max = (i + 1 < ni) ? (i + 1) : (ni - 1);

    // 计算缓存区间
    int j_end = (end_idx - 1) / (nk - 1) + 1;
    int k_end = (end_idx - 1) % (nk - 1) + 1;
    int A_min_addr = i_min * (nj * nk) + (j_start - 1) * nk + (k_start - 1);
    int A_max_addr = i_max * (nj * nk) + (j_end + 1) * nk + (k_end + 1);
    int cache_size = (A_max_addr - A_min_addr + 1) * sizeof(double);

    // 初始化缓存
    double *A_p = A; // 保存原始指针
    CACHEb_IniT(A, double, &A[A_min_addr], 0, cache_size);
    CACHEb_IniT(B, double, &B[i * (nk * nj) + j_start * nk + k_start], 0, (end_idx - start_idx) * sizeof(double));

    // 常量系数
    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    // 预计算层偏移基址 - 这样在循环中就不用重复计算了
    int i_minus_1_offset = (i - 1 - i_min) * nj * nk;
    int i_offset = (i - i_min) * nj * nk;
    int i_plus_1_offset = (i + 1 - i_min) * nj * nk;

    // j、k偏移基准 - 避免重复计算(j_start-1)和(k_start-1)
    int j_base = -(j_start - 1) * nk;
    int k_base = -(k_start - 1);

    double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, tmp_B;

    // 遍历当前线程负责的(j,k)位置
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // 计算j和k在缓存中的偏移
        int j_offset = j * nk + j_base;
        int k_offset = k + k_base;

        // i-1层访问 - 使用预计算的基址
        CACHEb_RD(A, &A[i_minus_1_offset + (j_offset - nk) + (k_offset - 1)], a1); // [i-1][j-1][k-1]
        CACHEb_RD(A, &A[i_minus_1_offset + (j_offset - nk) + (k_offset + 1)], a2); // [i-1][j-1][k+1]
        CACHEb_RD(A, &A[i_minus_1_offset + j_offset + (k_offset + 1)], a3);        // [i-1][j][k+1]
        CACHEb_RD(A, &A[i_minus_1_offset + (j_offset + nk) + (k_offset + 1)], a4); // [i-1][j+1][k+1]

        // i层访问
        CACHEb_RD(A, &A[i_offset + (j_offset - nk) + k_offset], a5); // [i][j-1][k]
        CACHEb_RD(A, &A[i_offset + j_offset + k_offset], a6);        // [i][j][k]
        CACHEb_RD(A, &A[i_offset + (j_offset + nk) + k_offset], a7); // [i][j+1][k]

        // i+1层访问
        CACHEb_RD(A, &A[i_plus_1_offset + (j_offset - nk) + (k_offset - 1)], a8);  // [i+1][j-1][k-1]
        CACHEb_RD(A, &A[i_plus_1_offset + (j_offset - nk) + (k_offset + 1)], a9);  // [i+1][j-1][k+1]
        CACHEb_RD(A, &A[i_plus_1_offset + j_offset + (k_offset + 1)], a10);        // [i+1][j][k+1]
        CACHEb_RD(A, &A[i_plus_1_offset + (j_offset + nk) + (k_offset + 1)], a11); // [i+1][j+1][k+1]

        // 计算B值 (保持不变)
        tmp_B = c11 * a1 + c21 * a1 + c31 * a1 + c11 * a2 + c21 * a3 + c31 * a4 + c12 * a5 + c22 * a6 + c32 * a7 +
                c13 * a8 + c23 * a8 + c33 * a8 + c13 * a9 + c23 * a10 + c33 * a11;

        // 写入B (使用简化的偏移计算)
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }

    CACHEb_FLUSH(B);
    CACHEb_INVALID(A);
}
#endif

#ifdef SMALL_DATASET
__global__ void convolution3D_kernel_cache(int ni, int nj, int nk, int i, double *A, double *B)
{
    // 加速器架构参数
    int total_threads = get_group_size();
    int thread_id = get_thread_id();

    // 计算当前线程处理的(j,k)范围
    int total_elements = (nj - 1) * (nk - 1);
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    // 关键边界计算
    int j_start = start_idx / (nk - 1) + 1;
    int k_start = start_idx % (nk - 1) + 1;

    // 初始化缓存
    CACHEs_IniT(A, double, A, 0, 15);
    CACHEb_IniT(B, double, &B[i * (nk * nj) + j_start * nk + k_start], 0, (end_idx - start_idx) * sizeof(double));

    // 常量系数
    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, tmp_B;

    // 遍历当前线程负责的(j,k)位置
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i-1层访问 - 使用预计算的基址
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a1); // [i-1][j-1][k-1]
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a2); // [i-1][j-1][k+1]

        tmp_B = c11 * a1 + c21 * a1 + c31 * a1 + c11 * a2;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a3); // [i-1][j][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c21 * a3;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a4); // [i-1][j+1][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c31 * a4;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j - 1) * nk + (k + 0)], a5); // [i][j-1][k]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c12 * a5;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 0) * nk + (k + 0)], a6); // [i][j][k]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c22 * a6;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 1) * nk + (k + 0)], a7); // [i][j+1][k]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c32 * a7;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a8); // [i+1][j-1][k-1]
        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a9); // [i+1][j-1][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c13 * a8 + c23 * a8 + c33 * a8 + c13 * a9;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a10); // [i+1][j][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c23 * a10;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a11); // [i+1][j+1][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c33 * a11;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }

    CACHEb_FLUSH(B);
    CACHEs_INVALID(A);
}
#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */
__global__ void convolution3D_kernel_cache(int ni, int nj, int nk, int i, double *A, double *B)
{
    // 加速器架构参数
    int total_threads = get_group_size();
    int thread_id = get_thread_id();

    // 计算当前线程处理的(j,k)范围
    int total_elements = (nj - 1) * (nk - 1);
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    // 关键边界计算
    int j_start = start_idx / (nk - 1) + 1;
    int k_start = start_idx % (nk - 1) + 1;

    // 初始化缓存
    CACHEs_IniT(A, double, A, 0, 15);
    CACHEb_IniT(B, double, &B[i * (nk * nj) + j_start * nk + k_start], 0, (end_idx - start_idx) * sizeof(double));

    // 常量系数
    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, tmp_B;

    // 遍历当前线程负责的(j,k)位置
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i-1层访问 - 使用预计算的基址
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a1); // [i-1][j-1][k-1]
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a2); // [i-1][j-1][k+1]

        tmp_B = c11 * a1 + c21 * a1 + c31 * a1 + c11 * a2;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a3); // [i-1][j][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c21 * a3;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a4); // [i-1][j+1][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c31 * a4;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j - 1) * nk + (k + 0)], a5); // [i][j-1][k]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c12 * a5;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 0) * nk + (k + 0)], a6); // [i][j][k]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c22 * a6;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 1) * nk + (k + 0)], a7); // [i][j+1][k]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c32 * a7;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a8); // [i+1][j-1][k-1]
        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a9); // [i+1][j-1][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c13 * a8 + c23 * a8 + c33 * a8 + c13 * a9;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a10); // [i+1][j][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c23 * a10;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a11); // [i+1][j+1][k+1]

        CACHEb_RD(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
        tmp_B += c33 * a11;
        CACHEb_WT(B, &B[(j - j_start) * nk + (k - k_start)], tmp_B);
    }

    CACHEb_FLUSH(B);
    CACHEs_INVALID(A);
}
#endif

#ifdef LARGE_DATASET
__global__ void convolution3D_kernel_cache(int ni, int nj, int nk, int i, double *A, double *B)
{
    // 加速器架构参数
    int total_threads = get_group_size();
    int thread_id = get_thread_id();

    // 计算当前线程处理的(j,k)范围
    int total_elements = (nj - 1) * (nk - 1);
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    // 关键边界计算
    int j_start = start_idx / (nk - 1) + 1;
    int k_start = start_idx % (nk - 1) + 1;

    // 初始化缓存
    CACHEs_IniT(A, double, A, 0, 13);
    CACHEs_IniT(B, double, B, 0, 15);
    // CACHEb_IniT(B, double, &B[i * (nk * nj) + j_start * nk + k_start], 0, (end_idx - start_idx) * sizeof(double));

    // 常量系数
    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, tmp_B;

    // 遍历当前线程负责的(j,k)位置
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i-1层访问 - 使用预计算的基址
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a1); // [i-1][j-1][k-1]
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a2); // [i-1][j-1][k+1]

        tmp_B = c11 * a1 + c21 * a1 + c31 * a1 + c11 * a2;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a3); // [i-1][j][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c21 * a3;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a4); // [i-1][j+1][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c31 * a4;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j - 1) * nk + (k + 0)], a5); // [i][j-1][k]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c12 * a5;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 0) * nk + (k + 0)], a6); // [i][j][k]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c22 * a6;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 1) * nk + (k + 0)], a7); // [i][j+1][k]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c32 * a7;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a8); // [i+1][j-1][k-1]
        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a9); // [i+1][j-1][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c13 * a8 + c23 * a8 + c33 * a8 + c13 * a9;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a10); // [i+1][j][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c23 * a10;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a11); // [i+1][j+1][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c33 * a11;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }

    CACHEs_FLUSH(B);
    CACHEs_INVALID(A);
}
#endif

#ifdef EXTRALARGE_DATASET
__global__ void convolution3D_kernel_cache(int ni, int nj, int nk, int i, double *A, double *B)
{
    // 加速器架构参数
    int total_threads = get_group_size();
    int thread_id = get_thread_id();

    // 计算当前线程处理的(j,k)范围
    int total_elements = (nj - 1) * (nk - 1);
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    // 关键边界计算
    int j_start = start_idx / (nk - 1) + 1;
    int k_start = start_idx % (nk - 1) + 1;

    // 初始化缓存
    CACHEs_IniT(A, double, A, 0, 13);
    CACHEs_IniT(B, double, B, 0, 15);
    // CACHEb_IniT(B, double, &B[i * (nk * nj) + j_start * nk + k_start], 0, (end_idx - start_idx) * sizeof(double));

    // 常量系数
    double c11 = +2, c12 = -3, c13 = +4, c21 = +5, c22 = +6, c23 = +7, c31 = -8, c32 = -9, c33 = +10;

    double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, tmp_B;

    // 遍历当前线程负责的(j,k)位置
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i-1层访问 - 使用预计算的基址
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a1); // [i-1][j-1][k-1]
        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a2); // [i-1][j-1][k+1]

        tmp_B = c11 * a1 + c21 * a1 + c31 * a1 + c11 * a2;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a3); // [i-1][j][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c21 * a3;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a4); // [i-1][j+1][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c31 * a4;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j - 1) * nk + (k + 0)], a5); // [i][j-1][k]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c12 * a5;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        // i层访问
        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 0) * nk + (k + 0)], a6); // [i][j][k]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c22 * a6;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 0) * (nk * nj) + (j + 1) * nk + (k + 0)], a7); // [i][j+1][k]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c32 * a7;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)], a8); // [i+1][j-1][k-1]
        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)], a9); // [i+1][j-1][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c13 * a8 + c23 * a8 + c33 * a8 + c13 * a9;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 0) * nk + (k + 1)], a10); // [i+1][j][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c23 * a10;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }
    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 1) + 1;
        int k = idx % (nk - 1) + 1;

        CACHEs_RD(A, &A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)], a11); // [i+1][j+1][k+1]

        CACHEb_RD(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
        tmp_B += c33 * a11;
        CACHEb_WT(B, &B[i * (nk * nj) + j * nk + k], tmp_B);
    }

    CACHEs_FLUSH(B);
    CACHEs_INVALID(A);
}
#endif

/*--------------------------------------------------------------*/
/* 一条向量 16 个 double                                         */
#define SIMD_LEN 16
#define VEC_BYTES 128        /* 16 × 8 = 128 byte              */
/*--------------------------------------------------------------*/
__global__ void convolution3D_kernel_vec(
    int ni, int nj, int nk,    /* 全尺寸                         */
    int i_fixed,               /* 本 kernel 负责的 i 面 (1~ni-2) */
    double *A, double *B)
{
    const int thread_id  = get_thread_id();
    const int group_size = get_group_size();

    /* ------------------ 1. 常量广播 -------------------------- */
    lvector double vc11 = (lvector double)vec_svbcast( 2.0);   /*  c11 */
    lvector double vc12 = (lvector double)vec_svbcast(-3.0);   /*  c12 */
    lvector double vc13 = (lvector double)vec_svbcast( 4.0);   /*  c13 */

    lvector double vc21 = (lvector double)vec_svbcast( 5.0);   /*  c21 */
    lvector double vc22 = (lvector double)vec_svbcast( 6.0);   /*  c22 */
    lvector double vc23 = (lvector double)vec_svbcast( 7.0);   /*  c23 */

    lvector double vc31 = (lvector double)vec_svbcast(-8.0);   /*  c31 */
    lvector double vc32 = (lvector double)vec_svbcast(-9.0);   /*  c32 */
    lvector double vc33 = (lvector double)vec_svbcast(10.0);   /*  c33 */

    /* ------------------ 2. 任务划分 -------------------------- */
    const int nk_inner    = nk - 2;               /* k 有效长度           */
    const int total_tasks = (nj - 2) * nk_inner;  /* 整个 (j,k) 平面任务数 */
    if (total_tasks <= 0)  return;

    const int base_tasks = total_tasks / group_size;
    const int remainder  = total_tasks % group_size;

    int start = (thread_id < remainder)
              ? thread_id * (base_tasks + 1)
              : remainder * (base_tasks + 1) +
                (thread_id - remainder) * base_tasks;
    int end   = start + ((thread_id < remainder) ? (base_tasks + 1)
                                                 :  base_tasks);

    /* ------------------ 3. 分配 27 个向量缓冲 ---------------- */
    lvector double *buf = (lvector double *)
            vector_malloc(sizeof(lvector double) * 27);

    lvector double *im1_jm1_km1 = buf +  0;  /* i-1, j-1, k-1 */
    lvector double *im1_jm1_k0  = buf +  1;
    lvector double *im1_jm1_kp1 = buf +  2;

    lvector double *im1_j0_km1  = buf +  3;
    lvector double *im1_j0_k0   = buf +  4;
    lvector double *im1_j0_kp1  = buf +  5;

    lvector double *im1_jp1_km1 = buf +  6;
    lvector double *im1_jp1_k0  = buf +  7;
    lvector double *im1_jp1_kp1 = buf +  8;

    lvector double * i_jm1_km1  = buf +  9;  /*  i , j-1,k-1 */
    lvector double * i_jm1_k0   = buf + 10;
    lvector double * i_jm1_kp1  = buf + 11;

    lvector double * i_j0_km1   = buf + 12;
    lvector double * i_j0_k0    = buf + 13;  /*  中心元素     */
    lvector double * i_j0_kp1   = buf + 14;

    lvector double * i_jp1_km1  = buf + 15;
    lvector double * i_jp1_k0   = buf + 16;
    lvector double * i_jp1_kp1  = buf + 17;

    lvector double *ip1_jm1_km1 = buf + 18;  /* i+1 ... */
    lvector double *ip1_jm1_k0  = buf + 19;
    lvector double *ip1_jm1_kp1 = buf + 20;

    lvector double *ip1_j0_km1  = buf + 21;
    lvector double *ip1_j0_k0   = buf + 22;
    lvector double *ip1_j0_kp1  = buf + 23;

    lvector double *ip1_jp1_km1 = buf + 24;
    lvector double *ip1_jp1_k0  = buf + 25;
    lvector double *ip1_jp1_kp1 = buf + 26;

    /* ------------ 4. 主循环：线性扫描所有 task --------------- */
    for (int t = start; t < end; )
    {
        int j = 1 + t / nk_inner;   /* 行号 (1~nj-2) */
        int k = 1 + t % nk_inner;   /* 列号 (1~nk-2) */

        int remain_row = nk_inner - (k - 1); /* 本行剩余 task */

        if (remain_row >= SIMD_LEN && (end - t) >= SIMD_LEN)
        {
            size_t off_im1 = (size_t)(i_fixed - 1) * nj * nk;
            size_t off_i   = (size_t) i_fixed      * nj * nk;
            size_t off_ip1 = (size_t)(i_fixed + 1) * nj * nk;

            size_t off_jm1 = (size_t)(j - 1) * nk;
            size_t off_j   = (size_t) j      * nk;
            size_t off_jp1 = (size_t)(j + 1) * nk;

            size_t k_left  = (size_t)(k - 1);

            /* --- i-1 slice --- */
            vector_load(A + off_im1 + off_jm1 + k_left, im1_jm1_km1, VEC_BYTES);
            vector_load(A + off_im1 + off_jm1 + 1 + k_left, im1_jm1_k0, VEC_BYTES);
            vector_load(A + off_im1 + off_jm1 + 2 + k_left, im1_jm1_kp1, VEC_BYTES);

            vector_load(A + off_im1 + off_j + k_left, im1_j0_km1, VEC_BYTES);
            vector_load(A + off_im1 + off_j + 1 + k_left, im1_j0_k0, VEC_BYTES);
            vector_load(A + off_im1 + off_j + 2 + k_left, im1_j0_kp1, VEC_BYTES);

            vector_load(A + off_im1 + off_jp1 + k_left, im1_jp1_km1, VEC_BYTES);
            vector_load(A + off_im1 + off_jp1 + 1 + k_left, im1_jp1_k0, VEC_BYTES);
            vector_load(A + off_im1 + off_jp1 + 2 + k_left, im1_jp1_kp1, VEC_BYTES);

            /* --- i slice --- */
            vector_load(A + off_i + off_jm1 + k_left, i_jm1_km1, VEC_BYTES);
            vector_load(A + off_i + off_jm1 + 1 + k_left, i_jm1_k0, VEC_BYTES);
            vector_load(A + off_i + off_jm1 + 2 + k_left, i_jm1_kp1, VEC_BYTES);

            vector_load(A + off_i + off_j + k_left, i_j0_km1, VEC_BYTES);
            vector_load(A + off_i + off_j + 1 + k_left, i_j0_k0, VEC_BYTES);
            vector_load(A + off_i + off_j + 2 + k_left, i_j0_kp1, VEC_BYTES);

            vector_load(A + off_i + off_jp1 + k_left, i_jp1_km1, VEC_BYTES);
            vector_load(A + off_i + off_jp1 + 1 + k_left, i_jp1_k0, VEC_BYTES);
            vector_load(A + off_i + off_jp1 + 2 + k_left, i_jp1_kp1, VEC_BYTES);

            /* --- i+1 slice --- */
            vector_load(A + off_ip1 + off_jm1 + k_left, ip1_jm1_km1, VEC_BYTES);
            vector_load(A + off_ip1 + off_jm1 + 1 + k_left, ip1_jm1_k0, VEC_BYTES);
            vector_load(A + off_ip1 + off_jm1 + 2 + k_left, ip1_jm1_kp1, VEC_BYTES);

            vector_load(A + off_ip1 + off_j + k_left, ip1_j0_km1, VEC_BYTES);
            vector_load(A + off_ip1 + off_j + 1 + k_left, ip1_j0_k0, VEC_BYTES);
            vector_load(A + off_ip1 + off_j + 2 + k_left, ip1_j0_kp1, VEC_BYTES);

            vector_load(A + off_ip1 + off_jp1 + k_left, ip1_jp1_km1, VEC_BYTES);
            vector_load(A + off_ip1 + off_jp1 + 1 + k_left, ip1_jp1_k0, VEC_BYTES);
            vector_load(A + off_ip1 + off_jp1 + 2 + k_left, ip1_jp1_kp1, VEC_BYTES);

            /* --- 运算 --- */
            lvector double res_vec = (lvector double)vec_svbcast(0.0);

            /* slice k-1 */
            res_vec = vec_mula(*im1_jm1_km1, vc11, res_vec);
            res_vec = vec_mula(*ip1_jm1_km1, vc13, res_vec);
            res_vec = vec_mula(*im1_j0_km1, vc21, res_vec);
            res_vec = vec_mula(*ip1_j0_km1, vc23, res_vec);
            res_vec = vec_mula(*im1_jp1_km1, vc31, res_vec);
            res_vec = vec_mula(*ip1_jp1_km1, vc33, res_vec);

            /* slice k */
            res_vec = vec_mula(*i_jm1_k0, vc12, res_vec);
            res_vec = vec_mula(*i_j0_k0, vc22, res_vec);
            res_vec = vec_mula(*i_jp1_k0, vc32, res_vec);

            /* slice k+1 */
            res_vec = vec_mula(*im1_jm1_kp1, vc11, res_vec);
            res_vec = vec_mula(*ip1_jm1_kp1, vc13, res_vec);
            res_vec = vec_mula(*im1_j0_kp1, vc21, res_vec);
            res_vec = vec_mula(*ip1_j0_kp1, vc23, res_vec);
            res_vec = vec_mula(*im1_jp1_kp1, vc31, res_vec);
            res_vec = vec_mula(*ip1_jp1_kp1, vc33, res_vec);

            vector_store(&res_vec,
                         B + off_i + off_j + k,
                         VEC_BYTES);

            t += SIMD_LEN;
        }
        else
        {
            /* ----------------- 标量尾部处理 ------------------- */
            double sum = 0.0;
            /* 按原公式累加 27 个元素 */
            sum +=  2.0 * A[(i_fixed - 1)*nj*nk + (j - 1)*nk + (k - 1)];
            sum +=  4.0 * A[(i_fixed + 1)*nj*nk + (j - 1)*nk + (k - 1)];
            sum +=  5.0 * A[(i_fixed - 1)*nj*nk +  j     *nk + (k - 1)];
            sum +=  7.0 * A[(i_fixed + 1)*nj*nk +  j     *nk + (k - 1)];
            sum += -8.0 * A[(i_fixed - 1)*nj*nk + (j + 1)*nk + (k - 1)];
            sum += 10.0 * A[(i_fixed + 1)*nj*nk + (j + 1)*nk + (k - 1)];

            sum += -3.0 * A[ i_fixed      *nj*nk + (j - 1)*nk +  k     ];
            sum +=  6.0 * A[ i_fixed      *nj*nk +  j      *nk +  k     ];
            sum += -9.0 * A[ i_fixed      *nj*nk + (j + 1)*nk +  k     ];

            sum +=  2.0 * A[(i_fixed - 1)*nj*nk + (j - 1)*nk + (k + 1)];
            sum +=  4.0 * A[(i_fixed + 1)*nj*nk + (j - 1)*nk + (k + 1)];
            sum +=  5.0 * A[(i_fixed - 1)*nj*nk +  j     *nk + (k + 1)];
            sum +=  7.0 * A[(i_fixed + 1)*nj*nk +  j     *nk + (k + 1)];
            sum += -8.0 * A[(i_fixed - 1)*nj*nk + (j + 1)*nk + (k + 1)];
            sum += 10.0 * A[(i_fixed + 1)*nj*nk + (j + 1)*nk + (k + 1)];

            B[i_fixed*nj*nk + j*nk + k] = sum;
            ++t;
        }
    }

    vector_free(buf);
}
#include "../3DCONV/kernel_vec.h"//大模型生成的存储文件
#include "../3DCONV/kernel_cache_llm.h"//SM缓存优化文件