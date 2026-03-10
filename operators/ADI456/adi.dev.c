#include <stdint.h>
#include <compiler/m3000.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"

__global__ void adi_kernel1(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            X[i1 * n + i2] = X[i1 * n + i2] - X[i1 * n + (i2 - 1)] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
            B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
        }
    }
}

__global__ void adi_kernel2(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    for (int i1 = start_row; i1 < end_row; i1++) {
        X[i1 * n + (n - 1)] = X[i1 * n + (n - 1)] / B[i1 * n + (n - 1)];
    }
}

__global__ void adi_kernel3(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 0; i2 < n - 2; i2++) {
            X[i1 * n + (n - i2 - 2)] =
                (X[i1 * n + (n - i2 - 2)] - X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) /
                B[i1 * n + (n - i2 - 3)];
        }
    }
}

__global__ void adi_kernel4(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    for (int i2 = start_col; i2 < end_col; i2++) {
        X[i1 * n + i2] = X[i1 * n + i2] - X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
        B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
    }
}

__global__ void adi_kernel5(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    for (int i2 = start_col; i2 < end_col; i2++) {
        X[(n - 1) * n + i2] = X[(n - 1) * n + i2] / B[(n - 1) * n + i2];
    }
}

__global__ void adi_kernel6(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    for (int i2 = start_col; i2 < end_col; i2++) {
        X[(n - 2 - i1) * n + i2] =
            (X[(n - 2 - i1) * n + i2] - X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2]) / B[(n - 2 - i1) * n + i2];
    }
}

#ifdef MINI_DATASET
__global__ void adi_kernel1_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEb_INIT(A, double, &A[start_row * n], 0, (end_row - start_row) * n * sizeof(double));
    CACHEb_INIT(B, double, &B[start_row * n], 0, (end_row - start_row) * n * sizeof(double));
    CACHEs_INIT(X, double, X, 0, 13);
    double tmp_A_i1_i2, tmp_B_i1_i2, tmp_B_i1_i2_1, tmp_X_i1_i2_1, tmp_X_i1_i2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEb_RD(A, &A[(i1 - start_row) * n + i2], tmp_A_i1_i2);
            CACHEb_RD(B, &B[(i1 - start_row) * n + i2], tmp_B_i1_i2);
            CACHEb_RD(B, &B[(i1 - start_row) * n + i2 - 1], tmp_B_i1_i2_1);
            CACHEs_RD(X, &X[i1 * n + i2 - 1], tmp_X_i1_i2_1);
            CACHEs_RD(X, &X[i1 * n + i2], tmp_X_i1_i2);
            tmp_X_i1_i2 = tmp_X_i1_i2 - tmp_X_i1_i2_1 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
            tmp_B_i1_i2 = tmp_B_i1_i2 - tmp_A_i1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
            CACHEb_WT(B, &B[(i1 - start_row) * n + i2], tmp_B_i1_i2);
            CACHEs_WT(X, &X[i1 * n + i2], tmp_X_i1_i2);
        }
    }
    CACHEb_FLUSH(B);
    CACHEs_FLUSH(X);
    CACHEb_INVALID(A);
}

__global__ void adi_kernel2_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEb_INIT(X, double, &X[start_row * n], 0, (end_row - start_row) * n * sizeof(double));
    CACHEb_INIT(B, double, &B[start_row * n], 0, (end_row - start_row) * n * sizeof(double));
    double tmp_X_i1_N_1, tmp_B_i1_N_1;
    for (int i1 = start_row; i1 < end_row; i1++) {
        CACHEb_RD(X, &X[(i1 - start_row) * n + n - 1], tmp_X_i1_N_1);
        CACHEb_RD(B, &B[(i1 - start_row) * n + n - 1], tmp_B_i1_N_1);
        tmp_X_i1_N_1 = tmp_X_i1_N_1 / tmp_B_i1_N_1;
        CACHEb_WT(X, &X[(i1 - start_row) * n + n - 1], tmp_X_i1_N_1);
    }
    CACHEb_FLUSH(X);
    CACHEb_INVALID(B);
}

__global__ void adi_kernel3_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEb_INIT(X, double, &X[start_row * n], 0, (end_row - start_row) * n * sizeof(double));
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEs_INIT(A, double, A, 0, 14);
    double tmp_X_i1_i2_2, tmp_X_i1_i2_3, tmp_A_i1_i2_3, tmp_B_i1_i2_3;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 0; i2 < n - 2; i2++) {
            CACHEb_RD(X, &X[(i1 - start_row) * n + (n - i2 - 2)], tmp_X_i1_i2_2);
            CACHEb_RD(X, &X[(i1 - start_row) * n + (n - i2 - 3)], tmp_X_i1_i2_3);
            CACHEs_RD(A, &A[i1 * n + (n - i2 - 3)], tmp_A_i1_i2_3);
            CACHEb_RD(B, &B[i1 * n + (n - i2 - 3)], tmp_B_i1_i2_3);
            tmp_X_i1_i2_2 = (tmp_X_i1_i2_2 - tmp_X_i1_i2_3 * tmp_A_i1_i2_3) / tmp_B_i1_i2_3;
            CACHEb_WT(X, &X[(i1 - start_row) * n + (n - i2 - 2)], tmp_X_i1_i2_2);
        }
    }
    CACHEb_FLUSH(X);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void adi_kernel4_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    double *B_p = B;
    CACHEb_INIT(X, double, &X[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(B, double, &B[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[i1 * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_X_i1_1_i2, tmp_B_i1_i2_1, tmp_A_i1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_i1_1_i2);
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_i1_i2_1);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_i1_i2);
        X_p[i1 * n + i2] = X_p[i1 * n + i2] - tmp_X_i1_1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
        B_p[i1 * n + i2] = B_p[i1 * n + i2] - tmp_A_i1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
    }
    CACHEb_INVALID(X);
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
}

__global__ void adi_kernel5_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    CACHEb_INIT(B, double, &B[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_1_i2, tmp_X_N_1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_1_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_1_i2);
        tmp_X_N_1_i2 = tmp_X_N_1_i2 / tmp_B_N_1_i2;
        CACHEb_WT(X, &X[i2 - start_col], tmp_X_N_1_i2);
    }
    CACHEb_INVALID(B);
    CACHEb_FLUSH(X);
}

__global__ void adi_kernel6_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    CACHEb_INIT(B, double, &B[(n - 2 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_2_i2, tmp_X_N_3_i2, tmp_A_N_3_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_2_i2);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_N_3_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_3_i2);
        X_p[(n - 2 - i1) * n + i2] = (X_p[(n - 2 - i1) * n + i2] - tmp_X_N_3_i2 * tmp_A_N_3_i2) / tmp_B_N_2_i2;
    }
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
    CACHEb_INVALID(X);
}
#endif

#ifdef SMALL_DATASET

__global__ void adi_kernel1_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEs_INIT(X, double, X, 0, 13);
    double tmp_A;
    double tmp_B1, tmp_B2;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + i2], tmp_A);
            CACHEs_RD(B, &B[i1 * n + i2], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_RD(B, &B[i1 * n + i2 - 1], tmp_B2);
            CACHEs_RD(X, &X[i1 * n + i2 - 1], tmp_X2);
            tmp_X1 = tmp_X1 - tmp_X2 * tmp_A / tmp_B2;
            tmp_B1 = tmp_B1 - tmp_A * tmp_A / tmp_B2;
            CACHEs_WT(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_WT(B, &B[i1 * n + i2], tmp_B1);
        }
    }
    CACHEs_INVALID(A);
    CACHEs_FLUSH(B);
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel2_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(X, double, X, 0, 15);
    double tmp_X_i1_N_1;
    for (int i1 = start_row; i1 < end_row; i1++) {
        CACHEs_RD(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
        tmp_X_i1_N_1 = tmp_X_i1_N_1 / B[i1 * n + (n - 1)];
        CACHEs_WT(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
    }
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel3_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 12);
    CACHEs_INIT(B, double, B, 0, 12);
    CACHEs_INIT(X, double, X, 0, 14);
    double tmp_A;
    double tmp_B1;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + (n - i2 - 3)], tmp_A);
            CACHEs_RD(B, &B[i1 * n + (n - i2 - 3)], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 3)], tmp_X2);
            tmp_X1 = (tmp_X1 - tmp_X2 * tmp_A) / tmp_B1;
            CACHEs_WT(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
        }
    }
    CACHEs_FLUSH(X);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void adi_kernel4_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    double *B_p = B;
    CACHEb_INIT(X, double, &X[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(B, double, &B[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[i1 * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_X_i1_1_i2, tmp_B_i1_i2_1, tmp_A_i1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_i1_1_i2);
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_i1_i2_1);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_i1_i2);
        X_p[i1 * n + i2] = X_p[i1 * n + i2] - tmp_X_i1_1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
        B_p[i1 * n + i2] = B_p[i1 * n + i2] - tmp_A_i1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
    }
    CACHEb_INVALID(X);
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
}

__global__ void adi_kernel5_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    CACHEb_INIT(B, double, &B[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_1_i2, tmp_X_N_1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_1_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_1_i2);
        tmp_X_N_1_i2 = tmp_X_N_1_i2 / tmp_B_N_1_i2;
        CACHEb_WT(X, &X[i2 - start_col], tmp_X_N_1_i2);
    }
    CACHEb_INVALID(B);
    CACHEb_FLUSH(X);
}

__global__ void adi_kernel6_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    CACHEb_INIT(B, double, &B[(n - 2 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_2_i2, tmp_X_N_3_i2, tmp_A_N_3_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_2_i2);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_N_3_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_3_i2);
        X_p[(n - 2 - i1) * n + i2] = (X_p[(n - 2 - i1) * n + i2] - tmp_X_N_3_i2 * tmp_A_N_3_i2) / tmp_B_N_2_i2;
    }
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
    CACHEb_INVALID(X);
}

#endif

#ifdef STANDARD_DATASET /* Default if unspecified. */

__global__ void adi_kernel1_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEs_INIT(X, double, X, 0, 13);
    double tmp_A;
    double tmp_B1, tmp_B2;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + i2], tmp_A);
            CACHEs_RD(B, &B[i1 * n + i2], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_RD(B, &B[i1 * n + i2 - 1], tmp_B2);
            CACHEs_RD(X, &X[i1 * n + i2 - 1], tmp_X2);
            tmp_X1 = tmp_X1 - tmp_X2 * tmp_A / tmp_B2;
            tmp_B1 = tmp_B1 - tmp_A * tmp_A / tmp_B2;
            CACHEs_WT(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_WT(B, &B[i1 * n + i2], tmp_B1);
        }
    }
    CACHEs_INVALID(A);
    CACHEs_FLUSH(B);
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel2_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(X, double, X, 0, 15);
    double tmp_X_i1_N_1;
    for (int i1 = start_row; i1 < end_row; i1++) {
        CACHEs_RD(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
        tmp_X_i1_N_1 = tmp_X_i1_N_1 / B[i1 * n + (n - 1)];
        CACHEs_WT(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
    }
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel3_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 12);
    CACHEs_INIT(B, double, B, 0, 12);
    CACHEs_INIT(X, double, X, 0, 14);
    double tmp_A;
    double tmp_B1;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + (n - i2 - 3)], tmp_A);
            CACHEs_RD(B, &B[i1 * n + (n - i2 - 3)], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 3)], tmp_X2);
            tmp_X1 = (tmp_X1 - tmp_X2 * tmp_A) / tmp_B1;
            CACHEs_WT(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
        }
    }
    CACHEs_FLUSH(X);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void adi_kernel4_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    double *B_p = B;
    CACHEb_INIT(X, double, &X[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(B, double, &B[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[i1 * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_X_i1_1_i2, tmp_B_i1_i2_1, tmp_A_i1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_i1_1_i2);
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_i1_i2_1);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_i1_i2);
        X_p[i1 * n + i2] = X_p[i1 * n + i2] - tmp_X_i1_1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
        B_p[i1 * n + i2] = B_p[i1 * n + i2] - tmp_A_i1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
    }
    CACHEb_INVALID(X);
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
}

__global__ void adi_kernel5_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    CACHEb_INIT(B, double, &B[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_1_i2, tmp_X_N_1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_1_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_1_i2);
        tmp_X_N_1_i2 = tmp_X_N_1_i2 / tmp_B_N_1_i2;
        CACHEb_WT(X, &X[i2 - start_col], tmp_X_N_1_i2);
    }
    CACHEb_INVALID(B);
    CACHEb_FLUSH(X);
}

__global__ void adi_kernel6_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    CACHEb_INIT(B, double, &B[(n - 2 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_2_i2, tmp_X_N_3_i2, tmp_A_N_3_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_2_i2);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_N_3_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_3_i2);
        X_p[(n - 2 - i1) * n + i2] = (X_p[(n - 2 - i1) * n + i2] - tmp_X_N_3_i2 * tmp_A_N_3_i2) / tmp_B_N_2_i2;
    }
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
    CACHEb_INVALID(X);
}

#endif

#ifdef LARGE_DATASET

__global__ void adi_kernel1_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEs_INIT(X, double, X, 0, 13);
    double tmp_A;
    double tmp_B1, tmp_B2;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + i2], tmp_A);
            CACHEs_RD(B, &B[i1 * n + i2], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_RD(B, &B[i1 * n + i2 - 1], tmp_B2);
            CACHEs_RD(X, &X[i1 * n + i2 - 1], tmp_X2);
            tmp_X1 = tmp_X1 - tmp_X2 * tmp_A / tmp_B2;
            tmp_B1 = tmp_B1 - tmp_A * tmp_A / tmp_B2;
            CACHEs_WT(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_WT(B, &B[i1 * n + i2], tmp_B1);
        }
    }
    CACHEs_INVALID(A);
    CACHEs_FLUSH(B);
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel2_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(X, double, X, 0, 15);
    double tmp_X_i1_N_1;
    for (int i1 = start_row; i1 < end_row; i1++) {
        CACHEs_RD(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
        tmp_X_i1_N_1 = tmp_X_i1_N_1 / B[i1 * n + (n - 1)];
        CACHEs_WT(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
    }
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel3_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 12);
    CACHEs_INIT(B, double, B, 0, 12);
    CACHEs_INIT(X, double, X, 0, 14);
    double tmp_A;
    double tmp_B1;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + (n - i2 - 3)], tmp_A);
            CACHEs_RD(B, &B[i1 * n + (n - i2 - 3)], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 3)], tmp_X2);
            tmp_X1 = (tmp_X1 - tmp_X2 * tmp_A) / tmp_B1;
            CACHEs_WT(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
        }
    }
    CACHEs_FLUSH(X);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void adi_kernel4_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    double *B_p = B;
    CACHEb_INIT(X, double, &X[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(B, double, &B[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[i1 * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_X_i1_1_i2, tmp_B_i1_i2_1, tmp_A_i1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_i1_1_i2);
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_i1_i2_1);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_i1_i2);
        X_p[i1 * n + i2] = X_p[i1 * n + i2] - tmp_X_i1_1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
        B_p[i1 * n + i2] = B_p[i1 * n + i2] - tmp_A_i1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
    }
    CACHEb_INVALID(X);
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
}

__global__ void adi_kernel5_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    CACHEb_INIT(B, double, &B[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_1_i2, tmp_X_N_1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_1_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_1_i2);
        tmp_X_N_1_i2 = tmp_X_N_1_i2 / tmp_B_N_1_i2;
        CACHEb_WT(X, &X[i2 - start_col], tmp_X_N_1_i2);
    }
    CACHEb_INVALID(B);
    CACHEb_FLUSH(X);
}

__global__ void adi_kernel6_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    CACHEb_INIT(B, double, &B[(n - 2 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_2_i2, tmp_X_N_3_i2, tmp_A_N_3_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_2_i2);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_N_3_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_3_i2);
        X_p[(n - 2 - i1) * n + i2] = (X_p[(n - 2 - i1) * n + i2] - tmp_X_N_3_i2 * tmp_A_N_3_i2) / tmp_B_N_2_i2;
    }
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
    CACHEb_INVALID(X);
}

#endif

#ifdef EXTRALARGE_DATASET
__global__ void adi_kernel1_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 14);
    CACHEs_INIT(B, double, B, 0, 14);
    CACHEs_INIT(X, double, X, 0, 13);
    double tmp_A;
    double tmp_B1, tmp_B2;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + i2], tmp_A);
            CACHEs_RD(B, &B[i1 * n + i2], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_RD(B, &B[i1 * n + i2 - 1], tmp_B2);
            CACHEs_RD(X, &X[i1 * n + i2 - 1], tmp_X2);
            tmp_X1 = tmp_X1 - tmp_X2 * tmp_A / tmp_B2;
            tmp_B1 = tmp_B1 - tmp_A * tmp_A / tmp_B2;
            CACHEs_WT(X, &X[i1 * n + i2], tmp_X1);
            CACHEs_WT(B, &B[i1 * n + i2], tmp_B1);
        }
    }
    CACHEs_INVALID(A);
    CACHEs_FLUSH(B);
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel2_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(X, double, X, 0, 15);
    double tmp_X_i1_N_1;
    for (int i1 = start_row; i1 < end_row; i1++) {
        CACHEs_RD(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
        tmp_X_i1_N_1 = tmp_X_i1_N_1 / B[i1 * n + (n - 1)];
        CACHEs_WT(X, &X[i1 * n + (n - 1)], tmp_X_i1_N_1);
    }
    CACHEs_FLUSH(X);
}

__global__ void adi_kernel3_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    CACHEs_INIT(A, double, A, 0, 12);
    CACHEs_INIT(B, double, B, 0, 12);
    CACHEs_INIT(X, double, X, 0, 14);
    double tmp_A;
    double tmp_B1;
    double tmp_X1, tmp_X2;
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            CACHEs_RD(A, &A[i1 * n + (n - i2 - 3)], tmp_A);
            CACHEs_RD(B, &B[i1 * n + (n - i2 - 3)], tmp_B1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
            CACHEs_RD(X, &X[i1 * n + (n - i2 - 3)], tmp_X2);
            tmp_X1 = (tmp_X1 - tmp_X2 * tmp_A) / tmp_B1;
            CACHEs_WT(X, &X[i1 * n + (n - i2 - 2)], tmp_X1);
        }
    }
    CACHEs_FLUSH(X);
    CACHEs_INVALID(A);
    CACHEs_INVALID(B);
}

__global__ void adi_kernel4_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    double *B_p = B;
    CACHEs_INIT(X, double, X, 0, 14);
    CACHEb_INIT(B, double, &B[(i1 - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[i1 * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_X_i1_1_i2, tmp_B_i1_i2_1, tmp_A_i1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEs_RD(X, &X[(i1 - 1) * n + i2], tmp_X_i1_1_i2);
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_i1_i2_1);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_i1_i2);
        X_p[i1 * n + i2] = X_p[i1 * n + i2] - tmp_X_i1_1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
        B_p[i1 * n + i2] = B_p[i1 * n + i2] - tmp_A_i1_i2 * tmp_A_i1_i2 / tmp_B_i1_i2_1;
    }
    CACHEs_INVALID(X);
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
}

__global__ void adi_kernel5_cache(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    CACHEb_INIT(B, double, &B[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_1_i2, tmp_X_N_1_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_1_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_1_i2);
        tmp_X_N_1_i2 = tmp_X_N_1_i2 / tmp_B_N_1_i2;
        CACHEb_WT(X, &X[i2 - start_col], tmp_X_N_1_i2);
    }
    CACHEb_INVALID(B);
    CACHEb_FLUSH(X);
}

__global__ void adi_kernel6_cache(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);
    double *X_p = X;
    CACHEb_INIT(B, double, &B[(n - 2 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(A, double, &A[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    CACHEb_INIT(X, double, &X[(n - 3 - i1) * n + start_col], 0, (end_col - start_col) * sizeof(double));
    double tmp_B_N_2_i2, tmp_X_N_3_i2, tmp_A_N_3_i2;
    for (int i2 = start_col; i2 < end_col; i2++) {
        CACHEb_RD(B, &B[i2 - start_col], tmp_B_N_2_i2);
        CACHEb_RD(A, &A[i2 - start_col], tmp_A_N_3_i2);
        CACHEb_RD(X, &X[i2 - start_col], tmp_X_N_3_i2);
        X_p[(n - 2 - i1) * n + i2] = (X_p[(n - 2 - i1) * n + i2] - tmp_X_N_3_i2 * tmp_A_N_3_i2) / tmp_B_N_2_i2;
    }
    CACHEb_INVALID(B);
    CACHEb_INVALID(A);
    CACHEb_INVALID(X);
}

#endif
#define SIMD_LEN  16
#define VEC_BYTES 128
// 未向量化，数据依赖
__global__ void adi_kernel1_vec(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            X[i1 * n + i2] = X[i1 * n + i2] - X[i1 * n + (i2 - 1)] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
            B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
        }
    }
}
// 未向量化
__global__ void adi_kernel2_vec(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    for (int i1 = start_row; i1 < end_row; i1++) {
        X[i1 * n + (n - 1)] = X[i1 * n + (n - 1)] / B[i1 * n + (n - 1)];
    }
}
// 未向量化，数据依赖
__global__ void adi_kernel3_vec(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 0; i2 < n - 2; i2++) {
            X[i1 * n + (n - i2 - 2)] =
                (X[i1 * n + (n - i2 - 2)] - X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) /
                B[i1 * n + (n - i2 - 3)];
        }
    }
}

/* ----------------------------------------------------------------- */
/*  kernel-4 : 垂直前向消元                                           */
/* ----------------------------------------------------------------- */

__global__ void adi_kernel4_vec(int n, int i1,
                                double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);

    // 计算当前线程处理的列范围
    int total_cols = end_col - start_col;
    int vec_iterations = total_cols / SIMD_LEN;
    // int remainder_cols = total_cols % SIMD_LEN;

    // 基础索引偏移
    int base_idx_curr = i1 * n;              // 当前行 (i1) 的起始索引
    int base_idx_prev = (i1 - 1) * n;        // 前一行 (i1-1) 的起始索引

    // 向量化处理主要部分
    for (int v = 0; v < vec_iterations; ++v) {
        int col_offset = start_col + v * SIMD_LEN;

        // --- 加载所有需要的数据 ---
        lvector double a_curr_vec, b_prev_vec, x_prev_vec, b_curr_vec, x_curr_vec;
        vector_load(&A[base_idx_curr + col_offset], &a_curr_vec, VEC_BYTES);
        vector_load(&B[base_idx_prev + col_offset], &b_prev_vec, VEC_BYTES);
        vector_load(&X[base_idx_prev + col_offset], &x_prev_vec, VEC_BYTES);
        vector_load(&B[base_idx_curr + col_offset], &b_curr_vec, VEC_BYTES);
        vector_load(&X[base_idx_curr + col_offset], &x_curr_vec, VEC_BYTES);

        // --- 计算 X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2] ---

        // 1. 计算 X[i1-1][i2] * A[i1][i2]
        lvector double xA_part = vec_muli(x_prev_vec, a_curr_vec);

        // 2. 计算 (X * A) / B_prev (使用直接向量除法)
        lvector double xA_div_B_prev = vm_fdivd16(xA_part, b_prev_vec);

        // 3. 计算 X_curr = X_curr - (X * A / B_prev)
        // 使用乘加实现减法: res = c - a * b  =>  res = c + (-1.0) * (a * b)
        // 这里 a*b = xA_div_B_prev, c = x_curr_vec
        lvector double neg_one_vec = (lvector double)vec_svbcast(-1.0);
        lvector double new_x_curr = vec_mula(neg_one_vec, xA_div_B_prev, x_curr_vec);


        // --- 计算 B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2] ---

        // 1. 计算 A[i1][i2] * A[i1][i2]
        lvector double AA_part = vec_muli(a_curr_vec, a_curr_vec);

        // 2. 计算 (A * A) / B_prev (使用直接向量除法)
        lvector double AA_div_B_prev = vm_fdivd16(AA_part, b_prev_vec);

        // 3. 计算 B_curr = B_curr - (A * A / B_prev)
        // 使用乘加实现减法: res = c - a * b  =>  res = c + (-1.0) * (a * b)
        // 这里 a*b = AA_div_B_prev, c = b_curr_vec
        lvector double new_b_curr = vec_mula(neg_one_vec, AA_div_B_prev, b_curr_vec);

        // --- 存储结果 ---
        vector_store(&new_x_curr, &X[base_idx_curr + col_offset], VEC_BYTES);
        vector_store(&new_b_curr, &B[base_idx_curr + col_offset], VEC_BYTES);
    }

    // --- 处理尾部标量部分 ---
    int tail_start_col = start_col + vec_iterations * SIMD_LEN;
    for (int i2 = tail_start_col; i2 < end_col; i2++) {
        // 严格按照原始顺序计算，确保精度
        double x_part = X[(i1 - 1) * n + i2] * A[i1 * n + i2];
        double x_term = x_part / B[(i1 - 1) * n + i2];
        X[i1 * n + i2] = X[i1 * n + i2] - x_term;

        double a_part = A[i1 * n + i2] * A[i1 * n + i2];
        double a_term = a_part / B[(i1 - 1) * n + i2];
        B[i1 * n + i2] = B[i1 * n + i2] - a_term;
    }
}

__global__ void adi_kernel5_vec(int n, double *A, double *B, double *X)
{
    int tid   = get_thread_id();
    int gsize = get_group_size();

    int base = n / gsize, extra = n % gsize;
    int st   = (tid < extra) ?
               tid * (base + 1) :
               extra * (base + 1) + (tid - extra) * base;
    int ed   = st + ((tid < extra) ? (base + 1) : base);
    if (ed <= st) return;

    lvector double *bufX = (lvector double *)vector_malloc(sizeof(lvector double) * 2);
    lvector double *bufB = bufX + 1;

    size_t off_row = (size_t)(n - 1) * n;

    for (int j = st; j < ed; )
    {
        int remain = ed - j;
        if (remain >= SIMD_LEN)
        {
            vector_load(X + off_row + j, bufX, VEC_BYTES);
            vector_load(B + off_row + j, bufB, VEC_BYTES);

            *bufX = vm_fdivd16(*bufX, *bufB);

            vector_store(bufX, X + off_row + j, VEC_BYTES);
            j += SIMD_LEN;
        }
        else            /* 尾部 ≤15 */
        {
            for ( ; j < ed; ++j)
                X[off_row + j] /= B[off_row + j];
        }
    }
    vector_free(bufX);
}

__global__ void adi_kernel6_vec(int n, int i1,
                            double *A, double *B, double *X)
{
    lvector double vneg1 = (lvector double)vec_svbcast(-1.0);
    int tid   = get_thread_id();
    int gsize = get_group_size();

    int base = n / gsize, extra = n % gsize;
    int j_st = (tid < extra) ?
               tid * (base + 1) :
               extra * (base + 1) + (tid - extra) * base;
    int j_ed = j_st + ((tid < extra) ? (base + 1) : base);
    if (j_ed <= j_st) return;

    int row_cur  = n - 2 - i1;      /* 待更新行 */
    int row_prev = n - 3 - i1;      /* 上一行   */

    /* 五条向量缓冲 ------------------------------------------------ */
    lvector double *buf = (lvector double *)
                          vector_malloc(sizeof(lvector double) * 5);
    lvector double *Xcur = buf + 0;
    lvector double *Bcur = buf + 1;
    lvector double *Xpre = buf + 2;
    lvector double *Apre = buf + 3;
    lvector double *tmp  = buf + 4;

    for (int j = j_st; j < j_ed; )
    {
        int remain = j_ed - j;
        if (remain >= SIMD_LEN)                                   /* SIMD */
        {
            size_t off_cur  = (size_t)row_cur  * n + j;
            size_t off_prev = (size_t)row_prev * n + j;

            vector_load(X + off_cur , Xcur, VEC_BYTES);
            vector_load(B + off_cur , Bcur, VEC_BYTES);

            vector_load(X + off_prev, Xpre, VEC_BYTES);
            vector_load(A + off_prev, Apre, VEC_BYTES);

            /* num = Xcur – Xpre * Apre */
            *tmp = vec_mulb(*Xpre, *Apre, *Xcur);   /* Xpre*Apre - Xcur = -num */
            *tmp = vec_muli(*tmp, vneg1);           /* num                  */

            /* Xnew = num / Bcur */
            *tmp = vm_fdivd16(*tmp, *Bcur);
            vector_store(tmp, X + off_cur, VEC_BYTES);

            j += SIMD_LEN;
        }
        else                                                      /* 尾标量 */
        {
            for ( ; j < j_ed; ++j)
            {
                double num =  X[row_cur  * n + j] -
                              X[row_prev * n + j] * A[row_prev * n + j];
                X[row_cur * n + j] = num / B[row_cur * n + j];
            }
        }
    }
    vector_free(buf);
}
#include "../ADI456/kernel_vec.h"
#include "../ADI456/kernel_cache_llm.h"//SM缓存优化文件