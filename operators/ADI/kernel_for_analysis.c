#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }

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