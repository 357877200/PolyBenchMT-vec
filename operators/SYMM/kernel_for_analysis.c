#include <stdint.h>
#include <math.h>
#define __global__
static inline int get_thread_id(void){ return 0; }
static inline int get_group_size(void){ return 1; }
static inline int min(int a, int b) { return a > b ? b : a; }
static inline int max(int a, int b) { return a > b ? a : b; }
__global__ void symm_kernel1(int m, int n, double alpha,
    double *A, double *B, double *C, double *temp2)
{
int tid         = get_thread_id();
int num_threads = get_group_size();
int cols_per_thread = (n + num_threads - 1) / num_threads;
int start_j = tid * cols_per_thread;
int end_j   = min(start_j + cols_per_thread, n);

for (int i = 0; i < m; i++) {
for (int j = start_j; j < end_j; j++) {
double t2 = 0.0;
for (int k = 0; k < i; k++) {
C[k * n + j] += alpha * B[i * n + j] * A[i * m + k];
t2 += B[k * n + j] * A[i * m + k];
}
temp2[i * n + j] = t2; // 保存临时结果
}
}
}

__global__ void symm_kernel2(int m, int n, double alpha, double beta,
    double *A, double *B, double *C, double *temp2)
{
int tid         = get_thread_id();
int num_threads = get_group_size();
int cols_per_thread = (n + num_threads - 1) / num_threads;
int start_j = tid * cols_per_thread;
int end_j   = min(start_j + cols_per_thread, n);

for (int i = 0; i < m; i++) {
for (int j = start_j; j < end_j; j++) {
double t2 = temp2[i * n + j];
C[i * n + j] = beta * C[i * n + j]
+ alpha * B[i * n + j] * A[i * m + i]
+ alpha * t2;
}
}
}