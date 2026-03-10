#include <stdint.h>
#include <compiler/m3000.h>
#include <compiler/vsip.h>
#include "hthread_device.h"
#include "vector_math.h"
#include "../common/cache_strategy/cache_wrapper.h"
// 自己编写的性能事件接口
#include "../common/prof_event.h"
#include "../common/compute_tool.h"
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
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void symm_kernel1_vec(int m, int n, double alpha,
                                 double *A, double *B, double *C, double *temp2)
{
    int tid         = get_thread_id();
    int num_threads = get_group_size();
    int cols_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * cols_per_thread;
    int end_j   = min(start_j + cols_per_thread, n);

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);

    lvector double *buf_bi = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_bk = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_ck = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_tmp2 = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_bi || !buf_bk || !buf_ck || !buf_tmp2) { /* free & return */ }

    for (int i = 0; i < m; i++) {
        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end = min(j + SIMD_LEN, end_j);
            int vec_size_bytes = (vec_end - j) * sizeof(double);

            lvector double temp2_vec = (lvector double)vec_svbcast(0.0);

            vector_load(&B[i * n + j], buf_bi, vec_size_bytes);
            lvector double vb_i = vec_ld(0, buf_bi);

            for (int k = 0; k < i; k++) {
                double aik = A[i * m + k];
                lvector double aik_vec = (lvector double)vec_svbcast(aik);

                // load C[k,j] and B[k,j]
                vector_load(&C[k * n + j], buf_ck, vec_size_bytes);
                vector_load(&B[k * n + j], buf_bk, vec_size_bytes);
                lvector double vc_k = vec_ld(0, buf_ck);
                lvector double vb_k = vec_ld(0, buf_bk);

                // C[k,j] += alpha * B[i,j] * aik
                vc_k = vec_mula(vc_k, (lvector double)vec_svbcast(1.0),
                                vec_muli(alpha_vec, vec_muli(vb_i, aik_vec)));

                vec_st(vc_k, 0, buf_ck);
                vector_store(buf_ck, &C[k * n + j], vec_size_bytes);

                // temp2_vec += B[k,j] * aik
                temp2_vec = vec_mula(temp2_vec, (lvector double)vec_svbcast(1.0),
                                     vec_muli(vb_k, aik_vec));
            }

            // 存储 temp2_vec
            vec_st(temp2_vec, 0, buf_tmp2);
            vector_store(buf_tmp2, &temp2[i * n + j], vec_size_bytes);
        }
    }

    vector_free(buf_bi);
    vector_free(buf_bk);
    vector_free(buf_ck);
    vector_free(buf_tmp2);
}
__global__ void symm_kernel2_vec(int m, int n, double alpha, double beta,
    double *A, double *B, double *C, double *temp2)
{
int tid         = get_thread_id();
int num_threads = get_group_size();
int cols_per_thread = (n + num_threads - 1) / num_threads;
int start_j = tid * cols_per_thread;
int end_j   = min(start_j + cols_per_thread, n);

lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
lvector double beta_vec  = (lvector double)vec_svbcast(beta);

lvector double *buf_bi   = (lvector double *)vector_malloc(VEC_BYTES);
lvector double *buf_ci   = (lvector double *)vector_malloc(VEC_BYTES);
lvector double *buf_tmp2 = (lvector double *)vector_malloc(VEC_BYTES);
if (!buf_bi || !buf_ci || !buf_tmp2) { /* free & return */ }

for (int i = 0; i < m; i++) {
double aii = A[i * m + i];
lvector double aii_vec = (lvector double)vec_svbcast(aii);

for (int j = start_j; j < end_j; j += SIMD_LEN) {
int vec_end = min(j + SIMD_LEN, end_j);
int vec_size_bytes = (vec_end - j) * sizeof(double);

vector_load(&B[i * n + j], buf_bi, vec_size_bytes);
lvector double vb_i = vec_ld(0, buf_bi);

vector_load(&C[i * n + j], buf_ci, vec_size_bytes);
lvector double vc_i = vec_ld(0, buf_ci);

vector_load(&temp2[i * n + j], buf_tmp2, vec_size_bytes);
lvector double tmp2_vec = vec_ld(0, buf_tmp2);

// beta * C[i,j]
vc_i = vec_muli(beta_vec, vc_i);

// + alpha * B[i,j] * aii
vc_i = vec_mula(vc_i, (lvector double)vec_svbcast(1.0),
vec_muli(alpha_vec, vec_muli(vb_i, aii_vec)));

// + alpha * temp2
vc_i = vec_mula(vc_i, (lvector double)vec_svbcast(1.0),
vec_muli(alpha_vec, tmp2_vec));

vec_st(vc_i, 0, buf_ci);
vector_store(buf_ci, &C[i * n + j], vec_size_bytes);
}
}

vector_free(buf_bi);
vector_free(buf_ci);
vector_free(buf_tmp2);
}
#include "../SYMM/kernel_vec.h"//大模型生成的存储文件
#include "../SYMM/kernel_cache_llm.h"//SM缓存优化文件