
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void syr2k_kernel_vec_qwen(int ni, int nj, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (ni * ni + num_threads - 1) / num_threads;
    int start_idx = thread_id * work_per_thread;
    int end_idx = min(start_idx + work_per_thread, ni * ni);

    lvector double *buf_a   = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b   = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c   = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b || !buf_c) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_c) vector_free(buf_c);
        return;
    }

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
    lvector double beta_vec = (lvector double)vec_svbcast(beta);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / ni;
        int j = idx % ni;

        c[idx] *= beta;
        for (int k = 0; k < nj; k++) {
            c[idx] += alpha * a[i * nj + k] * b[j * nj + k] + alpha * b[i * nj + k] * a[j * nj + k];
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_c);
}