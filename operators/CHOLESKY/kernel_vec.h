
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void cholesky_kernel_vec_qwen(int n, int barrier_id, double *A)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b || !buf_c) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_c) vector_free(buf_c);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    int per_thread = (n + num_threads - 1) / num_threads;
    int start_i = thread_id * per_thread;
    int end_i = min(start_i + per_thread, n);

    for (int i = start_i; i < end_i; ++i) {
        for (int j = 0; j < i; ++j) {
            double sum = A[i*n + j];
            for (int k = 0; k < j; ++k) {
                sum -= A[i*n + k] * A[j*n + k];
            }
            A[i*n + j] = sum / A[j*n + j];
        }

        double sum = A[i*n + i];
        for (int k = 0; k < i; ++k) {
            sum -= A[i*n + k] * A[i*n + k];
        }
        A[i*n + i] = sqrt(sum);
    }

    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_c);
}