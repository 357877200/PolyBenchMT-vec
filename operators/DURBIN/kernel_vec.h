
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void durbin_kernel1_vec_qwen(int k, int barrier_id, int n, double *r, double *y, double *z, double *alpha, double *beta)
{
    int tid = get_thread_id();
    if (tid != 0) return;

    *beta = (1.0 - (*alpha) * (*alpha)) * (*beta);

    double sum_total = 0.0;
    for (int i = 0; i < k; i++) {
        sum_total += r[k - i - 1] * y[i];
    }

    *alpha = -(r[k] + sum_total) / (*beta);

    double alpha_val = *alpha;
    for (int i = 0; i < k; i++) {
        z[i] = y[i] + alpha_val * y[k - i - 1];
    }
}

__global__ void durbin_kernel2_vec_qwen(int k, int barrier_id, double *y, double *z, double *alpha)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int per_thread = (k + num_threads - 1) / num_threads;
    int start_i = tid * per_thread;
    int end_i   = min(start_i + per_thread, k);

    for (int i = start_i; i < end_i; i++) {
        y[i] = z[i];
    }

    if (tid == 0) {
        y[k] = *alpha;
    }
}