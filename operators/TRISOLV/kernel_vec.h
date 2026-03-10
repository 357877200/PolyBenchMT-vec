
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void trisolv_kernel_vec_qwen(int n, double *L, double *x, double *b)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    if (thread_id != 0) {
        return;
    }

    lvector double *buf_l = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_l || !buf_x || !buf_b) {
        if (buf_l) vector_free(buf_l);
        if (buf_x) vector_free(buf_x);
        if (buf_b) vector_free(buf_b);
        return;
    }

    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int i = 0; i < n; ++i) {
        x[i] = b[i];
        double x_i = b[i];
        int j = 0;

        for (; j + SIMD_LEN <= i; j += SIMD_LEN) {
            vector_load(&L[i * n + j], buf_l, VEC_BYTES);
            vector_load(&x[j], buf_x, VEC_BYTES);

            lvector double vl = vec_ld(0, buf_l);
            lvector double vx = vec_ld(0, buf_x);

            lvector double vsub = vec_mulb(one_vec, vec_muli(vl, vx), (lvector double)vec_svbcast(0.0));
            double sum_val = sum_f64(vsub);

            x_i -= sum_val;
        }

        for (; j < i; ++j) {
            x_i -= L[i * n + j] * x[j];
        }

        x[i] = x_i / L[i * n + i];
    }

    vector_free(buf_l);
    vector_free(buf_x);
    vector_free(buf_b);
}