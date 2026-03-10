
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void gesummv_kernel_vec_qwen(int n, double alpha, double beta, double *A, double *B, double *tmp,
                               double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b || !buf_x) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_x) vector_free(buf_x);
        return;
    }

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
    lvector double beta_vec = (lvector double)vec_svbcast(beta);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = start_i; i < end_i; i++) {
        tmp[i] = 0;
        y[i] = 0;

        lvector double vtmp = zero_vec;
        lvector double vy = zero_vec;

        for (int j = 0; j < n; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, n);

            if (j + SIMD_LEN <= n) {
                vector_load(&A[i * n + j], buf_a, VEC_BYTES);
                vector_load(&B[i * n + j], buf_b, VEC_BYTES);
                vector_load(&x[j], buf_x, VEC_BYTES);

                lvector double va = vec_ld(0, buf_a);
                lvector double vb = vec_ld(0, buf_b);
                lvector double vx = vec_ld(0, buf_x);

                lvector double va_mul_x = vec_muli(va, vx);
                lvector double vb_mul_x = vec_muli(vb, vx);

                vtmp = vec_mula(vtmp, (lvector double)vec_svbcast(1.0), va_mul_x);
                vy = vec_mula(vy, (lvector double)vec_svbcast(1.0), vb_mul_x);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    tmp[i] += A[i * n + jj] * x[jj];
                    y[i] += B[i * n + jj] * x[jj];
                }
            }
        }

        double tmp_sum = sum_f64(vtmp);
        double y_sum = sum_f64(vy);
        tmp[i] += tmp_sum;
        y[i] += y_sum;

        y[i] = alpha * tmp[i] + beta * y[i];
    }

    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_x);
}