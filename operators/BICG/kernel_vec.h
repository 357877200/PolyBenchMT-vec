
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void bicg_kernel1_vec_qwen(int nx, int ny, double *A, double *r, double *s)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int items_per_thread = ny / num_threads;
    int remainder = ny % num_threads;

    int start_j = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_j = start_j + items_per_thread + (thread_id < remainder ? 1 : 0);

    for (int j = start_j; j < end_j; j++) {
        s[j] = 0.0;
    }

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_s = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_s) {
        if (buf_a) vector_free(buf_a);
        if (buf_s) vector_free(buf_s);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = 0; i < nx; i++) {
        double r_val = r[i];
        lvector double r_vec = (lvector double)vec_svbcast(r_val);

        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = j + SIMD_LEN <= end_j ? j + SIMD_LEN : end_j;

            if (j + SIMD_LEN <= end_j) {
                vector_load(&s[j], buf_s, VEC_BYTES);
                lvector double vs = vec_ld(0, buf_s);

                vector_load(&A[i * ny + j], buf_a, VEC_BYTES);
                lvector double va = vec_ld(0, buf_a);

                lvector double vprod = vec_muli(r_vec, va);
                lvector double vnew_s = vec_mula(vs, zero_vec, vprod);

                vec_st(vnew_s, 0, buf_s);
                vector_store(buf_s, &s[j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    s[jj] += r_val * A[i * ny + jj];
                }
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_s);
}

__global__ void bicg_kernel2_vec_qwen(int nx, int ny, double *A, double *p, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int items_per_thread = nx / num_threads;
    int remainder = nx % num_threads;

    int start_i = thread_id * items_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_i = start_i + items_per_thread + (thread_id < remainder ? 1 : 0);

    for (int i = start_i; i < end_i; i++) {
        q[i] = 0.0;
    }

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_p = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_p) {
        if (buf_a) vector_free(buf_a);
        if (buf_p) vector_free(buf_p);
        return;
    }

    for (int i = start_i; i < end_i; i++) {
        double q_val = 0.0;

        for (int j = 0; j < ny; j += SIMD_LEN) {
            int vec_end_j = j + SIMD_LEN <= ny ? j + SIMD_LEN : ny;

            if (j + SIMD_LEN <= ny) {
                vector_load(&A[i * ny + j], buf_a, VEC_BYTES);
                lvector double va = vec_ld(0, buf_a);

                vector_load(&p[j], buf_p, VEC_BYTES);
                lvector double vp = vec_ld(0, buf_p);

                lvector double vprod = vec_muli(va, vp);
                double sum_val = sum_f64(vprod);
                q_val += sum_val;
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    q_val += A[i * ny + jj] * p[jj];
                }
            }
        }

        q[i] = q_val;
    }

    vector_free(buf_a);
    vector_free(buf_p);
}