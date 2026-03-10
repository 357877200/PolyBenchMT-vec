
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void gemver_kernel1_vec_qwen(int n, double alpha, double beta, double *a, double *v1, double *v2,
                                   double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);

    lvector double *buf_v1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_v2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_v1 || !buf_v2 || !buf_a) {
        if (buf_v1) vector_free(buf_v1);
        if (buf_v2) vector_free(buf_v2);
        if (buf_a) vector_free(buf_a);
        return;
    }
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int i = start_idx; i < end_idx; ++i) {
        double u1_val = u1[i];
        double u2_val = u2[i];
        lvector double u1_vec = (lvector double)vec_svbcast(u1_val);
        lvector double u2_vec = (lvector double)vec_svbcast(u2_val);

        for (int j = 0; j < n; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, n);

            if (j + SIMD_LEN <= n) {
                vector_load(&v1[j], buf_v1, VEC_BYTES);
                vector_load(&v2[j], buf_v2, VEC_BYTES);
                vector_load(&a[i * n + j], buf_a, VEC_BYTES);

                lvector double vv1 = vec_ld(0, buf_v1);
                lvector double vv2 = vec_ld(0, buf_v2);
                lvector double va = vec_ld(0, buf_a);

                lvector double term1 = vec_muli(u1_vec, vv1);
                lvector double term2 = vec_muli(u2_vec, vv2);
                lvector double vres = vec_mula(va, one_vec, vec_mula(term1, one_vec, term2));

                vec_st(vres, 0, buf_a);
                vector_store(buf_a, &a[i * n + j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    a[i * n + jj] += u1_val * v1[jj] + u2_val * v2[jj];
                }
            }
        }
    }
    vector_free(buf_v1);
    vector_free(buf_v2);
    vector_free(buf_a);
}

__global__ void gemver_kernel2_vec_qwen(int n, double alpha, double beta, double *a, double *x, double *y,
                                   double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_z = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_x || !buf_z) {
        if (buf_a) vector_free(buf_a);
        if (buf_x) vector_free(buf_x);
        if (buf_z) vector_free(buf_z);
        return;
    }
    lvector double beta_vec = (lvector double)vec_svbcast(beta);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int j = 0; j < n; j++) {
        double y_val = y[j];
        lvector double y_vec = (lvector double)vec_svbcast(y_val);

        for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
            int vec_end_i = min(i + SIMD_LEN, end_idx);

            if (i + SIMD_LEN <= end_idx) {
                vector_load(&a[j * n + i], buf_a, VEC_BYTES);
                vector_load(&x[i], buf_x, VEC_BYTES);

                lvector double va = vec_ld(0, buf_a);
                lvector double vx = vec_ld(0, buf_x);

                lvector double vres = vec_mula(vx, vec_muli(beta_vec, vec_muli(va, y_vec)), one_vec);

                vec_st(vres, 0, buf_x);
                vector_store(buf_x, &x[i], VEC_BYTES);
            } else {
                for (int ii = i; ii < vec_end_i; ++ii) {
                    x[ii] += beta * a[j * n + ii] * y_val;
                }
            }
        }
    }

    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end_i = min(i + SIMD_LEN, end_idx);

        if (i + SIMD_LEN <= end_idx) {
            vector_load(&x[i], buf_x, VEC_BYTES);
            vector_load(&z[i], buf_z, VEC_BYTES);

            lvector double vx = vec_ld(0, buf_x);
            lvector double vz = vec_ld(0, buf_z);

            lvector double vres = vec_mula(vx, one_vec, vz);

            vec_st(vres, 0, buf_x);
            vector_store(buf_x, &x[i], VEC_BYTES);
        } else {
            for (int ii = i; ii < vec_end_i; ++ii) {
                x[ii] += z[ii];
            }
        }
    }
    vector_free(buf_a);
    vector_free(buf_x);
    vector_free(buf_z);
}

__global__ void gemver_kernel3_vec_qwen(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_x) {
        if (buf_a) vector_free(buf_a);
        if (buf_x) vector_free(buf_x);
        return;
    }
    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);

    for (int i = start_idx; i < end_idx; ++i) {
        double w_scalar = 0.0;

        for (int j = 0; j < n; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, n);

            if (j + SIMD_LEN <= n) {
                vector_load(&a[i * n + j], buf_a, VEC_BYTES);
                vector_load(&x[j], buf_x, VEC_BYTES);

                lvector double va = vec_ld(0, buf_a);
                lvector double vx = vec_ld(0, buf_x);

                lvector double vprod = vec_muli(alpha_vec, vec_muli(va, vx));
                double sum_val = sum_f64(vprod);
                w_scalar += sum_val;
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    w_scalar += alpha * a[i * n + jj] * x[jj];
                }
            }
        }
        w[i] += w_scalar;
    }
    vector_free(buf_a);
    vector_free(buf_x);
}