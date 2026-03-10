
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void symm_kernel1_vec_qwen(int m, int n, double alpha,
    double *A, double *B, double *C, double *temp2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int cols_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * cols_per_thread;
    int end_j = min(start_j + cols_per_thread, n);

    lvector double *buf_a_ik = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b_i = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b_k = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_t2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a_ik || !buf_b_i || !buf_b_k || !buf_c || !buf_t2 || !buf_res) {
        if (buf_a_ik) vector_free(buf_a_ik);
        if (buf_b_i) vector_free(buf_b_i);
        if (buf_b_k) vector_free(buf_b_k);
        if (buf_c) vector_free(buf_c);
        if (buf_t2) vector_free(buf_t2);
        if (buf_res) vector_free(buf_res);
        return;
    }

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = 0; i < m; i++) {
        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, end_j);

            if (j + SIMD_LEN <= end_j) {
                vector_load(&B[i * n + j], buf_b_i, VEC_BYTES);
                lvector double vb_i = vec_ld(0, buf_b_i);

                lvector double vt2 = zero_vec;
                for (int k = 0; k < i; k++) {
                    vector_load(&A[i * m + k], buf_a_ik, sizeof(double));
                    vector_load(&B[k * n + j], buf_b_k, VEC_BYTES);
                    vector_load(&C[k * n + j], buf_c, VEC_BYTES);

                    lvector double va_ik = vec_ld(0, buf_a_ik);
                    lvector double vb_k = vec_ld(0, buf_b_k);
                    lvector double vc = vec_ld(0, buf_c);

                    lvector double vtemp1 = vec_muli(vb_i, va_ik);
                    vtemp1 = vec_muli(vtemp1, alpha_vec);
                    lvector double vc_new = vec_mula(vc, vtemp1, zero_vec);
                    vec_st(vc_new, 0, buf_c);
                    vector_store(buf_c, &C[k * n + j], VEC_BYTES);

                    lvector double vtemp2 = vec_muli(vb_k, va_ik);
                    vt2 = vec_mula(vt2, vtemp2, zero_vec);
                }
                vec_st(vt2, 0, buf_t2);
                vector_store(buf_t2, &temp2[i * n + j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    double t2 = 0.0;
                    for (int k = 0; k < i; k++) {
                        C[k * n + jj] += alpha * B[i * n + jj] * A[i * m + k];
                        t2 += B[k * n + jj] * A[i * m + k];
                    }
                    temp2[i * n + jj] = t2;
                }
            }
        }
    }

    vector_free(buf_a_ik);
    vector_free(buf_b_i);
    vector_free(buf_b_k);
    vector_free(buf_c);
    vector_free(buf_t2);
    vector_free(buf_res);
}

__global__ void symm_kernel2_vec_qwen(int m, int n, double alpha, double beta,
    double *A, double *B, double *C, double *temp2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int cols_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * cols_per_thread;
    int end_j = min(start_j + cols_per_thread, n);

    lvector double *buf_a_ii = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b_i = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_t2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a_ii || !buf_b_i || !buf_c || !buf_t2 || !buf_res) {
        if (buf_a_ii) vector_free(buf_a_ii);
        if (buf_b_i) vector_free(buf_b_i);
        if (buf_c) vector_free(buf_c);
        if (buf_t2) vector_free(buf_t2);
        if (buf_res) vector_free(buf_res);
        return;
    }

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
    lvector double beta_vec = (lvector double)vec_svbcast(beta);

    for (int i = 0; i < m; i++) {
        vector_load(&A[i * m + i], buf_a_ii, sizeof(double));
        lvector double va_ii = vec_ld(0, buf_a_ii);
        
        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, end_j);

            if (j + SIMD_LEN <= end_j) {
                vector_load(&B[i * n + j], buf_b_i, VEC_BYTES);
                vector_load(&C[i * n + j], buf_c, VEC_BYTES);
                vector_load(&temp2[i * n + j], buf_t2, VEC_BYTES);

                lvector double vb_i = vec_ld(0, buf_b_i);
                lvector double vc = vec_ld(0, buf_c);
                lvector double vt2 = vec_ld(0, buf_t2);

                lvector double vterm1 = vec_muli(vc, beta_vec);
                lvector double vterm2 = vec_muli(vb_i, va_ii);
                vterm2 = vec_muli(vterm2, alpha_vec);
                lvector double vterm3 = vec_muli(vt2, alpha_vec);

                lvector double vres = vec_mula(vterm1, vterm2, vterm3);

                vec_st(vres, 0, buf_res);
                vector_store(buf_res, &C[i * n + j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    double t2 = temp2[i * n + jj];
                    C[i * n + jj] = beta * C[i * n + jj]
                                   + alpha * B[i * n + jj] * A[i * m + i]
                                   + alpha * t2;
                }
            }
        }
    }

    vector_free(buf_a_ii);
    vector_free(buf_b_i);
    vector_free(buf_c);
    vector_free(buf_t2);
    vector_free(buf_res);
}