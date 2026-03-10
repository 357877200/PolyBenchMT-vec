
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void trmm_kernel_vec_qwen(int m, int n, double alpha, double *A, double *B)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int work_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * work_per_thread;
    int end_j = min(start_j + work_per_thread, n);

    lvector double *buf_b_k = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_k = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_b_k || !buf_res || !buf_a_k) {
        if (buf_b_k) vector_free(buf_b_k);
        if (buf_res) vector_free(buf_res);
        if (buf_a_k) vector_free(buf_a_k);
        return;
    }

    lvector double one_vec = (lvector double)vec_svbcast(1.0);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);
    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);

    for (int i = 0; i < m; i++) {
        for (int j = start_j; j < end_j; j++) {
            int vec_end_j = min(j + SIMD_LEN, end_j);

            if (j + SIMD_LEN <= end_j) {
                vector_load(&B[i * n + j], buf_res, VEC_BYTES);
                lvector double vb_i = vec_ld(0, buf_res);
                lvector double accum = zero_vec;

                for (int k = i + 1; k < m; k++) {
                    vector_load(&B[k * n + j], buf_b_k, VEC_BYTES);
                    lvector double vb_k = vec_ld(0, buf_b_k);

                    lvector double a_k_val = vec_svbcast(A[k * m + i]);
                    accum = vec_mula(accum, one_vec, vec_muli(a_k_val, vb_k));
                }
                vb_i = vec_mula(vb_i, one_vec, accum);
                vb_i = vec_muli(alpha_vec, vb_i);

                vec_st(vb_i, 0, buf_res);
                vector_store(buf_res, &B[i * n + j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    for (int k = i + 1; k < m; k++) {
                        B[i * n + jj] += A[k * m + i] * B[k * n + jj];
                    }
                    B[i * n + jj] = alpha * B[i * n + jj];
                }
            }
            j = vec_end_j - 1;
        }
    }

    vector_free(buf_b_k);
    vector_free(buf_res);
    vector_free(buf_a_k);
}