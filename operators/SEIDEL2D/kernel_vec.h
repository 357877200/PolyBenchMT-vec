
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void seidel2d_kernel_vec_qwen(int tsteps, int n, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = (n - 2) * (n - 2);
    if (total_elements <= 0) return;

    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1)
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx   = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    if (start_idx >= end_idx) return;

    lvector double *buf_a_c1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_c2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_c3 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_0 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_p1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_p2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_p3 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_sum = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_a_c1 || !buf_a_c2 || !buf_a_c3 || !buf_a_0 || !buf_a_p1 || !buf_a_p2 || !buf_a_p3 || !buf_sum) {
        if (buf_a_c1) vector_free(buf_a_c1);
        if (buf_a_c2) vector_free(buf_a_c2);
        if (buf_a_c3) vector_free(buf_a_c3);
        if (buf_a_0) vector_free(buf_a_0);
        if (buf_a_p1) vector_free(buf_a_p1);
        if (buf_a_p2) vector_free(buf_a_p2);
        if (buf_a_p3) vector_free(buf_a_p3);
        if (buf_sum) vector_free(buf_sum);
        return;
    }

    lvector double ninth_vec = (lvector double)vec_svbcast(1.0 / 9.0);

    for (int t = 0; t < tsteps; t++) {
        int processed_elements = 0;
        while (processed_elements < end_idx - start_idx) {
            int idx = start_idx + processed_elements;
            int i = idx / (n - 2) + 1;
            int j_start = idx % (n - 2) + 1;

            int j_vec_len = 0;
            for (int j_offset = 0; j_offset < SIMD_LEN; ++j_offset) {
                int current_idx = idx + j_offset;
                if (current_idx >= end_idx) break;
                int i_current = current_idx / (n - 2) + 1;
                int j_current = current_idx % (n - 2) + 1;
                if (i_current != i) break;
                j_vec_len++;
            }

            if (j_vec_len == SIMD_LEN) {
                vector_load(&A[(i-1)*n + (j_start-1)], buf_a_c1, VEC_BYTES);
                vector_load(&A[(i-1)*n + j_start], buf_a_c2, VEC_BYTES);
                vector_load(&A[(i-1)*n + (j_start+1)], buf_a_c3, VEC_BYTES);
                vector_load(&A[i*n + (j_start-1)], buf_a_0, VEC_BYTES);
                vector_load(&A[i*n + (j_start+1)], buf_a_p1, VEC_BYTES);
                vector_load(&A[(i+1)*n + (j_start-1)], buf_a_p2, VEC_BYTES);
                vector_load(&A[(i+1)*n + j_start], buf_a_p3, VEC_BYTES);

                lvector double v_c1 = vec_ld(0, buf_a_c1);
                lvector double v_c2 = vec_ld(0, buf_a_c2);
                lvector double v_c3 = vec_ld(0, buf_a_c3);
                lvector double v_0 = vec_ld(0, buf_a_0);
                lvector double v_p1 = vec_ld(0, buf_a_p1);
                lvector double v_p2 = vec_ld(0, buf_a_p2);
                lvector double v_p3 = vec_ld(0, buf_a_p3);

                lvector double v_sum_row1 = vec_mula(v_c1, (lvector double)vec_svbcast(1.0), vec_mula(v_c2, (lvector double)vec_svbcast(1.0), v_c3));
                lvector double v_sum_row2 = vec_mula(v_0, (lvector double)vec_svbcast(1.0), v_p1);
                lvector double v_sum_row3 = vec_mula(v_p2, (lvector double)vec_svbcast(1.0), v_p3);

                lvector double v_sum_all = vec_mula(v_sum_row1, (lvector double)vec_svbcast(1.0), vec_mula(v_sum_row2, (lvector double)vec_svbcast(1.0), v_sum_row3));
                
                lvector double v_res = vec_muli(v_sum_all, ninth_vec);
                
                vec_st(v_res, 0, buf_sum);
                vector_store(buf_sum, &A[i*n + j_start], VEC_BYTES);
                processed_elements += SIMD_LEN;
            } else {
                for (int jj = 0; jj < j_vec_len; ++jj) {
                    int current_idx = idx + jj;
                    int i_current = current_idx / (n - 2) + 1;
                    int j_current = current_idx % (n - 2) + 1;
                    A[i_current * n + j_current] =
                        (A[(i_current - 1) * n + (j_current - 1)] + A[(i_current - 1) * n + j_current] + A[(i_current - 1) * n + (j_current + 1)]
                       + A[i_current * n + (j_current - 1)] + A[i_current * n + j_current] + A[i_current * n + (j_current + 1)]
                       + A[(i_current + 1) * n + (j_current - 1)] + A[(i_current + 1) * n + j_current] + A[(i_current + 1) * n + (j_current + 1)]) / 9.0;
                }
                processed_elements += j_vec_len;
            }
        }
    }

    vector_free(buf_a_c1);
    vector_free(buf_a_c2);
    vector_free(buf_a_c3);
    vector_free(buf_a_0);
    vector_free(buf_a_p1);
    vector_free(buf_a_p2);
    vector_free(buf_a_p3);
    vector_free(buf_sum);
}