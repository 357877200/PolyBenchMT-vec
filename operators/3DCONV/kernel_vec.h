
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void convolution3D_kernel_vec_qwen(int ni, int nj, int nk, int i, double *A, double *B) {
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    int total_elements = (nj - 2) * (nk - 2);
    int elements_per_thread = total_elements / total_threads;
    int extra_elements = total_elements % total_threads;
    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    lvector double *buf_A = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_A || !buf_B) {
        if (buf_A) vector_free(buf_A);
        if (buf_B) vector_free(buf_B);
        return;
    }

    lvector double c11_vec = (lvector double)vec_svbcast(2.0);
    lvector double c12_vec = (lvector double)vec_svbcast(-3.0);
    lvector double c13_vec = (lvector double)vec_svbcast(4.0);
    lvector double c21_vec = (lvector double)vec_svbcast(5.0);
    lvector double c22_vec = (lvector double)vec_svbcast(6.0);
    lvector double c23_vec = (lvector double)vec_svbcast(7.0);
    lvector double c31_vec = (lvector double)vec_svbcast(-8.0);
    lvector double c32_vec = (lvector double)vec_svbcast(-9.0);
    lvector double c33_vec = (lvector double)vec_svbcast(10.0);

    for (int idx = start_idx; idx < end_idx; idx++) {
        int j = idx / (nk - 2) + 1;
        int k = idx % (nk - 2) + 1;
        int base_idx_A = i * (nk * nj);
        int base_idx_B = base_idx_A + j * nk + k;

        if (k + SIMD_LEN - 1 < nk - 1) {
            lvector double v_res = (lvector double)vec_svbcast(0.0);
            
            vector_load(&A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)], buf_A, VEC_BYTES);
            lvector double v_a11 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c11_vec, vec_muli(c11_vec, v_a11));
            
            vector_load(&A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)], buf_A, VEC_BYTES);
            lvector double v_a13 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c13_vec, vec_muli(c13_vec, v_a13));
            
            vector_load(&A[(i - 1) * (nk * nj) + j * nk + (k - 1)], buf_A, VEC_BYTES);
            lvector double v_a21 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c21_vec, vec_muli(c21_vec, v_a21));
            
            vector_load(&A[(i + 1) * (nk * nj) + j * nk + (k - 1)], buf_A, VEC_BYTES);
            lvector double v_a23 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c23_vec, vec_muli(c23_vec, v_a23));
            
            vector_load(&A[(i - 1) * (nk * nj) + (j + 1) * nk + (k - 1)], buf_A, VEC_BYTES);
            lvector double v_a31 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c31_vec, vec_muli(c31_vec, v_a31));
            
            vector_load(&A[(i + 1) * (nk * nj) + (j + 1) * nk + (k - 1)], buf_A, VEC_BYTES);
            lvector double v_a33 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c33_vec, vec_muli(c33_vec, v_a33));
            
            vector_load(&A[i * (nk * nj) + (j - 1) * nk + k], buf_A, VEC_BYTES);
            lvector double v_a12 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c12_vec, vec_muli(c12_vec, v_a12));
            
            vector_load(&A[i * (nk * nj) + j * nk + k], buf_A, VEC_BYTES);
            lvector double v_a22 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c22_vec, vec_muli(c22_vec, v_a22));
            
            vector_load(&A[i * (nk * nj) + (j + 1) * nk + k], buf_A, VEC_BYTES);
            lvector double v_a32 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c32_vec, vec_muli(c32_vec, v_a32));
            
            vector_load(&A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)], buf_A, VEC_BYTES);
            lvector double v_a111 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c11_vec, vec_muli(c11_vec, v_a111));
            
            vector_load(&A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)], buf_A, VEC_BYTES);
            lvector double v_a113 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c13_vec, vec_muli(c13_vec, v_a113));
            
            vector_load(&A[(i - 1) * (nk * nj) + j * nk + (k + 1)], buf_A, VEC_BYTES);
            lvector double v_a121 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c21_vec, vec_muli(c21_vec, v_a121));
            
            vector_load(&A[(i + 1) * (nk * nj) + j * nk + (k + 1)], buf_A, VEC_BYTES);
            lvector double v_a123 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c23_vec, vec_muli(c23_vec, v_a123));
            
            vector_load(&A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)], buf_A, VEC_BYTES);
            lvector double v_a131 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c31_vec, vec_muli(c31_vec, v_a131));
            
            vector_load(&A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)], buf_A, VEC_BYTES);
            lvector double v_a133 = vec_ld(0, buf_A);
            v_res = vec_mula(v_res, c33_vec, vec_muli(c33_vec, v_a133));
            
            vec_st(v_res, 0, buf_B);
            vector_store(buf_B, &B[base_idx_B], VEC_BYTES);
        } else {
            double c11 = 2, c12 = -3, c13 = 4, c21 = 5, c22 = 6, c23 = 7, c31 = -8, c32 = -9, c33 = 10;
            for (int kk = k; kk < nk - 1; kk++) {
                B[i * (nk * nj) + j * nk + kk] = 
                    c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (kk - 1)] + 
                    c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (kk - 1)] +
                    c21 * A[(i - 1) * (nk * nj) + j * nk + (kk - 1)] + 
                    c23 * A[(i + 1) * (nk * nj) + j * nk + (kk - 1)] +
                    c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (kk - 1)] + 
                    c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (kk - 1)] +
                    c12 * A[i * (nk * nj) + (j - 1) * nk + kk] + 
                    c22 * A[i * (nk * nj) + j * nk + kk] +
                    c32 * A[i * (nk * nj) + (j + 1) * nk + kk] + 
                    c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (kk + 1)] +
                    c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (kk + 1)] + 
                    c21 * A[(i - 1) * (nk * nj) + j * nk + (kk + 1)] +
                    c23 * A[(i + 1) * (nk * nj) + j * nk + (kk + 1)] + 
                    c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (kk + 1)] +
                    c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (kk + 1)];
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
}