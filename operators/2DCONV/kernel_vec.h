
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void convolution2D_kernel_vec_qwen(int ni, int nj, double *A, double *B, uint64_t *before_hot_data, uint64_t *after_hot_data)
{
    int group_size = get_group_size();
    int thread_id = get_thread_id();

    double c11 = +0.2, c21 = +0.5, c31 = -0.8;
    double c12 = -0.3, c22 = +0.6, c32 = -0.9;
    double c13 = +0.4, c23 = +0.7, c33 = +0.10;

    const int total_tasks = (ni - 2) * (nj - 2);
    if (total_tasks <= 0)
        return;

    const int base_tasks = total_tasks / group_size;
    const int remainder = total_tasks % group_size;

    int start = (thread_id < remainder) ? thread_id * (base_tasks + 1)
                                        : remainder * (base_tasks + 1) + (thread_id - remainder) * base_tasks;
    int end = start + ((thread_id < remainder) ? (base_tasks + 1) : base_tasks);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        return;
    }

    lvector double c11_vec = (lvector double)vec_svbcast(c11);
    lvector double c21_vec = (lvector double)vec_svbcast(c21);
    lvector double c31_vec = (lvector double)vec_svbcast(c31);
    lvector double c12_vec = (lvector double)vec_svbcast(c12);
    lvector double c22_vec = (lvector double)vec_svbcast(c22);
    lvector double c32_vec = (lvector double)vec_svbcast(c32);
    lvector double c13_vec = (lvector double)vec_svbcast(c13);
    lvector double c23_vec = (lvector double)vec_svbcast(c23);
    lvector double c33_vec = (lvector double)vec_svbcast(c33);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int t = start; t < end; ++t) {
        const int i = 1 + t / (nj - 2);
        const int j = 1 + t % (nj - 2);

        int base_j = j;
        int vec_end_j = min(j + SIMD_LEN, nj - 1);

        if (j + SIMD_LEN <= nj - 1) {
            vector_load(&A[(i - 1) * nj + (j - 1)], buf_a, VEC_BYTES);
            lvector double a11 = vec_ld(0, buf_a);
            vector_load(&A[(i - 1) * nj + j], buf_a, VEC_BYTES);
            lvector double a21 = vec_ld(0, buf_a);
            vector_load(&A[(i - 1) * nj + (j + 1)], buf_a, VEC_BYTES);
            lvector double a31 = vec_ld(0, buf_a);
            
            lvector double val1 = vec_mula(vec_muli(c11_vec, a11), vec_muli(c21_vec, a21), vec_muli(c31_vec, a31));
            
            vector_load(&B[i * nj + j], buf_b, VEC_BYTES);
            lvector double b = vec_ld(0, buf_b);
            lvector double b_new = vec_mula(b, zero_vec, val1);
            vec_st(b_new, 0, buf_b);
            vector_store(buf_b, &B[i * nj + j], VEC_BYTES);
        } else {
            for (int jj = j; jj < nj - 1; ++jj) {
                B[i * nj + jj] = c11 * A[(i - 1) * nj + (jj - 1)] + 
                                 c21 * A[(i - 1) * nj + jj] + 
                                 c31 * A[(i - 1) * nj + (jj + 1)];
            }
        }
    }

    for (int t = start; t < end; ++t) {
        const int i = 1 + t / (nj - 2);
        const int j = 1 + t % (nj - 2);

        if (j + SIMD_LEN <= nj - 1) {
            vector_load(&A[i * nj + (j - 1)], buf_a, VEC_BYTES);
            lvector double a12 = vec_ld(0, buf_a);
            vector_load(&A[i * nj + j], buf_a, VEC_BYTES);
            lvector double a22 = vec_ld(0, buf_a);
            vector_load(&A[i * nj + (j + 1)], buf_a, VEC_BYTES);
            lvector double a32 = vec_ld(0, buf_a);
            
            lvector double val2 = vec_mula(vec_muli(c12_vec, a12), vec_muli(c22_vec, a22), vec_muli(c32_vec, a32));
            
            vector_load(&B[i * nj + j], buf_b, VEC_BYTES);
            lvector double b = vec_ld(0, buf_b);
            lvector double b_new = vec_mula(b, zero_vec, val2);
            vec_st(b_new, 0, buf_b);
            vector_store(buf_b, &B[i * nj + j], VEC_BYTES);
        } else {
            for (int jj = j; jj < nj - 1; ++jj) {
                B[i * nj + jj] += c12 * A[i * nj + (jj - 1)] + 
                                  c22 * A[i * nj + jj] + 
                                  c32 * A[i * nj + (jj + 1)];
            }
        }
    }

    for (int t = start; t < end; ++t) {
        const int i = 1 + t / (nj - 2);
        const int j = 1 + t % (nj - 2);

        if (j + SIMD_LEN <= nj - 1) {
            vector_load(&A[(i + 1) * nj + (j - 1)], buf_a, VEC_BYTES);
            lvector double a13 = vec_ld(0, buf_a);
            vector_load(&A[(i + 1) * nj + j], buf_a, VEC_BYTES);
            lvector double a23 = vec_ld(0, buf_a);
            vector_load(&A[(i + 1) * nj + (j + 1)], buf_a, VEC_BYTES);
            lvector double a33 = vec_ld(0, buf_a);
            
            lvector double val3 = vec_mula(vec_muli(c13_vec, a13), vec_muli(c23_vec, a23), vec_muli(c33_vec, a33));
            
            vector_load(&B[i * nj + j], buf_b, VEC_BYTES);
            lvector double b = vec_ld(0, buf_b);
            lvector double b_new = vec_mula(b, zero_vec, val3);
            vec_st(b_new, 0, buf_b);
            vector_store(buf_b, &B[i * nj + j], VEC_BYTES);
        } else {
            for (int jj = j; jj < nj - 1; ++jj) {
                B[i * nj + jj] += c13 * A[(i + 1) * nj + (jj - 1)] + 
                                  c23 * A[(i + 1) * nj + jj] + 
                                  c33 * A[(i + 1) * nj + (jj + 1)];
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
}