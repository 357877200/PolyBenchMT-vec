
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void heat3d_kernel1_vec_qwen(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = n * n * n;
    int start = thread_id * (total_elements / num_threads);
    int end   = (thread_id == num_threads - 1) ? total_elements
                                               : (thread_id + 1) * (total_elements / num_threads);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        return;
    }

    lvector double coeff_vec = (lvector double)vec_svbcast(0.125);
    lvector double two_vec = (lvector double)vec_svbcast(2.0);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end = min(idx + SIMD_LEN, end);

        if (idx + SIMD_LEN <= end) {
            for (int vec_idx = 0; vec_idx < SIMD_LEN; ++vec_idx) {
                int current_idx = idx + vec_idx;
                int i = current_idx / (n * n);
                int j = (current_idx / n) % n;
                int k = current_idx % n;

                if (i >= 1 && i < n-1 &&
                    j >= 1 && j < n-1 &&
                    k >= 1 && k < n-1) {
                    int ip = (i+1)*n*n + j*n + k;
                    int im = (i-1)*n*n + j*n + k;
                    int jp = i*n*n + (j+1)*n + k;
                    int jm = i*n*n + (j-1)*n + k;
                    int kp = i*n*n + j*n + (k+1);
                    int km = i*n*n + j*n + (k-1);

                    vector_load(&A[current_idx], buf_a, sizeof(double));
                    vector_load(&A[ip], buf_a + 1, sizeof(double));
                    vector_load(&A[im], buf_a + 2, sizeof(double));
                    vector_load(&A[jp], buf_a + 3, sizeof(double));
                    vector_load(&A[jm], buf_a + 4, sizeof(double));
                    vector_load(&A[kp], buf_a + 5, sizeof(double));
                    vector_load(&A[km], buf_a + 6, sizeof(double));

                    lvector double va = vec_ld(0, buf_a);
                    lvector double vip = vec_ld(0, buf_a + 1);
                    lvector double vim = vec_ld(0, buf_a + 2);
                    lvector double vjp = vec_ld(0, buf_a + 3);
                    lvector double vjm = vec_ld(0, buf_a + 4);
                    lvector double vkp = vec_ld(0, buf_a + 5);
                    lvector double vkm = vec_ld(0, buf_a + 6);

                    lvector double term_i = vec_mulb(vip, coeff_vec, vec_muli(two_vec, va));
                    term_i = vec_mula(term_i, coeff_vec, vim);
                    lvector double term_j = vec_mulb(vjp, coeff_vec, vec_muli(two_vec, va));
                    term_j = vec_mula(term_j, coeff_vec, vjm);
                    lvector double term_k = vec_mulb(vkp, coeff_vec, vec_muli(two_vec, va));
                    term_k = vec_mula(term_k, coeff_vec, vkm);

                    lvector double vres = vec_mula(term_i, one_vec, term_j);
                    vres = vec_mula(vres, one_vec, term_k);
                    vres = vec_mula(vres, one_vec, va);

                    vec_st(vres, 0, buf_b);
                    vector_store(buf_b, &B[current_idx], sizeof(double));
                }
            }
        } else {
            for (int current_idx = idx; current_idx < vec_end; ++current_idx) {
                int i = current_idx / (n * n);
                int j = (current_idx / n) % n;
                int k = current_idx % n;

                if (i >= 1 && i < n-1 &&
                    j >= 1 && j < n-1 &&
                    k >= 1 && k < n-1) {
                    int ip = (i+1)*n*n + j*n + k;
                    int im = (i-1)*n*n + j*n + k;
                    int jp = i*n*n + (j+1)*n + k;
                    int jm = i*n*n + (j-1)*n + k;
                    int kp = i*n*n + j*n + (k+1);
                    int km = i*n*n + j*n + (k-1);

                    B[current_idx] = 0.125 * (A[ip] - 2.0*A[current_idx] + A[im])
                                   + 0.125 * (A[jp] - 2.0*A[current_idx] + A[jm])
                                   + 0.125 * (A[kp] - 2.0*A[current_idx] + A[km])
                                   + A[current_idx];
                }
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
}

__global__ void heat3d_kernel2_vec_qwen(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = n * n * n;
    int start = thread_id * (total_elements / num_threads);
    int end   = (thread_id == num_threads - 1) ? total_elements
                                               : (thread_id + 1) * (total_elements / num_threads);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        return;
    }

    lvector double coeff_vec = (lvector double)vec_svbcast(0.125);
    lvector double two_vec = (lvector double)vec_svbcast(2.0);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end = min(idx + SIMD_LEN, end);

        if (idx + SIMD_LEN <= end) {
            for (int vec_idx = 0; vec_idx < SIMD_LEN; ++vec_idx) {
                int current_idx = idx + vec_idx;
                int i = current_idx / (n * n);
                int j = (current_idx / n) % n;
                int k = current_idx % n;

                if (i >= 1 && i < n-1 &&
                    j >= 1 && j < n-1 &&
                    k >= 1 && k < n-1) {
                    int ip = (i+1)*n*n + j*n + k;
                    int im = (i-1)*n*n + j*n + k;
                    int jp = i*n*n + (j+1)*n + k;
                    int jm = i*n*n + (j-1)*n + k;
                    int kp = i*n*n + j*n + (k+1);
                    int km = i*n*n + j*n + (k-1);

                    vector_load(&B[current_idx], buf_b, sizeof(double));
                    vector_load(&B[ip], buf_b + 1, sizeof(double));
                    vector_load(&B[im], buf_b + 2, sizeof(double));
                    vector_load(&B[jp], buf_b + 3, sizeof(double));
                    vector_load(&B[jm], buf_b + 4, sizeof(double));
                    vector_load(&B[kp], buf_b + 5, sizeof(double));
                    vector_load(&B[km], buf_b + 6, sizeof(double));

                    lvector double vb = vec_ld(0, buf_b);
                    lvector double vip = vec_ld(0, buf_b + 1);
                    lvector double vim = vec_ld(0, buf_b + 2);
                    lvector double vjp = vec_ld(0, buf_b + 3);
                    lvector double vjm = vec_ld(0, buf_b + 4);
                    lvector double vkp = vec_ld(0, buf_b + 5);
                    lvector double vkm = vec_ld(0, buf_b + 6);

                    lvector double term_i = vec_mulb(vip, coeff_vec, vec_muli(two_vec, vb));
                    term_i = vec_mula(term_i, coeff_vec, vim);
                    lvector double term_j = vec_mulb(vjp, coeff_vec, vec_muli(two_vec, vb));
                    term_j = vec_mula(term_j, coeff_vec, vjm);
                    lvector double term_k = vec_mulb(vkp, coeff_vec, vec_muli(two_vec, vb));
                    term_k = vec_mula(term_k, coeff_vec, vkm);

                    lvector double vres = vec_mula(term_i, one_vec, term_j);
                    vres = vec_mula(vres, one_vec, term_k);
                    vres = vec_mula(vres, one_vec, vb);

                    vec_st(vres, 0, buf_a);
                    vector_store(buf_a, &A[current_idx], sizeof(double));
                }
            }
        } else {
            for (int current_idx = idx; current_idx < vec_end; ++current_idx) {
                int i = current_idx / (n * n);
                int j = (current_idx / n) % n;
                int k = current_idx % n;

                if (i >= 1 && i < n-1 &&
                    j >= 1 && j < n-1 &&
                    k >= 1 && k < n-1) {
                    int ip = (i+1)*n*n + j*n + k;
                    int im = (i-1)*n*n + j*n + k;
                    int jp = i*n*n + (j+1)*n + k;
                    int jm = i*n*n + (j-1)*n + k;
                    int kp = i*n*n + j*n + (k+1);
                    int km = i*n*n + j*n + (k-1);

                    A[current_idx] = 0.125 * (B[ip] - 2.0*B[current_idx] + B[im])
                                   + 0.125 * (B[jp] - 2.0*B[current_idx] + B[jm])
                                   + 0.125 * (B[kp] - 2.0*B[current_idx] + B[km])
                                   + B[current_idx];
                }
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
}