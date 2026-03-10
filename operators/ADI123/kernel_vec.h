
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void adi_kernel1_vec_qwen(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b || !buf_x) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_x) vector_free(buf_x);
        return;
    }

    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2 += SIMD_LEN) {
            if (i2 + SIMD_LEN <= n) {
                vector_load(&A[i1 * n + i2], buf_a, VEC_BYTES);
                vector_load(&B[i1 * n + i2], buf_b, VEC_BYTES);
                vector_load(&X[i1 * n + i2], buf_x, VEC_BYTES);
                
                vector_load(&A[i1 * n + (i2 - 1)], buf_a, VEC_BYTES);
                vector_load(&B[i1 * n + (i2 - 1)], buf_b, VEC_BYTES);
                vector_load(&X[i1 * n + (i2 - 1)], buf_x, VEC_BYTES);

                lvector double va = vec_ld(0, buf_a);
                lvector double vb = vec_ld(0, buf_b);
                lvector double vx = vec_ld(0, buf_x);
                
                lvector double vprev_a = vec_ld(0, buf_a);
                lvector double vprev_b = vec_ld(0, buf_b);
                lvector double vprev_x = vec_ld(0, buf_x);

                lvector double vdiv = vm_fdivd16(vprev_a, vprev_b);
                lvector double vx_new = vec_mulb(vx, vprev_x, vec_muli(vprev_x, vdiv));
                lvector double vb_new = vec_mulb(vb, vec_muli(vprev_a, vdiv), vec_muli(vprev_a, vdiv));

                vec_st(vx_new, 0, buf_x);
                vec_st(vb_new, 0, buf_b);
                vector_store(buf_x, &X[i1 * n + i2], VEC_BYTES);
                vector_store(buf_b, &B[i1 * n + i2], VEC_BYTES);
            } else {
                for (int jj = i2; jj < n; ++jj) {
                    X[i1 * n + jj] = X[i1 * n + jj] - X[i1 * n + (jj - 1)] * A[i1 * n + jj] / B[i1 * n + (jj - 1)];
                    B[i1 * n + jj] = B[i1 * n + jj] - A[i1 * n + jj] * A[i1 * n + jj] / B[i1 * n + (jj - 1)];
                }
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_x);
}

__global__ void adi_kernel2_vec_qwen(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);

    for (int i1 = start_row; i1 < end_row; i1++) {
        X[i1 * n + (n - 1)] = X[i1 * n + (n - 1)] / B[i1 * n + (n - 1)];
    }
}

__global__ void adi_kernel3_vec_qwen(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int rows_per_thread = n / total_threads;
    int extra_rows = n % total_threads;
    int start_row = thread_id * rows_per_thread + (thread_id < extra_rows ? thread_id : extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b || !buf_x) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_x) vector_free(buf_x);
        return;
    }

    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 0; i2 < n - 2; i2 += SIMD_LEN) {
            if (i2 + SIMD_LEN <= n - 2) {
                vector_load(&A[i1 * n + (n - i2 - 2)], buf_a, VEC_BYTES);
                vector_load(&B[i1 * n + (n - i2 - 2)], buf_b, VEC_BYTES);
                vector_load(&X[i1 * n + (n - i2 - 2)], buf_x, VEC_BYTES);
                
                vector_load(&A[i1 * n + (n - i2 - 3)], buf_a, VEC_BYTES);
                vector_load(&B[i1 * n + (n - i2 - 3)], buf_b, VEC_BYTES);
                vector_load(&X[i1 * n + (n - i2 - 3)], buf_x, VEC_BYTES);

                lvector double va = vec_ld(0, buf_a);
                lvector double vb = vec_ld(0, buf_b);
                lvector double vx = vec_ld(0, buf_x);
                
                lvector double vprev_a = vec_ld(0, buf_a);
                lvector double vprev_b = vec_ld(0, buf_b);
                lvector double vprev_x = vec_ld(0, buf_x);

                lvector double vx_new = vm_fdivd16(vec_mulb(vx, vprev_x, vec_muli(vprev_x, vprev_a)), vprev_b);

                vec_st(vx_new, 0, buf_x);
                vector_store(buf_x, &X[i1 * n + (n - i2 - 2)], VEC_BYTES);
            } else {
                for (int jj = i2; jj < n - 2; ++jj) {
                    X[i1 * n + (n - jj - 2)] = (X[i1 * n + (n - jj - 2)] - X[i1 * n + (n - jj - 3)] * A[i1 * n + (n - jj - 3)]) / B[i1 * n + (n - jj - 3)];
                }
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_x);
}