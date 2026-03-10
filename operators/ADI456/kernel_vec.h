
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void adi_kernel4_vec_qwen(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);

    lvector double *buf_A = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X_prev = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B_prev = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_A || !buf_B || !buf_X || !buf_X_prev || !buf_B_prev) {
        if (buf_A) vector_free(buf_A);
        if (buf_B) vector_free(buf_B);
        if (buf_X) vector_free(buf_X);
        if (buf_X_prev) vector_free(buf_X_prev);
        if (buf_B_prev) vector_free(buf_B_prev);
        return;
    }

    for (int i2 = start_col; i2 < end_col; i2 += SIMD_LEN) {
        if (i2 + SIMD_LEN <= end_col) {
            vector_load(&A[i1 * n + i2], buf_A, VEC_BYTES);
            vector_load(&B[i1 * n + i2], buf_B, VEC_BYTES);
            vector_load(&X[i1 * n + i2], buf_X, VEC_BYTES);
            vector_load(&X[(i1 - 1) * n + i2], buf_X_prev, VEC_BYTES);
            vector_load(&B[(i1 - 1) * n + i2], buf_B_prev, VEC_BYTES);

            lvector double va = vec_ld(0, buf_A);
            lvector double vb = vec_ld(0, buf_B);
            lvector double vx = vec_ld(0, buf_X);
            lvector double vx_prev = vec_ld(0, buf_X_prev);
            lvector double vb_prev = vec_ld(0, buf_B_prev);

            lvector double vdiv = vm_fdivd16(vb_prev, (lvector double)vec_svbcast(1e-6));
            lvector double vterm1 = vec_muli(vx_prev, va);
            lvector double vterm2 = vm_fdivd16(vterm1, vdiv);
            lvector double vx_new = vec_mulb(vx, (lvector double)vec_svbcast(1.0), vterm2);
            lvector double vterm3 = vec_muli(va, va);
            lvector double vb_new = vec_mulb(vb, (lvector double)vec_svbcast(1.0), vm_fdivd16(vterm3, vdiv));

            vec_st(vx_new, 0, buf_X);
            vec_st(vb_new, 0, buf_B);
            vector_store(buf_X, &X[i1 * n + i2], VEC_BYTES);
            vector_store(buf_B, &B[i1 * n + i2], VEC_BYTES);
        } else {
            for (int jj = i2; jj < end_col; ++jj) {
                X[i1 * n + jj] = X[i1 * n + jj] - X[(i1 - 1) * n + jj] * A[i1 * n + jj] / (B[(i1 - 1) * n + jj] + 1e-6);
                B[i1 * n + jj] = B[i1 * n + jj] - A[i1 * n + jj] * A[i1 * n + jj] / (B[(i1 - 1) * n + jj] + 1e-6);
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
    vector_free(buf_X);
    vector_free(buf_X_prev);
    vector_free(buf_B_prev);
}

__global__ void adi_kernel5_vec_qwen(int n, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);

    lvector double *buf_B = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_B || !buf_X) {
        if (buf_B) vector_free(buf_B);
        if (buf_X) vector_free(buf_X);
        return;
    }

    for (int i2 = start_col; i2 < end_col; i2 += SIMD_LEN) {
        if (i2 + SIMD_LEN <= end_col) {
            vector_load(&B[(n - 1) * n + i2], buf_B, VEC_BYTES);
            vector_load(&X[(n - 1) * n + i2], buf_X, VEC_BYTES);

            lvector double vb = vec_ld(0, buf_B);
            lvector double vx = vec_ld(0, buf_X);

            lvector double vb_safe = vec_mula(vb, (lvector double)vec_svbcast(1.0), (lvector double)vec_svbcast(1e-6));
            lvector double vx_new = vm_fdivd16(vx, vb_safe);

            vec_st(vx_new, 0, buf_X);
            vector_store(buf_X, &X[(n - 1) * n + i2], VEC_BYTES);
        } else {
            for (int jj = i2; jj < end_col; ++jj) {
                X[(n - 1) * n + jj] = X[(n - 1) * n + jj] / (B[(n - 1) * n + jj] + 1e-6);
            }
        }
    }

    vector_free(buf_B);
    vector_free(buf_X);
}

__global__ void adi_kernel6_vec_qwen(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);

    lvector double *buf_A = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X_next = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_A || !buf_B || !buf_X || !buf_X_next) {
        if (buf_A) vector_free(buf_A);
        if (buf_B) vector_free(buf_B);
        if (buf_X) vector_free(buf_X);
        if (buf_X_next) vector_free(buf_X_next);
        return;
    }

    for (int i2 = start_col; i2 < end_col; i2 += SIMD_LEN) {
        if (i2 + SIMD_LEN <= end_col) {
            vector_load(&A[(n - 3 - i1) * n + i2], buf_A, VEC_BYTES);
            vector_load(&B[(n - 2 - i1) * n + i2], buf_B, VEC_BYTES);
            vector_load(&X[(n - 2 - i1) * n + i2], buf_X, VEC_BYTES);
            vector_load(&X[(n - i1 - 3) * n + i2], buf_X_next, VEC_BYTES);

            lvector double va = vec_ld(0, buf_A);
            lvector double vb = vec_ld(0, buf_B);
            lvector double vx = vec_ld(0, buf_X);
            lvector double vx_next = vec_ld(0, buf_X_next);

            lvector double vb_safe = vec_mula(vb, (lvector double)vec_svbcast(1.0), (lvector double)vec_svbcast(1e-6));
            lvector double vterm1 = vec_muli(vx_next, va);
            lvector double vterm2 = vec_mulb(vx, (lvector double)vec_svbcast(1.0), vterm1);
            lvector double vx_new = vm_fdivd16(vterm2, vb_safe);

            vec_st(vx_new, 0, buf_X);
            vector_store(buf_X, &X[(n - 2 - i1) * n + i2], VEC_BYTES);
        } else {
            for (int jj = i2; jj < end_col; ++jj) {
                X[(n - 2 - i1) * n + jj] = (X[(n - 2 - i1) * n + jj] - X[(n - i1 - 3) * n + jj] * A[(n - 3 - i1) * n + jj]) / (B[(n - 2 - i1) * n + jj] + 1e-6);
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
    vector_free(buf_X);
    vector_free(buf_X_next);
}