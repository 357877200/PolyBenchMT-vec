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
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 1; i2 < n; i2++) {
            X[i1 * n + i2] = X[i1 * n + i2] - X[i1 * n + (i2 - 1)] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
            B[i1 * n + i2] = B[i1 * n + i2] - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
        }
    }
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
    for (int i1 = start_row; i1 < end_row; i1++) {
        for (int i2 = 0; i2 < n - 2; i2++) {
            X[i1 * n + (n - i2 - 2)] =
                (X[i1 * n + (n - i2 - 2)] - X[i1 * n + (n - i2 - 3)] * A[i1 * n + (n - i2 - 3)]) /
                B[i1 * n + (n - i2 - 3)];
        }
    }
}
#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void adi_kernel4_vec_qwen(int n, int i1, double *A, double *B, double *X)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();
    int cols_per_thread = n / total_threads;
    int extra_cols = n % total_threads;
    int start_col = thread_id * cols_per_thread + (thread_id < extra_cols ? thread_id : extra_cols);
    int end_col = start_col + cols_per_thread + (thread_id < extra_cols ? 1 : 0);

    lvector double *buf_A = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B_prev = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_X_prev = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    
    if (!buf_A || !buf_B_prev || !buf_X || !buf_X_prev || !buf_res) {
        if (buf_A) vector_free(buf_A);
        if (buf_B_prev) vector_free(buf_B_prev);
        if (buf_X) vector_free(buf_X);
        if (buf_X_prev) vector_free(buf_X_prev);
        if (buf_res) vector_free(buf_res);
        return;
    }

    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int i2 = start_col; i2 < end_col; i2 += SIMD_LEN) {
        int vec_end = min(i2 + SIMD_LEN, end_col);
        
        if (i2 + SIMD_LEN <= end_col) {
            vector_load(&A[i1 * n + i2], buf_A, VEC_BYTES);
            vector_load(&B[(i1 - 1) * n + i2], buf_B_prev, VEC_BYTES);
            vector_load(&X[i1 * n + i2], buf_X, VEC_BYTES);
            vector_load(&X[(i1 - 1) * n + i2], buf_X_prev, VEC_BYTES);

            lvector double vA = vec_ld(0, buf_A);
            lvector double vB_prev = vec_ld(0, buf_B_prev);
            lvector double vX = vec_ld(0, buf_X);
            lvector double vX_prev = vec_ld(0, buf_X_prev);

            lvector double vdiv = vm_fdivd16(vec_muli(vX_prev, vA), vB_prev);
            lvector double vX_new = vec_mulb(vX, one_vec, vdiv);
            
            vec_st(vX_new, 0, buf_res);
            vector_store(buf_res, &X[i1 * n + i2], VEC_BYTES);

            lvector double vA_sq = vec_muli(vA, vA);
            lvector double vdiv2 = vm_fdivd16(vA_sq, vB_prev);
            lvector double vB_new = vec_mulb(vec_ld(0, buf_B_prev), one_vec, vdiv2);
            
            vector_load(&B[i1 * n + i2], buf_res, VEC_BYTES);
            lvector double vB_current = vec_ld(0, buf_res);
            vB_new = vec_mulb(vB_current, one_vec, vdiv2);
            
            vec_st(vB_new, 0, buf_res);
            vector_store(buf_res, &B[i1 * n + i2], VEC_BYTES);
        } else {
            for (int j = i2; j < vec_end; j++) {
                X[i1 * n + j] = X[i1 * n + j] - X[(i1 - 1) * n + j] * A[i1 * n + j] / B[(i1 - 1) * n + j];
                B[i1 * n + j] = B[i1 * n + j] - A[i1 * n + j] * A[i1 * n + j] / B[(i1 - 1) * n + j];
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B_prev);
    vector_free(buf_X);
    vector_free(buf_X_prev);
    vector_free(buf_res);
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
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    
    if (!buf_B || !buf_X || !buf_res) {
        if (buf_B) vector_free(buf_B);
        if (buf_X) vector_free(buf_X);
        if (buf_res) vector_free(buf_res);
        return;
    }

    for (int i2 = start_col; i2 < end_col; i2 += SIMD_LEN) {
        int vec_end = min(i2 + SIMD_LEN, end_col);
        
        if (i2 + SIMD_LEN <= end_col) {
            vector_load(&B[(n - 1) * n + i2], buf_B, VEC_BYTES);
            vector_load(&X[(n - 1) * n + i2], buf_X, VEC_BYTES);

            lvector double vB = vec_ld(0, buf_B);
            lvector double vX = vec_ld(0, buf_X);

            lvector double vres = vm_fdivd16(vX, vB);

            vec_st(vres, 0, buf_res);
            vector_store(buf_res, &X[(n - 1) * n + i2], VEC_BYTES);
        } else {
            for (int j = i2; j < vec_end; j++) {
                X[(n - 1) * n + j] = X[(n - 1) * n + j] / B[(n - 1) * n + j];
            }
        }
    }

    vector_free(buf_B);
    vector_free(buf_X);
    vector_free(buf_res);
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
    lvector double *buf_X_prev = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    
    if (!buf_A || !buf_B || !buf_X || !buf_X_prev || !buf_res) {
        if (buf_A) vector_free(buf_A);
        if (buf_B) vector_free(buf_B);
        if (buf_X) vector_free(buf_X);
        if (buf_X_prev) vector_free(buf_X_prev);
        if (buf_res) vector_free(buf_res);
        return;
    }

    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int i2 = start_col; i2 < end_col; i2 += SIMD_LEN) {
        int vec_end = min(i2 + SIMD_LEN, end_col);
        
        if (i2 + SIMD_LEN <= end_col) {
            vector_load(&A[(n - 3 - i1) * n + i2], buf_A, VEC_BYTES);
            vector_load(&B[(n - 2 - i1) * n + i2], buf_B, VEC_BYTES);
            vector_load(&X[(n - 2 - i1) * n + i2], buf_X, VEC_BYTES);
            vector_load(&X[(n - i1 - 3) * n + i2], buf_X_prev, VEC_BYTES);

            lvector double vA = vec_ld(0, buf_A);
            lvector double vB = vec_ld(0, buf_B);
            lvector double vX = vec_ld(0, buf_X);
            lvector double vX_prev = vec_ld(0, buf_X_prev);

            lvector double vmul = vec_muli(vX_prev, vA);
            lvector double vsub = vec_mulb(vX, one_vec, vmul);
            lvector double vres = vm_fdivd16(vsub, vB);

            vec_st(vres, 0, buf_res);
            vector_store(buf_res, &X[(n - 2 - i1) * n + i2], VEC_BYTES);
        } else {
            for (int j = i2; j < vec_end; j++) {
                X[(n - 2 - i1) * n + j] =
                    (X[(n - 2 - i1) * n + j] - X[(n - i1 - 3) * n + j] * A[(n - 3 - i1) * n + j]) / B[(n - 2 - i1) * n + j];
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
    vector_free(buf_X);
    vector_free(buf_X_prev);
    vector_free(buf_res);
}