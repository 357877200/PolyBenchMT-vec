
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void jacobi2D_kernel1_vec_qwen(int n, double* A, double* B) 
{
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);
    
    lvector double *buf_A = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_A || !buf_B) {
        if (buf_A) vector_free(buf_A);
        if (buf_B) vector_free(buf_B);
        return;
    }

    lvector double coeff_vec = (lvector double)vec_svbcast(0.2);
    
    for(int i = start_row; i < end_row; i++) {
        for(int j = 1; j < n-1; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, n-1);

            if (j + SIMD_LEN <= n-1) {
                vector_load(&A[i*n + j], buf_A, VEC_BYTES);
                lvector double va_center = vec_ld(0, buf_A);
                
                vector_load(&A[i*n + (j-1)], buf_A, VEC_BYTES);
                lvector double va_left = vec_ld(0, buf_A);
                
                vector_load(&A[i*n + (j+1)], buf_A, VEC_BYTES);
                lvector double va_right = vec_ld(0, buf_A);
                
                vector_load(&A[(i+1)*n + j], buf_A, VEC_BYTES);
                lvector double va_down = vec_ld(0, buf_A);
                
                vector_load(&A[(i-1)*n + j], buf_A, VEC_BYTES);
                lvector double va_up = vec_ld(0, buf_A);

                lvector double sum = vec_mula(va_center, (lvector double)vec_svbcast(1.0), va_left);
                sum = vec_mula(sum, (lvector double)vec_svbcast(1.0), va_right);
                sum = vec_mula(sum, (lvector double)vec_svbcast(1.0), va_down);
                sum = vec_mula(sum, (lvector double)vec_svbcast(1.0), va_up);
                
                lvector double vres = vec_muli(sum, coeff_vec);
                
                vec_st(vres, 0, buf_B);
                vector_store(buf_B, &B[i*n + j], VEC_BYTES);
            } else {
                for(int jj = j; jj < vec_end_j; jj++) {
                    B[i*n + jj] = 0.2 * (A[i*n + jj] + 
                                       A[i*n + (jj-1)] + 
                                       A[i*n + (jj+1)] + 
                                       A[(i+1)*n + jj] + 
                                       A[(i-1)*n + jj]);
                }
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
}

__global__ void jacobi2D_kernel2_vec_qwen(int n, double* A, double* B) 
{
    int total_threads = get_group_size();
    int thread_id = get_thread_id();
    
    int rows_per_thread = (n - 2 + total_threads - 1) / total_threads;
    int start_row = 1 + thread_id * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);
    
    lvector double *buf_A = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_B = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_A || !buf_B) {
        if (buf_A) vector_free(buf_A);
        if (buf_B) vector_free(buf_B);
        return;
    }
    
    for(int i = start_row; i < end_row; i++) {
        for(int j = 1; j < n-1; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, n-1);

            if (j + SIMD_LEN <= n-1) {
                vector_load(&B[i*n + j], buf_B, VEC_BYTES);
                lvector double vb = vec_ld(0, buf_B);
                
                vec_st(vb, 0, buf_A);
                vector_store(buf_A, &A[i*n + j], VEC_BYTES);
            } else {
                for(int jj = j; jj < vec_end_j; jj++) {
                    A[i*n + jj] = B[i*n + jj];
                }
            }
        }
    }

    vector_free(buf_A);
    vector_free(buf_B);
}