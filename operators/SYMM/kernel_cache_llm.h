#define ELEMS_PER_PART 1024

__global__ void symm_kernel1_cache_llm(int m, int n, double alpha,
    double *A, double *B, double *C, double *temp2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int cols_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * cols_per_thread;
    int end_j = min(start_j + cols_per_thread, n);

    double* cache_A_row = (double*)scalar_malloc(m * sizeof(double));
    double* cache_B_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_C_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_temp2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start_j; j < end_j; ) {
        int batch_cols = min(ELEMS_PER_PART, end_j - j);
        
        for (int i = 0; i < m; i++) {
            scalar_load(&A[i * m], cache_A_row, m * sizeof(double));
            
            scalar_load(&B[i * n + j], cache_B_col, batch_cols * sizeof(double));
            
            for (int bj = 0; bj < batch_cols; bj++) {
                cache_temp2[bj] = 0.0;
            }
            
            for (int k = 0; k < i; k++) {
                scalar_load(&C[k * n + j], cache_C_col, batch_cols * sizeof(double));
                
                for (int bj = 0; bj < batch_cols; bj++) {
                    cache_C_col[bj] += alpha * cache_B_col[bj] * cache_A_row[k];
                }
                
                scalar_store(cache_C_col, &C[k * n + j], batch_cols * sizeof(double));
                
                for (int bj = 0; bj < batch_cols; bj++) {
                    cache_temp2[bj] += B[k * n + j + bj] * cache_A_row[k];
                }
            }
            
            scalar_store(cache_temp2, &temp2[i * n + j], batch_cols * sizeof(double));
        }
        
        j += batch_cols;
    }

    scalar_free(cache_A_row);
    scalar_free(cache_B_col);
    scalar_free(cache_C_col);
    scalar_free(cache_temp2);
}

__global__ void symm_kernel2_cache_llm(int m, int n, double alpha, double beta,
    double *A, double *B, double *C, double *temp2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int cols_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * cols_per_thread;
    int end_j = min(start_j + cols_per_thread, n);

    double* cache_A_diag = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_C_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_temp2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start_j; j < end_j; ) {
        int batch_cols = min(ELEMS_PER_PART, end_j - j);
        
        for (int i = 0; i < m; i++) {
            scalar_load(&C[i * n + j], cache_C_col, batch_cols * sizeof(double));
            scalar_load(&B[i * n + j], cache_B_col, batch_cols * sizeof(double));
            scalar_load(&temp2[i * n + j], cache_temp2, batch_cols * sizeof(double));
            
            double A_ii = A[i * m + i];
            
            for (int bj = 0; bj < batch_cols; bj++) {
                cache_C_col[bj] = beta * cache_C_col[bj]
                                + alpha * cache_B_col[bj] * A_ii
                                + alpha * cache_temp2[bj];
            }
            
            scalar_store(cache_C_col, &C[i * n + j], batch_cols * sizeof(double));
        }
        
        j += batch_cols;
    }

    scalar_free(cache_A_diag);
    scalar_free(cache_B_col);
    scalar_free(cache_C_col);
    scalar_free(cache_temp2);
}