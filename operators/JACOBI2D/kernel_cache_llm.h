#define ELEMS_PER_PART 1024

__global__ void jacobi2D_kernel1_cache_llm(int n, double* A, double* B) 
{
    int gsz = get_group_size();
    int tid = get_thread_id();
    
    int total_rows = n - 2;
    if (total_rows <= 0) return;
    
    int rows_per_thread = (total_rows + gsz - 1) / gsz;
    int start_row = 1 + tid * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);
    
    double* cache_above = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_curr = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_below = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_out = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    
    for (int i = start_row; i < end_row; i++) {
        for (int j_start = 1; j_start < n-1; ) {
            int batch_size = min(ELEMS_PER_PART, (n-1) - j_start);
            
            int load_start_j = (j_start > 1) ? (j_start - 1) : 0;
            int load_end_j = min(j_start + batch_size, n - 1);
            int load_len = load_end_j - load_start_j + 1;
            
            scalar_load(&A[(i-1)*n + load_start_j], cache_above, load_len * sizeof(double));
            scalar_load(&A[i*n + load_start_j], cache_curr, load_len * sizeof(double));
            scalar_load(&A[(i+1)*n + load_start_j], cache_below, load_len * sizeof(double));
            
            for (int bj = 0; bj < batch_size; bj++) {
                int j = j_start + bj;
                int offset_m1 = (j - 1) - load_start_j;
                int offset_0 = j - load_start_j;
                int offset_p1 = (j + 1) - load_start_j;
                
                cache_out[bj] = 0.2f * (cache_curr[offset_0] + 
                                      cache_curr[offset_m1] + 
                                      cache_curr[offset_p1] + 
                                      cache_below[offset_0] + 
                                      cache_above[offset_0]);
            }
            
            scalar_store(cache_out, &B[i*n + j_start], batch_size * sizeof(double));
            j_start += batch_size;
        }
    }
    
    scalar_free(cache_above);
    scalar_free(cache_curr);
    scalar_free(cache_below);
    scalar_free(cache_out);
}

__global__ void jacobi2D_kernel2_cache_llm(int n, double* A, double* B) 
{
    int gsz = get_group_size();
    int tid = get_thread_id();
    
    int total_rows = n - 2;
    if (total_rows <= 0) return;
    
    int rows_per_thread = (total_rows + gsz - 1) / gsz;
    int start_row = 1 + tid * rows_per_thread;
    int end_row = min(start_row + rows_per_thread, n - 1);
    
    double* cache_src = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_dst = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    
    for (int i = start_row; i < end_row; i++) {
        for (int j_start = 1; j_start < n-1; ) {
            int batch_size = min(ELEMS_PER_PART, (n-1) - j_start);
            
            scalar_load(&B[i*n + j_start], cache_src, batch_size * sizeof(double));
            
            for (int bj = 0; bj < batch_size; bj++) {
                cache_dst[bj] = cache_src[bj];
            }
            
            scalar_store(cache_dst, &A[i*n + j_start], batch_size * sizeof(double));
            j_start += batch_size;
        }
    }
    
    scalar_free(cache_src);
    scalar_free(cache_dst);
}