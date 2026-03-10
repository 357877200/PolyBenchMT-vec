#define ELEMS_PER_PART 512
__global__ void convolution3D_kernel_cache_llm(int ni, int nj, int nk, int i, double *A, double *B) {
    int gsz = get_group_size();
    int tid = get_thread_id();
    
    if (nj < 3 || nk < 3) return;
    int total_tasks = (nj - 2) * (nk - 2);
    int base = total_tasks / gsz;
    int rem = total_tasks % gsz;
    
    int start = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);
    
    double c11 = +2, c12 = -3, c13 = +4;
    double c21 = +5, c22 = +6, c23 = +7;
    double c31 = -8, c32 = -9, c33 = +10;
    
    
    
    double* cache_prev_prev = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_prev_curr = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_prev_next = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_curr_prev = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_curr_curr = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_curr_next = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_next_prev = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_next_curr = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_next_next = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_out = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    
    for (int t = start; t < end; ) {
        int first_j = t / (nk - 2) + 1;
        int first_k = t % (nk - 2) + 1;
        int j = first_j;
        
        int remain_in_row = (nk - 2) - first_k + 1;
        int batch_tasks = min(ELEMS_PER_PART, min(end - t, remain_in_row));
        
        int load_start_k = (first_k > 1) ? (first_k - 1) : 0;
        int load_end_k = min(first_k + batch_tasks, nk - 1);
        int load_len = load_end_k - load_start_k + 1;
        
        int nj_nk = nj * nk;
        int i_nj_nk = i * nj_nk;
        int i_minus_1_nj_nk = (i - 1) * nj_nk;
        int i_plus_1_nj_nk = (i + 1) * nj_nk;
        
        scalar_load(&A[i_minus_1_nj_nk + (j-1)*nk + load_start_k], cache_prev_prev, load_len * sizeof(double));
        scalar_load(&A[i_minus_1_nj_nk + j*nk + load_start_k], cache_prev_curr, load_len * sizeof(double));
        scalar_load(&A[i_minus_1_nj_nk + (j+1)*nk + load_start_k], cache_prev_next, load_len * sizeof(double));
        
        scalar_load(&A[i_nj_nk + (j-1)*nk + load_start_k], cache_curr_prev, load_len * sizeof(double));
        scalar_load(&A[i_nj_nk + j*nk + load_start_k], cache_curr_curr, load_len * sizeof(double));
        scalar_load(&A[i_nj_nk + (j+1)*nk + load_start_k], cache_curr_next, load_len * sizeof(double));
        
        scalar_load(&A[i_plus_1_nj_nk + (j-1)*nk + load_start_k], cache_next_prev, load_len * sizeof(double));
        scalar_load(&A[i_plus_1_nj_nk + j*nk + load_start_k], cache_next_curr, load_len * sizeof(double));
        scalar_load(&A[i_plus_1_nj_nk + (j+1)*nk + load_start_k], cache_next_next, load_len * sizeof(double));
        
        for (int bi = 0; bi < batch_tasks; ++bi) {
            int k = first_k + bi;
            int off_m1 = (k - 1) - load_start_k;
            int off_0 = k - load_start_k;
            int off_p1 = (k + 1) - load_start_k;
            
            double res = 
                c11 * cache_prev_prev[off_m1] + c13 * cache_next_prev[off_m1] +
                c21 * cache_prev_curr[off_m1] + c23 * cache_next_curr[off_m1] +
                c31 * cache_prev_next[off_m1] + c33 * cache_next_next[off_m1] +
                c12 * cache_curr_prev[off_0]  + c22 * cache_curr_curr[off_0]  +
                c32 * cache_curr_next[off_0]  + c11 * cache_prev_prev[off_p1] +
                c13 * cache_next_prev[off_p1] + c21 * cache_prev_curr[off_p1] +
                c23 * cache_next_curr[off_p1] + c31 * cache_prev_next[off_p1] +
                c33 * cache_next_next[off_p1];
                
            cache_out[bi] = res;
        }
        
        scalar_store(cache_out, &B[i_nj_nk + j*nk + first_k], batch_tasks * sizeof(double));
        
        t += batch_tasks;
    }
    
    scalar_free(cache_prev_prev);
    scalar_free(cache_prev_curr);
    scalar_free(cache_prev_next);
    scalar_free(cache_curr_prev);
    scalar_free(cache_curr_curr);
    scalar_free(cache_curr_next);
    scalar_free(cache_next_prev);
    scalar_free(cache_next_curr);
    scalar_free(cache_next_next);
    scalar_free(cache_out);
}