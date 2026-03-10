#define ELEMS_PER_PART 1024

__global__ void lu_kernel1_cache_llm(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    double tmp = A[k * n + k];
    double* cache_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start_idx; j < end_idx; )
    {
        int batch_size = min(ELEMS_PER_PART, end_idx - j);
        
        scalar_load(&A[k * n + j], cache_row, batch_size * sizeof(double));
        
        for (int b = 0; b < batch_size; ++b) {
            cache_row[b] = cache_row[b] / tmp;
        }
        
        scalar_store(cache_row, &A[k * n + j], batch_size * sizeof(double));
        
        j += batch_size;
    }

    scalar_free(cache_row);
}

__global__ void lu_kernel2_cache_llm(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    double* cache_k_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_i_k = (double*)scalar_malloc((n - k - 1) * sizeof(double));
    double* cache_block = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j_block_start = start_idx; j_block_start < end_idx; j_block_start += ELEMS_PER_PART)
    {
        int j_block_end = min(j_block_start + ELEMS_PER_PART, end_idx);
        int block_size = j_block_end - j_block_start;

        scalar_load(&A[k * n + j_block_start], cache_k_row, block_size * sizeof(double));

        for (int i = k + 1; i < n; ++i) {
            double a_ik = A[i * n + k];
            
            scalar_load(&A[i * n + j_block_start], cache_block, block_size * sizeof(double));
            
            for (int bj = 0; bj < block_size; ++bj) {
                cache_block[bj] = cache_block[bj] - a_ik * cache_k_row[bj];
            }
            
            scalar_store(cache_block, &A[i * n + j_block_start], block_size * sizeof(double));
        }
    }

    scalar_free(cache_k_row);
    scalar_free(cache_i_k);
    scalar_free(cache_block);
}