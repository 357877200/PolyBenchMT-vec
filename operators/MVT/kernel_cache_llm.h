#define ELEMS_PER_PART 1024

__global__ void mvt_kernel1_cache_llm(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    double* cache_a_row = (double*)scalar_malloc(n * sizeof(double));
    double* cache_y1 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_x1 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start_idx; i < end_idx; )
    {
        int batch_size = min(ELEMS_PER_PART, end_idx - i);
        
        scalar_load(&x1[i], cache_x1, batch_size * sizeof(double));
        
        for (int bi = 0; bi < batch_size; ++bi) {
            int current_i = i + bi;
            scalar_load(&a[current_i * n], cache_a_row, n * sizeof(double));
            
            double sum = 0.0;
            for (int j = 0; j < n; j += ELEMS_PER_PART) {
                int j_batch = min(ELEMS_PER_PART, n - j);
                scalar_load(&y_1[j], cache_y1, j_batch * sizeof(double));
                
                for (int jj = 0; jj < j_batch; ++jj) {
                    sum += cache_a_row[j + jj] * cache_y1[jj];
                }
            }
            cache_x1[bi] += sum;
        }
        
        scalar_store(cache_x1, &x1[i], batch_size * sizeof(double));
        i += batch_size;
    }

    scalar_free(cache_a_row);
    scalar_free(cache_y1);
    scalar_free(cache_x1);
}

__global__ void mvt_kernel2_cache_llm(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    double* cache_a_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_x2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_y2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = 0; j < n; j += ELEMS_PER_PART)
    {
        int j_batch = min(ELEMS_PER_PART, n - j);
        scalar_load(&y_2[j], cache_y2, j_batch * sizeof(double));
        
        for (int i = start_idx; i < end_idx; )
        {
            int i_batch = min(ELEMS_PER_PART, end_idx - i);
            scalar_load(&x2[i], cache_x2, i_batch * sizeof(double));
            
            for (int jj = 0; jj < j_batch; ++jj) {
                int current_j = j + jj;
                scalar_load(&a[current_j * n + i], cache_a_col, i_batch * sizeof(double));
                
                for (int ii = 0; ii < i_batch; ++ii) {
                    cache_x2[ii] += cache_a_col[ii] * cache_y2[jj];
                }
            }
            
            scalar_store(cache_x2, &x2[i], i_batch * sizeof(double));
            i += i_batch;
        }
    }

    scalar_free(cache_a_col);
    scalar_free(cache_x2);
    scalar_free(cache_y2);
}