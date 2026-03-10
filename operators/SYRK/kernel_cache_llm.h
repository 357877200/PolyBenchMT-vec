#define ELEMS_PER_PART 1024

__global__ void syrk_kernel_cache_llm(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }

    double* cache_a_i = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_a_j = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_c_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start_idx; i < end_idx; ++i) {
        scalar_load(&a[i * nj], cache_a_i, nj * sizeof(double));
        
        for (int j_start = 0; j_start < ni; ) {
            int batch_size = min(ELEMS_PER_PART, ni - j_start);
            
            scalar_load(&c[i * ni + j_start], cache_c_row, batch_size * sizeof(double));
            
            for (int bj = 0; bj < batch_size; ++bj) {
                int j = j_start + bj;
                cache_c_row[bj] *= beta;
                
                scalar_load(&a[j * nj], cache_a_j, nj * sizeof(double));
                
                double sum = 0.0;
                for (int k = 0; k < nj; k++) {
                    sum += alpha * cache_a_i[k] * cache_a_j[k];
                }
                cache_c_row[bj] += sum;
            }
            
            scalar_store(cache_c_row, &c[i * ni + j_start], batch_size * sizeof(double));
            j_start += batch_size;
        }
    }

    scalar_free(cache_a_i);
    scalar_free(cache_a_j);
    scalar_free(cache_c_row);
}