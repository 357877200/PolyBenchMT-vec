#define ELEMS_PER_PART 1024

__global__ void covar_kernel1_cache_llm(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int j_start = (m * thread_id) / group_size;
    int j_end = (m * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) {
        j_end = m;
    }

    double* cache_mean = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_data = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = j_start; j < j_end; ) {
        int batch_size = min(ELEMS_PER_PART, j_end - j);
        
        for (int bj = 0; bj < batch_size; ++bj) {
            cache_mean[bj] = 0.0;
        }

        for (int i = 0; i < n; i++) {
            scalar_load(&data[i * m + j], cache_data, batch_size * sizeof(double));
            for (int bj = 0; bj < batch_size; ++bj) {
                cache_mean[bj] += cache_data[bj];
            }
        }

        for (int bj = 0; bj < batch_size; ++bj) {
            cache_mean[bj] /= (double)n;
        }

        scalar_store(cache_mean, &mean[j], batch_size * sizeof(double));
        j += batch_size;
    }

    scalar_free(cache_mean);
    scalar_free(cache_data);
}

__global__ void covar_kernel2_cache_llm(int m, int n, double *mean, double *data)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int total_elements = m * n;
    int base = total_elements / gsz;
    int rem  = total_elements % gsz;

    int start_idx = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end_idx = start_idx + ((tid < rem) ? (base + 1) : base);

    double* cache_data = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_mean = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start_idx; idx < end_idx; ) {
        int batch_size = min(ELEMS_PER_PART, end_idx - idx);

        // 批量加载 data 元素
        scalar_load(&data[idx], cache_data, batch_size * sizeof(double));

        for (int bi = 0; bi < batch_size; ++bi) {
            int cur_idx = idx + bi;
            int j = cur_idx % m;  // 列号
            // 逐个加载对应列均值（一次能缓存多个）
            scalar_load(&mean[j], &cache_mean[bi], sizeof(double));
            cache_data[bi] -= cache_mean[bi];
        }

        // 批量写回
        scalar_store(cache_data, &data[idx], batch_size * sizeof(double));

        idx += batch_size;
    }

    scalar_free(cache_data);
    scalar_free(cache_mean);
}

__global__ void covar_kernel3_cache_llm(int m, int n, double *symmat, double *data)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = m * m;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    double* cache_symmat = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_data_j1 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_data_j2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start_idx; idx < end_idx; ) {
        int j1 = idx % m;
        int j2 = idx / m;
        
        int batch_size = min(ELEMS_PER_PART, end_idx - idx);
        batch_size = min(batch_size, m - j1);

        for (int bj = 0; bj < batch_size; ++bj) {
            int current_j1 = j1 + bj;
            int current_j2 = j2;
            
            if (current_j1 <= current_j2) {
                cache_symmat[bj] = 0.0;
                
                for (int i = 0; i < n; i++) {
                    scalar_load(&data[i * m + current_j1], &cache_data_j1[bj], sizeof(double));
                    scalar_load(&data[i * m + current_j2], &cache_data_j2[bj], sizeof(double));
                    cache_symmat[bj] += cache_data_j1[bj] * cache_data_j2[bj];
                }
                
                scalar_store(&cache_symmat[bj], &symmat[current_j1 * m + current_j2], sizeof(double));
                scalar_store(&cache_symmat[bj], &symmat[current_j2 * m + current_j1], sizeof(double));
            }
        }

        idx += batch_size;
    }

    scalar_free(cache_symmat);
    scalar_free(cache_data_j1);
    scalar_free(cache_data_j2);
}