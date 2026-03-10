#define ELEMS_PER_PART 1024

__global__ void corr_kernel1_cache_llm(int m, int n, double *mean, double *data)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = m;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start_j = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_j = start_j + ((tid < rem) ? (base + 1) : base);

    double* cache_mean = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_data = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start_j; j < end_j; j += ELEMS_PER_PART) {
        int batch_size = min(ELEMS_PER_PART, end_j - j);
        
        for (int b = 0; b < batch_size; ++b) {
            cache_mean[b] = 0.0;
        }

        for (int i = 0; i < n; i++) {
            scalar_load(&data[i * m + j], cache_data, batch_size * sizeof(double));
            for (int b = 0; b < batch_size; ++b) {
                cache_mean[b] += cache_data[b];
            }
        }

        for (int b = 0; b < batch_size; ++b) {
            cache_mean[b] /= (double)FLOAT_N;
        }

        scalar_store(cache_mean, &mean[j], batch_size * sizeof(double));
    }

    scalar_free(cache_mean);
    scalar_free(cache_data);
}

__global__ void corr_kernel2_cache_llm(int m, int n, double *mean, double *std, double *data)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = m;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start_j = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_j = start_j + ((tid < rem) ? (base + 1) : base);

    double* cache_mean = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_std = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_data = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start_j; j < end_j; j += ELEMS_PER_PART) {
        int batch_size = min(ELEMS_PER_PART, end_j - j);
        
        scalar_load(&mean[j], cache_mean, batch_size * sizeof(double));
        
        for (int b = 0; b < batch_size; ++b) {
            cache_std[b] = 0.0;
        }

        for (int i = 0; i < n; i++) {
            scalar_load(&data[i * m + j], cache_data, batch_size * sizeof(double));
            for (int b = 0; b < batch_size; ++b) {
                double diff = cache_data[b] - cache_mean[b];
                cache_std[b] += diff * diff;
            }
        }

        for (int b = 0; b < batch_size; ++b) {
            cache_std[b] /= FLOAT_N;
            cache_std[b] = sqrt(cache_std[b]);
            if (cache_std[b] <= EPS) {
                cache_std[b] = 1.0;
            }
        }

        scalar_store(cache_std, &std[j], batch_size * sizeof(double));
    }

    scalar_free(cache_mean);
    scalar_free(cache_std);
    scalar_free(cache_data);
}

__global__ void corr_kernel3_cache_llm(int m, int n, double *mean, double *std, double *data)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = n * m;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start_idx = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_idx = start_idx + ((tid < rem) ? (base + 1) : base);

    double* cache_mean = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_std = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_data = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start_idx; idx < end_idx; ) {
        int i_start = idx / m;
        int j_start = idx % m;
        
        int remain_in_row = m - j_start;
        int batch_size = min(ELEMS_PER_PART, min(end_idx - idx, remain_in_row));
        
        int i = i_start;
        scalar_load(&mean[j_start], cache_mean, batch_size * sizeof(double));
        scalar_load(&std[j_start], cache_std, batch_size * sizeof(double));
        scalar_load(&data[i * m + j_start], cache_data, batch_size * sizeof(double));

        for (int b = 0; b < batch_size; ++b) {
            cache_data[b] -= cache_mean[b];
            cache_data[b] /= (sqrt(FLOAT_N) * cache_std[b]);
        }

        scalar_store(cache_data, &data[i * m + j_start], batch_size * sizeof(double));
        idx += batch_size;
    }

    scalar_free(cache_mean);
    scalar_free(cache_std);
    scalar_free(cache_data);
}

__global__ void corr_kernel4_cache_llm(int m, int n, double *symmat, double *data)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = m - 1;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start_j1 = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_j1 = start_j1 + ((tid < rem) ? (base + 1) : base);

    double* cache_data_j1 = (double*)scalar_malloc(n * sizeof(double));
    double* cache_data_j2 = (double*)scalar_malloc(n * sizeof(double));
    double* cache_symmat = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j1 = start_j1; j1 < end_j1; j1++) {
        symmat[j1 * m + j1] = 1.0;

        for (int i = 0; i < n; i++) {
            cache_data_j1[i] = data[i * m + j1];
        }

        for (int j2_start = j1 + 1; j2_start < m; j2_start += ELEMS_PER_PART) {
            int batch_size = min(ELEMS_PER_PART, m - j2_start);
            
            for (int b = 0; b < batch_size; b++) {
                cache_symmat[b] = 0.0;
            }

            for (int i = 0; i < n; i++) {
                scalar_load(&data[i * m + j2_start], cache_data_j2, batch_size * sizeof(double));
                for (int b = 0; b < batch_size; b++) {
                    cache_symmat[b] += cache_data_j1[i] * cache_data_j2[b];
                }
            }

            for (int b = 0; b < batch_size; b++) {
                int j2 = j2_start + b;
                symmat[j1 * m + j2] = cache_symmat[b];
                symmat[j2 * m + j1] = cache_symmat[b];
            }
        }
    }

    scalar_free(cache_data_j1);
    scalar_free(cache_data_j2);
    scalar_free(cache_symmat);
}