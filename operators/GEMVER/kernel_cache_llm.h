#define ELEMS_PER_PART 1024

__global__ void gemver_kernel1_cache_llm(int n, double alpha, double beta, double *a, double *v1, double *v2,
                               double *u1, double *u2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    double* cache_u1 = (double*)scalar_malloc(sizeof(double));
    double* cache_u2 = (double*)scalar_malloc(sizeof(double));
    double* cache_v1 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_v2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_a_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start_idx; i < end_idx; ++i) {
        scalar_load(&u1[i], cache_u1, sizeof(double));
        scalar_load(&u2[i], cache_u2, sizeof(double));
        
        for (int j_start = 0; j_start < n; j_start += ELEMS_PER_PART) {
            int batch_size = min(ELEMS_PER_PART, n - j_start);
            
            scalar_load(&v1[j_start], cache_v1, batch_size * sizeof(double));
            scalar_load(&v2[j_start], cache_v2, batch_size * sizeof(double));
            scalar_load(&a[i * n + j_start], cache_a_row, batch_size * sizeof(double));
            
            for (int jj = 0; jj < batch_size; ++jj) {
                cache_a_row[jj] += (*cache_u1) * cache_v1[jj] + (*cache_u2) * cache_v2[jj];
            }
            
            scalar_store(cache_a_row, &a[i * n + j_start], batch_size * sizeof(double));
        }
    }

    scalar_free(cache_u1);
    scalar_free(cache_u2);
    scalar_free(cache_v1);
    scalar_free(cache_v2);
    scalar_free(cache_a_row);
}

__global__ void gemver_kernel2_cache_llm(int n, double alpha, double beta, double *a, double *x, double *y,
                               double *z)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    double* cache_x = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_y = (double*)scalar_malloc(sizeof(double));
    double* cache_a_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_z = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = 0; j < n; j++) {
        scalar_load(&y[j], cache_y, sizeof(double));
        
        for (int i_start = start_idx; i_start < end_idx; i_start += ELEMS_PER_PART) {
            int batch_size = min(ELEMS_PER_PART, end_idx - i_start);
            
            scalar_load(&x[i_start], cache_x, batch_size * sizeof(double));
            scalar_load(&a[j * n + i_start], cache_a_col, batch_size * sizeof(double));
            
            for (int ii = 0; ii < batch_size; ++ii) {
                cache_x[ii] += beta * cache_a_col[ii] * (*cache_y);
            }
            
            scalar_store(cache_x, &x[i_start], batch_size * sizeof(double));
        }
    }

    for (int i_start = start_idx; i_start < end_idx; i_start += ELEMS_PER_PART) {
        int batch_size = min(ELEMS_PER_PART, end_idx - i_start);
        
        scalar_load(&x[i_start], cache_x, batch_size * sizeof(double));
        scalar_load(&z[i_start], cache_z, batch_size * sizeof(double));
        
        for (int ii = 0; ii < batch_size; ++ii) {
            cache_x[ii] += cache_z[ii];
        }
        
        scalar_store(cache_x, &x[i_start], batch_size * sizeof(double));
    }

    scalar_free(cache_x);
    scalar_free(cache_y);
    scalar_free(cache_a_col);
    scalar_free(cache_z);
}

__global__ void gemver_kernel3_cache_llm(int n, double alpha, double beta, double *a, double *x, double *w)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = tid * elements_per_thread + (tid < remainder ? tid : remainder);
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    double* cache_w = (double*)scalar_malloc(sizeof(double));
    double* cache_x = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_a_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start_idx; i < end_idx; ++i) {
        scalar_load(&w[i], cache_w, sizeof(double));
        
        for (int j_start = 0; j_start < n; j_start += ELEMS_PER_PART) {
            int batch_size = min(ELEMS_PER_PART, n - j_start);
            
            scalar_load(&x[j_start], cache_x, batch_size * sizeof(double));
            scalar_load(&a[i * n + j_start], cache_a_row, batch_size * sizeof(double));
            
            for (int jj = 0; jj < batch_size; ++jj) {
                *cache_w += alpha * cache_a_row[jj] * cache_x[jj];
            }
        }
        
        scalar_store(cache_w, &w[i], sizeof(double));
    }

    scalar_free(cache_w);
    scalar_free(cache_x);
    scalar_free(cache_a_row);
}