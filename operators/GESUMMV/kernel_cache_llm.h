#define ELEMS_PER_PART 1024

__global__ void gesummv_kernel_cache_llm(int n, double alpha, double beta, double *A, double *B, double *tmp, double *x, double *y)
{
    int tid = get_thread_id();
    int group_size = get_group_size();

    int base = n / group_size;
    int remainder = n % group_size;

    int start_i = tid * base + (tid < remainder ? tid : remainder);
    int end_i = start_i + base + (tid < remainder ? 1 : 0);

    double* cache_x = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_A = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start_i; i < end_i; i++) {
        double tmp_val = 0.0;
        double y_val = 0.0;

        for (int j_start = 0; j_start < n; j_start += ELEMS_PER_PART) {
            int batch_size = min(ELEMS_PER_PART, n - j_start);
            
            scalar_load(&x[j_start], cache_x, batch_size * sizeof(double));
            scalar_load(&A[i * n + j_start], cache_A, batch_size * sizeof(double));
            scalar_load(&B[i * n + j_start], cache_B, batch_size * sizeof(double));

            for (int jj = 0; jj < batch_size; jj++) {
                tmp_val += cache_A[jj] * cache_x[jj];
                y_val += cache_B[jj] * cache_x[jj];
            }
        }

        tmp[i] = tmp_val;
        y[i] = alpha * tmp_val + beta * y_val;
    }

    scalar_free(cache_x);
    scalar_free(cache_A);
    scalar_free(cache_B);
}