#define ELEMS_PER_PART 1024

__global__ void gemm_kernel_cache_llm(int ni, int nj, int nk, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);

    double* cache_a_row = (double*)scalar_malloc(nk * sizeof(double));
    double* cache_b_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_c = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nj;
        int j_start = idx % nj;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nj - j_start);

        scalar_load(&c[i * nj + j_start], cache_c, batch_tasks * sizeof(double));

        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_c[bj] *= beta;
        }

        scalar_load(&a[i * nk], cache_a_row, nk * sizeof(double));

        for (int k = 0; k < nk; ++k) {
            scalar_load(&b[k * nj + j_start], cache_b_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_c[bj] += alpha * cache_a_row[k] * cache_b_row[bj];
            }
        }

        scalar_store(cache_c, &c[i * nj + j_start], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_a_row);
    scalar_free(cache_b_row);
    scalar_free(cache_c);
}