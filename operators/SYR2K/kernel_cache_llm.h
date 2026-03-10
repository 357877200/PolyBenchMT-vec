#define ELEMS_PER_PART 1024

__global__ void syr2k_kernel_cache_llm(int ni, int nj, double alpha, double beta, double *a, double *b, double *c)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_tasks = ni * ni;
    int work_per_thread = total_tasks / num_threads;
    int remainder = total_tasks % num_threads;

    int start_idx = thread_id * work_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = start_idx + work_per_thread + (thread_id < remainder ? 1 : 0);

    double* cache_a_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_b_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_c_block = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_a_trans_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_b_trans_row = (double*)scalar_malloc(nj * sizeof(double));

    for (int idx = start_idx; idx < end_idx; )
    {
        int i_start = idx / ni;
        int j_start = idx % ni;
        
        int remaining_in_row = ni - j_start;
        int batch_size = min(ELEMS_PER_PART, end_idx - idx);
        batch_size = min(batch_size, remaining_in_row);

        int i = i_start;
        int j = j_start;

        scalar_load(&c[i * ni + j], cache_c_block, batch_size * sizeof(double));

        for (int b = 0; b < batch_size; ++b) {
            cache_c_block[b] *= beta;
        }

        scalar_load(&a[i * nj], cache_a_row, nj * sizeof(double));

        for (int k = 0; k < nj; ++k) {
            scalar_load(&b[k * ni + j], cache_b_trans_row, batch_size * sizeof(double));
            for (int b = 0; b < batch_size; ++b) {
                cache_c_block[b] += alpha * cache_a_row[k] * cache_b_trans_row[b];
            }
        }

        scalar_load(&b[i * nj], cache_b_row, nj * sizeof(double));

        for (int k = 0; k < nj; ++k) {
            scalar_load(&a[k * ni + j], cache_a_trans_row, batch_size * sizeof(double));
            for (int b = 0; b < batch_size; ++b) {
                cache_c_block[b] += alpha * cache_b_row[k] * cache_a_trans_row[b];
            }
        }

        scalar_store(cache_c_block, &c[i * ni + j], batch_size * sizeof(double));

        idx += batch_size;
    }

    scalar_free(cache_a_row);
    scalar_free(cache_b_row);
    scalar_free(cache_c_block);
    scalar_free(cache_a_trans_row);
    scalar_free(cache_b_trans_row);
}