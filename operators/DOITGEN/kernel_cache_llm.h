#define ELEMS_PER_PART 1024

__global__ void doitgen_kernel1_cache_llm(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = np * nq;
    const int base = total_tasks / gsz;
    const int rem = total_tasks % gsz;

    int start = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_sum = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_A_row = (double*)scalar_malloc(np * sizeof(double));
    double* cache_C4_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int q = idx / np;
        int p_start = idx % np;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, np - p_start);

        int sum_offset = r * (nq * np) + q * np + p_start;

        scalar_load(&sum[sum_offset], cache_sum, batch_tasks * sizeof(double));
        for (int bi = 0; bi < batch_tasks; ++bi) {
            cache_sum[bi] = 0.0;
        }

        for (int s = 0; s < np; ++s) {
            scalar_load(&A[r * (nq * np) + q * np + s], cache_A_row, sizeof(double));
            
            scalar_load(&C4[s * np + p_start], cache_C4_col, batch_tasks * sizeof(double));
            
            for (int bi = 0; bi < batch_tasks; ++bi) {
                cache_sum[bi] += cache_A_row[0] * cache_C4_col[bi];
            }
        }

        scalar_store(cache_sum, &sum[sum_offset], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_sum);
    scalar_free(cache_A_row);
    scalar_free(cache_C4_col);
}

#define ELEMS_PER_PART 1024

__global__ void doitgen_kernel2_cache_llm(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = np * nq;
    const int base = total_tasks / gsz;
    const int rem = total_tasks % gsz;

    int start = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_sum = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_A = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        int sum_offset = r * (nq * np) + idx;
        int A_offset = r * (nq * np) + idx;

        scalar_load(&sum[sum_offset], cache_sum, batch_tasks * sizeof(double));
        
        for (int bi = 0; bi < batch_tasks; ++bi) {
            cache_A[bi] = cache_sum[bi];
        }
        
        scalar_store(cache_A, &A[A_offset], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_sum);
    scalar_free(cache_A);
}