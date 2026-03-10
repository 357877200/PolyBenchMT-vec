#define ELEMS_PER_PART 1024

__global__ void jacobi1D_kernel1_cache_llm(int n, double *A, double *B)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    if (n < 3) return;
    const int total_tasks = n - 2;
    const int base = total_tasks / gsz;
    const int rem = total_tasks % gsz;

    int start = (tid < rem) 
        ? tid * (base + 1) 
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_A = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int t = start; t < end; )
    {
        int first_i = 1 + t;
        int batch_tasks = min(ELEMS_PER_PART, end - t);

        int load_start = (first_i > 1) ? (first_i - 1) : 0;
        int load_end = min(first_i + batch_tasks, n - 1);
        int load_len = load_end - load_start + 1;

        scalar_load(&A[load_start], cache_A, load_len * sizeof(double));

        for (int bi = 0; bi < batch_tasks; ++bi) {
            int i = first_i + bi;
            int local_idx = i - load_start;
            cache_B[bi] = 0.33333f * (cache_A[local_idx - 1] + cache_A[local_idx] + cache_A[local_idx + 1]);
        }

        scalar_store(cache_B, &B[first_i], batch_tasks * sizeof(double));

        t += batch_tasks;
    }

    scalar_free(cache_A);
    scalar_free(cache_B);
}

#define ELEMS_PER_PART 1024

__global__ void jacobi1D_kernel2_cache_llm(int n, double *A, double *B)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    if (n < 3) return;
    const int total_tasks = n - 2;
    const int base = total_tasks / gsz;
    const int rem = total_tasks % gsz;

    int start = (tid < rem) 
        ? tid * (base + 1) 
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_A = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int t = start; t < end; )
    {
        int first_i = 1 + t;
        int batch_tasks = min(ELEMS_PER_PART, end - t);

        scalar_load(&B[first_i], cache_B, batch_tasks * sizeof(double));

        for (int bi = 0; bi < batch_tasks; ++bi) {
            cache_A[bi] = cache_B[bi];
        }

        scalar_store(cache_A, &A[first_i], batch_tasks * sizeof(double));

        t += batch_tasks;
    }

    scalar_free(cache_B);
    scalar_free(cache_A);
}