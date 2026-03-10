#define ELEMS_PER_PART 1024

__global__ void convolution2D_kernel_cache_llm(int ni, int nj, double *A, double *B)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    double c11 = +0.2, c21 = +0.5, c31 = -0.8;
    double c12 = -0.3, c22 = +0.6, c32 = -0.9;
    double c13 = +0.4, c23 = +0.7, c33 = +0.10;

    const int total_tasks = (ni - 2) * (nj - 2);
    if (total_tasks <= 0) return;

    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_above = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_curr  = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_below = (double*)scalar_malloc((ELEMS_PER_PART + 2) * sizeof(double));
    double* cache_out   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int t = start; t < end; )
    {
        int first_i = 1 + t / (nj - 2);
        int first_j = 1 + t % (nj - 2);
        int i = first_i;

        int remain_in_row = (nj - 2) - first_j + 1;
        int batch_tasks = min(ELEMS_PER_PART, min(end - t, remain_in_row));

        int load_start_j = (first_j > 1) ? (first_j - 1) : 0;
        int load_end_j   = min(first_j + batch_tasks, nj - 1);
        int load_len = load_end_j - load_start_j + 1;

        scalar_load(&A[(i-1)*nj + load_start_j], cache_above, load_len * sizeof(double));
        scalar_load(&A[i*nj     + load_start_j], cache_curr,  load_len * sizeof(double));
        scalar_load(&A[(i+1)*nj + load_start_j], cache_below, load_len * sizeof(double));

        for (int bi = 0; bi < batch_tasks; ++bi) {
            int j = first_j + bi;
            int off_m1 = (j - 1) - load_start_j;
            int off_0  = j - load_start_j;
            int off_p1 = (j + 1) - load_start_j;
            
            double val = 0.0;
            val += c11 * cache_above[off_m1] + c21 * cache_above[off_0] + c31 * cache_above[off_p1];
            val += c12 * cache_curr [off_m1] + c22 * cache_curr [off_0] + c32 * cache_curr [off_p1];
            val += c13 * cache_below[off_m1] + c23 * cache_below[off_0] + c33 * cache_below[off_p1];
            cache_out[bi] = val;
        }

        scalar_store(cache_out, &B[i * nj + first_j], batch_tasks * sizeof(double));

        t += batch_tasks;
    }

    scalar_free(cache_above);
    scalar_free(cache_curr);
    scalar_free(cache_below);
    scalar_free(cache_out);
}