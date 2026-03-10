__global__ void mm2_kernel1_cache_llm(int ni, int nj, int nk, double alpha, double *tmp, double *A, double *B)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nj;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);


    double* cache_a_row = (double*)scalar_malloc(nk * sizeof(double));
    double* cache_b_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_tmp   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nj;
        int j_start = idx % nj;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nj - j_start);

        scalar_load(&tmp[i * nj + j_start], cache_tmp, batch_tasks * sizeof(double));
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_tmp[bj] = 0.0;
        }

        scalar_load(&A[i * nk], cache_a_row, nk * sizeof(double));

        for (int k = 0; k < nk; ++k) {
            scalar_load(&B[k * nj + j_start], cache_b_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_tmp[bj] += alpha * cache_a_row[k] * cache_b_row[bj];
            }
        }

        scalar_store(cache_tmp, &tmp[i * nj + j_start], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_a_row);
    scalar_free(cache_b_row);
    scalar_free(cache_tmp);
}

__global__ void mm2_kernel2_cache_llm(int ni, int nj, int nl, double beta, double *tmp, double *C, double *D)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nl;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);


    double* cache_tmp_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_c_row   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_d       = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nl;
        int j_start = idx % nl;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nl - j_start);

        scalar_load(&D[i * nl + j_start], cache_d, batch_tasks * sizeof(double));
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_d[bj] *= beta;
        }

        scalar_load(&tmp[i * nj], cache_tmp_row, nj * sizeof(double));

        for (int k = 0; k < nj; ++k) {
            scalar_load(&C[k * nl + j_start], cache_c_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_d[bj] += cache_tmp_row[k] * cache_c_row[bj];
            }
        }

        scalar_store(cache_d, &D[i * nl + j_start], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_tmp_row);
    scalar_free(cache_c_row);
    scalar_free(cache_d);
}