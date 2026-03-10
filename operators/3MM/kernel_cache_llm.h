#define ELEMS_PER_PART 1024

__global__ void mm3_kernel1_cache_llm(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nj;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_a_row = (double*)scalar_malloc(nk * sizeof(double));
    double* cache_b_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_e     = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nj;
        int j_start = idx % nj;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nj - j_start);

        scalar_load(&E[i * nj + j_start], cache_e, batch_tasks * sizeof(double));
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_e[bj] = 0.0;
        }

        scalar_load(&A[i * nk], cache_a_row, nk * sizeof(double));

        for (int k = 0; k < nk; ++k) {
            scalar_load(&B[k * nj + j_start], cache_b_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_e[bj] += cache_a_row[k] * cache_b_row[bj];
            }
        }

        scalar_store(cache_e, &E[i * nj + j_start], batch_tasks * sizeof(double));
        idx += batch_tasks;
    }

    scalar_free(cache_a_row);
    scalar_free(cache_b_row);
    scalar_free(cache_e);
}

__global__ void mm3_kernel2_cache_llm(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = nj * nl;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_c_row = (double*)scalar_malloc(nm * sizeof(double));
    double* cache_d_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_f     = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nl;
        int j_start = idx % nl;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nl - j_start);

        scalar_load(&F[i * nl + j_start], cache_f, batch_tasks * sizeof(double));
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_f[bj] = 0.0;
        }

        scalar_load(&C[i * nm], cache_c_row, nm * sizeof(double));

        for (int k = 0; k < nm; ++k) {
            scalar_load(&D[k * nl + j_start], cache_d_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_f[bj] += cache_c_row[k] * cache_d_row[bj];
            }
        }

        scalar_store(cache_f, &F[i * nl + j_start], batch_tasks * sizeof(double));
        idx += batch_tasks;
    }

    scalar_free(cache_c_row);
    scalar_free(cache_d_row);
    scalar_free(cache_f);
}

__global__ void mm3_kernel3_cache_llm(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ni * nl;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_e_row = (double*)scalar_malloc(nj * sizeof(double));
    double* cache_f_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_g     = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int i = idx / nl;
        int j_start = idx % nl;
        int batch_tasks = min(ELEMS_PER_PART, end - idx);
        batch_tasks = min(batch_tasks, nl - j_start);

        scalar_load(&G[i * nl + j_start], cache_g, batch_tasks * sizeof(double));
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_g[bj] = 0.0;
        }

        scalar_load(&E[i * nj], cache_e_row, nj * sizeof(double));

        for (int k = 0; k < nj; ++k) {
            scalar_load(&F[k * nl + j_start], cache_f_row, batch_tasks * sizeof(double));
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_g[bj] += cache_e_row[k] * cache_f_row[bj];
            }
        }

        scalar_store(cache_g, &G[i * nl + j_start], batch_tasks * sizeof(double));
        idx += batch_tasks;
    }

    scalar_free(cache_e_row);
    scalar_free(cache_f_row);
    scalar_free(cache_g);
}