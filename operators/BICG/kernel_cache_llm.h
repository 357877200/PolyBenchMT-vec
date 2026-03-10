#define ELEMS_PER_PART 1024

__global__ void bicg_kernel1_cache_llm(int nx, int ny, double *A, double *r, double *s)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = ny;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start_j = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_j = start_j + ((tid < rem) ? (base + 1) : base);

    double* cache_s = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_r = (double*)scalar_malloc(nx * sizeof(double));
    double* cache_A_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start_j; j < end_j; ) {
        int batch_tasks = min(ELEMS_PER_PART, end_j - j);
        
        scalar_load(&s[j], cache_s, batch_tasks * sizeof(double));
        
        for (int bj = 0; bj < batch_tasks; ++bj) {
            cache_s[bj] = 0.0f;
        }
        
        for (int i = 0; i < nx; ++i) {
            scalar_load(&r[i], &cache_r[i], sizeof(double));
            
            scalar_load(&A[i * ny + j], cache_A_col, batch_tasks * sizeof(double));
            
            for (int bj = 0; bj < batch_tasks; ++bj) {
                cache_s[bj] += cache_r[i] * cache_A_col[bj];
            }
        }
        
        scalar_store(cache_s, &s[j], batch_tasks * sizeof(double));
        j += batch_tasks;
    }

    scalar_free(cache_s);
    scalar_free(cache_r);
    scalar_free(cache_A_col);
}

#define ELEMS_PER_PART 1024

__global__ void bicg_kernel2_cache_llm(int nx, int ny, double *A, double *p, double *q)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = nx;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start_i = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_i = start_i + ((tid < rem) ? (base + 1) : base);

    double* cache_q = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_p = (double*)scalar_malloc(ny * sizeof(double));
    double* cache_A_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start_i; i < end_i; ) {
        int batch_tasks = min(ELEMS_PER_PART, end_i - i);
        
        for (int bi = 0; bi < batch_tasks; ++bi) {
            cache_q[bi] = 0.0f;
        }
        
        for (int j = 0; j < ny; ++j) {
            scalar_load(&p[j], &cache_p[j], sizeof(double));
            
            for (int bi = 0; bi < batch_tasks; ++bi) {
                int row_idx = i + bi;
                scalar_load(&A[row_idx * ny + j], &cache_A_row[bi], sizeof(double));
            }
            
            for (int bi = 0; bi < batch_tasks; ++bi) {
                cache_q[bi] += cache_A_row[bi] * cache_p[j];
            }
        }
        
        scalar_store(cache_q, &q[i], batch_tasks * sizeof(double));
        i += batch_tasks;
    }

    scalar_free(cache_q);
    scalar_free(cache_p);
    scalar_free(cache_A_row);
}