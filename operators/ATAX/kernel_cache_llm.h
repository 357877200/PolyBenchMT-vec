#define ELEMS_PER_PART 1024
// -------------------- atax_kernel1 缓存优化 --------------------
__global__ void atax_kernel1_cache_llm(int nx, int ny, double *A, double *x, double *tmp)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int total_elements = nx;
    int base = total_elements / gsz;
    int rem  = total_elements % gsz;

    int start = (tid < rem) ? tid * (base + 1)
                            : rem * (base + 1) + (tid - rem) * base;
    int end   = start + ((tid < rem) ? (base + 1) : base);

    double* cache_A_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_x     = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = start; i < end; ++i) {
        double sum = 0.0;
        for (int col = 0; col < ny; ) {
            int batch_tasks = min(ELEMS_PER_PART, ny - col);

            // 加载 A行 和 x 批次
            scalar_load(&A[i * ny + col], cache_A_row, batch_tasks * sizeof(double));
            scalar_load(&x[col], cache_x, batch_tasks * sizeof(double));

            for (int bj = 0; bj < batch_tasks; ++bj) {
                sum += cache_A_row[bj] * cache_x[bj];
            }
            col += batch_tasks;
        }
        tmp[i] = sum;
    }

    scalar_free(cache_A_row);
    scalar_free(cache_x);
}

// -------------------- atax_kernel2 缓存优化 --------------------
__global__ void atax_kernel2_cache_llm(int nx, int ny, double *A, double *y, double *tmp)
{
    int tid = get_thread_id(); 
    int gsz = get_group_size();

    int total_elements = ny;
    int base = total_elements / gsz;
    int rem  = total_elements % gsz;

    int start = (tid < rem) ? tid * (base + 1)
                            : rem * (base + 1) + (tid - rem) * base;
    int end   = start + ((tid < rem) ? (base + 1) : base);

    double* cache_A_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_tmp   = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j = start; j < end; ++j) {
        y[j] = 0.0;
    }

    for (int i = 0; i < nx; ++i) {
        for (int col = start; col < end; ) {
            int batch_tasks = min(ELEMS_PER_PART, end - col);

            // 批量加载 A行片段 和 tmp
            scalar_load(&A[i * ny + col], cache_A_col, batch_tasks * sizeof(double));
            scalar_load(&tmp[i], cache_tmp, sizeof(double)); // tmp[i] 只加载一次

            for (int bj = 0; bj < batch_tasks; ++bj) {
                y[col + bj] += cache_A_col[bj] * cache_tmp[0];
            }

            col += batch_tasks;
        }
    }

    scalar_free(cache_A_col);
    scalar_free(cache_tmp);
}