#define ELEMS_PER_PART 1024
__global__ void adi_kernel1_cache_llm(int n, double *A, double *B, double *X)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int rows_per_thread = n / gsz;
    int extra_rows = n % gsz;
    int start_row = tid * rows_per_thread + (tid < extra_rows ? tid : extra_rows);
    int end_row   = start_row + rows_per_thread + (tid < extra_rows ? 1 : 0);

    // 分配缓存：整行缓存
    double* cache_A = (double*)scalar_malloc(n * sizeof(double));
    double* cache_B = (double*)scalar_malloc(n * sizeof(double));
    double* cache_X = (double*)scalar_malloc(n * sizeof(double));

    for (int i1 = start_row; i1 < end_row; ++i1)
    {
        // 整行一次性加载到缓存
        scalar_load(&A[i1 * n], cache_A, n * sizeof(double));
        scalar_load(&B[i1 * n], cache_B, n * sizeof(double));
        scalar_load(&X[i1 * n], cache_X, n * sizeof(double));

        // 顺序递推计算
        for (int i2 = 1; i2 < n; ++i2) {
            cache_X[i2] = cache_X[i2] - cache_X[i2 - 1] * cache_A[i2] / cache_B[i2 - 1];
            cache_B[i2] = cache_B[i2] - cache_A[i2] * cache_A[i2] / cache_B[i2 - 1];
        }

        // 写回结果
        scalar_store(cache_X, &X[i1 * n], n * sizeof(double));
        scalar_store(cache_B, &B[i1 * n], n * sizeof(double));
    }

    scalar_free(cache_A);
    scalar_free(cache_B);
    scalar_free(cache_X);
}

__global__ void adi_kernel2_cache_llm(int n, double *A, double *B, double *X)
{
    int gsz = get_group_size();
    int tid = get_thread_id();
    
    int rows_per_thread = n / gsz;
    int extra_rows = n % gsz;
    int start_row = tid * rows_per_thread + (tid < extra_rows ? tid : extra_rows);
    int end_row = start_row + rows_per_thread + (tid < extra_rows ? 1 : 0);
    
    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_X = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    
    for (int i1 = start_row; i1 < end_row; ) {
        int batch_size = min(ELEMS_PER_PART, end_row - i1);
        
        for (int i = 0; i < batch_size; i++) {
            int actual_i = i1 + i;
            scalar_load(&X[actual_i * n + (n - 1)], &cache_X[i], sizeof(double));
            scalar_load(&B[actual_i * n + (n - 1)], &cache_B[i], sizeof(double));
            cache_X[i] = cache_X[i] / cache_B[i];
        }
        
        for (int i = 0; i < batch_size; i++) {
            int actual_i = i1 + i;
            scalar_store(&cache_X[i], &X[actual_i * n + (n - 1)], sizeof(double));
        }
        
        i1 += batch_size;
    }
    
    scalar_free(cache_B);
    scalar_free(cache_X);
}

__global__ void adi_kernel3_cache_llm(int n, double *A, double *B, double *X)
{
    int tid = get_thread_id(); 
    int gsz = get_group_size();

    int rows_per_thread = n / gsz;
    int extra_rows = n % gsz;
    int start_row = tid * rows_per_thread + (tid < extra_rows ? tid : extra_rows);
    int end_row   = start_row + rows_per_thread + (tid < extra_rows ? 1 : 0);

    double* cache_A = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_X = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i1 = start_row; i1 < end_row; ++i1)
    {
        for (int offset = 0; offset < n - 2; )
        {
            int batch_tasks = min(ELEMS_PER_PART, (n - 2) - offset);

            // 批量加载：A/B/X按反向计算的序列
            for (int bj = 0; bj < batch_tasks; ++bj) {
                int idx = n - offset - bj - 3; // n - i2 - 3
                cache_A[bj] = A[i1 * n + idx];
                cache_B[bj] = B[i1 * n + idx];
                cache_X[bj] = X[i1 * n + idx + 1]; // n - i2 - 2
            }

            for (int bj = 0; bj < batch_tasks; ++bj) {
                int idx = n - offset - bj - 3;
                cache_X[bj] = (cache_X[bj] - X[i1 * n + idx] * cache_A[bj]) / cache_B[bj];
            }

            // 写回计算结果
            for (int bj = 0; bj < batch_tasks; ++bj) {
                int write_idx = n - offset - bj - 2;
                X[i1 * n + write_idx] = cache_X[bj];
            }

            offset += batch_tasks;
        }
    }

    scalar_free(cache_A);
    scalar_free(cache_B);
    scalar_free(cache_X);
}

__global__ void adi_kernel4_cache_llm(int n, int i1, double *A, double *B, double *X)
{
    int tid = get_thread_id();
    int gsz = get_group_size();
 
    // 按列分配给线程
    int cols_per_thread = n / gsz;
    int extra_cols = n % gsz;
    int start_col = tid * cols_per_thread + (tid < extra_cols ? tid : extra_cols);
    int end_col   = start_col + cols_per_thread + (tid < extra_cols ? 1 : 0);
 
    // 分配批处理缓存
    double* cache_A     = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B_cur = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B_prev= (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_X_cur = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_X_prev= (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
 
    // 批量处理列
    for (int col = start_col; col < end_col; )
    {
        int batch_tasks = min(ELEMS_PER_PART, end_col - col);
 
        // 批量加载当前位置的 A、B、X
        scalar_load(&A[i1 * n + col], cache_A,     batch_tasks * sizeof(double));
        scalar_load(&B[i1 * n + col], cache_B_cur, batch_tasks * sizeof(double));
        scalar_load(&B[(i1 - 1) * n + col], cache_B_prev, batch_tasks * sizeof(double));
        scalar_load(&X[i1 * n + col], cache_X_cur, batch_tasks * sizeof(double));
        scalar_load(&X[(i1 - 1) * n + col], cache_X_prev, batch_tasks * sizeof(double));
 
        // 批量更新计算
        for (int k = 0; k < batch_tasks; ++k) {
            cache_X_cur[k] = cache_X_cur[k] - cache_X_prev[k] * cache_A[k] / cache_B_prev[k];
            cache_B_cur[k] = cache_B_cur[k] - cache_A[k] * cache_A[k] / cache_B_prev[k];
        }
 
        // 批量写回
        scalar_store(cache_X_cur, &X[i1 * n + col], batch_tasks * sizeof(double));
        scalar_store(cache_B_cur, &B[i1 * n + col], batch_tasks * sizeof(double));
 
        col += batch_tasks;
    }
 
    scalar_free(cache_A);
    scalar_free(cache_B_cur);
    scalar_free(cache_B_prev);
    scalar_free(cache_X_cur);
    scalar_free(cache_X_prev);
}
__global__ void adi_kernel5_cache_llm(int n, double *A, double *B, double *X)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = n;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_X = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int batch_tasks = min(ELEMS_PER_PART, end - idx);

        scalar_load(&X[(n - 1) * n + idx], cache_X, batch_tasks * sizeof(double));
        scalar_load(&B[(n - 1) * n + idx], cache_B, batch_tasks * sizeof(double));

        for (int bi = 0; bi < batch_tasks; ++bi) {
            cache_X[bi] = cache_X[bi] / cache_B[bi];
        }

        scalar_store(cache_X, &X[(n - 1) * n + idx], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_X);
    scalar_free(cache_B);
}

__global__ void adi_kernel6_cache_llm(int n, int i1, double *A, double *B, double *X)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_tasks = n;
    const int base = total_tasks / gsz;
    const int rem  = total_tasks % gsz;

    int start = (tid < rem)
        ? tid * (base + 1)
        : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_X_curr = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_X_prev = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_A = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; )
    {
        int batch_tasks = min(ELEMS_PER_PART, end - idx);

        int curr_row = n - 2 - i1;
        int prev_row = n - i1 - 3;

        scalar_load(&X[curr_row * n + idx], cache_X_curr, batch_tasks * sizeof(double));
        scalar_load(&X[prev_row * n + idx], cache_X_prev, batch_tasks * sizeof(double));
        scalar_load(&A[prev_row * n + idx], cache_A, batch_tasks * sizeof(double));
        scalar_load(&B[curr_row * n + idx], cache_B, batch_tasks * sizeof(double));

        for (int bi = 0; bi < batch_tasks; ++bi) {
            cache_X_curr[bi] = (cache_X_curr[bi] - cache_X_prev[bi] * cache_A[bi]) / cache_B[bi];
        }

        scalar_store(cache_X_curr, &X[curr_row * n + idx], batch_tasks * sizeof(double));

        idx += batch_tasks;
    }

    scalar_free(cache_X_curr);
    scalar_free(cache_X_prev);
    scalar_free(cache_A);
    scalar_free(cache_B);
}