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