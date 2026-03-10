#define ELEMS_PER_PART 1024

__global__ void trmm_kernel_cache_llm(int m, int n, double alpha, double *A, double *B)
{
    int tid         = get_thread_id();
    int num_threads = get_group_size();

    // 列并行分配
    int work_per_thread = (n + num_threads - 1) / num_threads;
    int start_j = tid * work_per_thread;
    int end_j   = min(start_j + work_per_thread, n);

    // 分配缓存
    double* cache_A_col = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B_row = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_B_out = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int i = 0; i < m; i++) {
        for (int j_start = start_j; j_start < end_j; ) {
            int batch_size = min(ELEMS_PER_PART, end_j - j_start);
            
            // 加载B的当前行块到缓存
            scalar_load(&B[i * n + j_start], cache_B_out, batch_size * sizeof(double));
            
            // 对每个k进行更新
            for (int k = i + 1; k < m; k++) {
                // 加载A的列数据
                scalar_load(&A[k * m + i], cache_A_col, sizeof(double));
                
                // 加载B的k行数据
                scalar_load(&B[k * n + j_start], cache_B_row, batch_size * sizeof(double));
                
                // 向量乘加操作
                for (int bj = 0; bj < batch_size; bj++) {
                    cache_B_out[bj] += cache_A_col[0] * cache_B_row[bj];
                }
            }
            
            // 乘以alpha
            for (int bj = 0; bj < batch_size; bj++) {
                cache_B_out[bj] *= alpha;
            }
            
            // 写回结果
            scalar_store(cache_B_out, &B[i * n + j_start], batch_size * sizeof(double));
            
            j_start += batch_size;
        }
    }

    scalar_free(cache_A_col);
    scalar_free(cache_B_row);
    scalar_free(cache_B_out);
}