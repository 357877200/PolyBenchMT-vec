
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void mvt_kernel1_vec_qwen(int n, double *a, double *x1, double *y_1)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_y) {
        if (buf_a) vector_free(buf_a);
        if (buf_y) vector_free(buf_y);
        return;
    }

    for (int i = start_idx; i < end_idx; ++i) {
        double sum_val = 0.0;
        for (int j = 0; j < n; j += SIMD_LEN) {
            if (j + SIMD_LEN <= n) {
                vector_load(&a[i * n + j], buf_a, VEC_BYTES);
                vector_load(&y_1[j], buf_y, VEC_BYTES);
                
                lvector double va = vec_ld(0, buf_a);
                lvector double vy = vec_ld(0, buf_y);
                lvector double vprod = vec_muli(va, vy);
                
                double partial_sum = sum_f64(vprod);
                sum_val += partial_sum;
            } else {
                for (int jj = j; jj < n; ++jj) {
                    sum_val += a[i * n + jj] * y_1[jj];
                }
            }
        }
        x1[i] += sum_val;
    }

    vector_free(buf_a);
    vector_free(buf_y);
}

__global__ void mvt_kernel2_vec_qwen(int n, double *a, double *x2, double *y_2)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder) ? tid * (elements_per_thread + 1)
                                      : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_x) {
        if (buf_a) vector_free(buf_a);
        if (buf_x) vector_free(buf_x);
        return;
    }

    for (int j = 0; j < n; ++j) {
        lvector double yj_vec = (lvector double)vec_svbcast(y_2[j]);
        int i = start_idx;
        
        for (i = start_idx; i <= end_idx - SIMD_LEN; i += SIMD_LEN) {
            vector_load(&a[j * n + i], buf_a, VEC_BYTES);
            vector_load(&x2[i], buf_x, VEC_BYTES);
            
            lvector double va = vec_ld(0, buf_a);
            lvector double vx = vec_ld(0, buf_x);
            lvector double vprod = vec_muli(va, yj_vec);
            lvector double vnew = vec_mula(vx, (lvector double)vec_svbcast(1.0), vprod);
            
            vec_st(vnew, 0, buf_x);
            vector_store(buf_x, &x2[i], VEC_BYTES);
        }
        
        for (; i < end_idx; ++i) {
            x2[i] += a[j * n + i] * y_2[j];
        }
    }

    vector_free(buf_a);
    vector_free(buf_x);
}