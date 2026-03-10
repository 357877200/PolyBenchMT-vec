#define ELEMS_PER_PART 1024

__global__ void fdtd2d_kernel1_cache_llm(int nx, int ny, int t, double *_fict_, double *ex, double *ey, double *hz)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_elements = nx * ny;
    int base = total_elements / gsz;
    int rem = total_elements % gsz;

    int start = tid * base + (tid < rem ? tid : rem);
    int end = start + base + (tid < rem ? 1 : 0);

    double* cache_ey = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_hz = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_hz_prev = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    if (tid == 0) {
        for (int j = 0; j < ny; j++) {
            ey[j] = _fict_[t];
        }
    }

    for (int idx = start; idx < end; ) {
        int batch_size = min(ELEMS_PER_PART, end - idx);
        int first_row = idx / ny;
        int first_col = idx % ny;
        
        int load_start = max(idx - ny, 0);
        int load_end = min(idx + batch_size, total_elements);
        int actual_batch = load_end - idx;

        scalar_load(&ey[idx], cache_ey, actual_batch * sizeof(double));
        scalar_load(&hz[idx], cache_hz, actual_batch * sizeof(double));
        scalar_load(&hz[load_start], cache_hz_prev, actual_batch * sizeof(double));

        for (int i = 0; i < actual_batch; i++) {
            int global_idx = idx + i;
            if (global_idx >= ny) {
                int prev_idx = global_idx - ny;
                if (prev_idx >= load_start && prev_idx < load_start + actual_batch) {
                    int cache_pos = prev_idx - load_start;
                    cache_ey[i] = cache_ey[i] - 0.5f * (cache_hz[i] - cache_hz_prev[cache_pos]);
                }
            }
        }

        scalar_store(cache_ey, &ey[idx], actual_batch * sizeof(double));
        idx += actual_batch;
    }

    scalar_free(cache_ey);
    scalar_free(cache_hz);
    scalar_free(cache_hz_prev);
}

__global__ void fdtd2d_kernel2_cache_llm(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_elements = nx * ny;
    int base = total_elements / gsz;
    int rem = total_elements % gsz;

    int start = tid * base + (tid < rem ? tid : rem);
    int end = start + base + (tid < rem ? 1 : 0);

    if (start / ny == 0) {
        start++;
    }

    double* cache_ex = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_hz = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_hz_prev = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; ) {
        int batch_size = min(ELEMS_PER_PART, end - idx);
        int first_col = idx % ny;
        
        int actual_batch = 0;
        for (int i = 0; i < batch_size; i++) {
            int global_idx = idx + i;
            int col = global_idx % ny;
            if (col > 0) {
                actual_batch++;
            } else {
                break;
            }
        }

        if (actual_batch == 0) {
            idx++;
            continue;
        }

        scalar_load(&ex[idx], cache_ex, actual_batch * sizeof(double));
        scalar_load(&hz[idx], cache_hz, actual_batch * sizeof(double));
        scalar_load(&hz[idx - 1], cache_hz_prev, actual_batch * sizeof(double));

        for (int i = 0; i < actual_batch; i++) {
            cache_ex[i] = cache_ex[i] - 0.5f * (cache_hz[i] - cache_hz_prev[i]);
        }

        scalar_store(cache_ex, &ex[idx], actual_batch * sizeof(double));
        idx += actual_batch;
    }

    scalar_free(cache_ex);
    scalar_free(cache_hz);
    scalar_free(cache_hz_prev);
}

__global__ void fdtd2d_kernel3_cache_llm(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_elements = nx * ny;
    int base = total_elements / gsz;
    int rem = total_elements % gsz;

    int start = tid * base + (tid < rem ? tid : rem);
    int end = start + base + (tid < rem ? 1 : 0);

    double* cache_hz = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_ex_curr = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_ex_next = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_ey_curr = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_ey_next = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start; idx < end; ) {
        int batch_size = min(ELEMS_PER_PART, end - idx);
        int first_row = idx / ny;
        int first_col = idx % ny;
        
        int actual_batch = 0;
        for (int i = 0; i < batch_size; i++) {
            int global_idx = idx + i;
            int row = global_idx / ny;
            int col = global_idx % ny;
            if (row < nx - 1 && col < ny - 1) {
                actual_batch++;
            } else {
                break;
            }
        }

        if (actual_batch == 0) {
            idx++;
            continue;
        }

        scalar_load(&hz[idx], cache_hz, actual_batch * sizeof(double));
        
        for (int i = 0; i < actual_batch; i++) {
            int global_idx = idx + i;
            int row = global_idx / ny;
            int col = global_idx % ny;
            
            cache_ex_curr[i] = ex[global_idx];
            cache_ex_next[i] = ex[global_idx + 1];
            cache_ey_curr[i] = ey[global_idx];
            cache_ey_next[i] = ey[global_idx + ny];
        }

        for (int i = 0; i < actual_batch; i++) {
            cache_hz[i] = cache_hz[i] - 0.7f * (cache_ex_next[i] - cache_ex_curr[i] + cache_ey_next[i] - cache_ey_curr[i]);
        }

        scalar_store(cache_hz, &hz[idx], actual_batch * sizeof(double));
        idx += actual_batch;
    }

    scalar_free(cache_hz);
    scalar_free(cache_ex_curr);
    scalar_free(cache_ex_next);
    scalar_free(cache_ey_curr);
    scalar_free(cache_ey_next);
}