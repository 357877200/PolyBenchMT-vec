
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void fdtd2d_kernel1_vec_qwen(int nx, int ny, int t, double *_fict_, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;
    int per_thread = (total_elements + num_threads - 1) / num_threads;
    int start = thread_id * per_thread;
    int end = min(start + per_thread, total_elements);

    if (thread_id == 0) {
        for (int j = 0; j < ny; j++) {
            ey[j] = _fict_[t];
        }
        start = ny;
    }

    lvector double *buf_ey = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_hz = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_hz_prev = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_ey || !buf_hz || !buf_hz_prev) {
        if (buf_ey) vector_free(buf_ey);
        if (buf_hz) vector_free(buf_hz);
        if (buf_hz_prev) vector_free(buf_hz_prev);
        return;
    }

    lvector double half_vec = (lvector double)vec_svbcast(0.5);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end_idx = min(idx + SIMD_LEN, end);
        
        if (idx + SIMD_LEN <= end && idx >= ny) {
            vector_load(&ey[idx], buf_ey, VEC_BYTES);
            vector_load(&hz[idx], buf_hz, VEC_BYTES);
            vector_load(&hz[idx - ny], buf_hz_prev, VEC_BYTES);

            lvector double v_ey = vec_ld(0, buf_ey);
            lvector double v_hz = vec_ld(0, buf_hz);
            lvector double v_hz_prev = vec_ld(0, buf_hz_prev);

            lvector double v_diff = vec_mulb(v_hz, one_vec, v_hz_prev);
            lvector double v_term = vec_muli(half_vec, v_diff);
            lvector double v_res = vec_mulb(v_ey, one_vec, v_term);

            vec_st(v_res, 0, buf_ey);
            vector_store(buf_ey, &ey[idx], VEC_BYTES);
        } else {
            for (int jj = idx; jj < vec_end_idx; ++jj) {
                if (jj >= ny) {
                    ey[jj] = ey[jj] - 0.5 * (hz[jj] - hz[jj - ny]);
                }
            }
        }
    }

    vector_free(buf_ey);
    vector_free(buf_hz);
    vector_free(buf_hz_prev);
}

__global__ void fdtd2d_kernel2_vec_qwen(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;
    int per_thread = (total_elements + num_threads - 1) / num_threads;
    int start = thread_id * per_thread;
    int end = min(start + per_thread, total_elements);
    
    if (start / ny == 0) {
        start++;
    }

    lvector double *buf_ex = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_hz = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_hz_prev = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_ex || !buf_hz || !buf_hz_prev) {
        if (buf_ex) vector_free(buf_ex);
        if (buf_hz) vector_free(buf_hz);
        if (buf_hz_prev) vector_free(buf_hz_prev);
        return;
    }

    lvector double half_vec = (lvector double)vec_svbcast(0.5);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end_idx = min(idx + SIMD_LEN, end);
        
        if (idx + SIMD_LEN <= end) {
            vector_load(&ex[idx], buf_ex, VEC_BYTES);
            vector_load(&hz[idx], buf_hz, VEC_BYTES);
            vector_load(&hz[idx - 1], buf_hz_prev, VEC_BYTES);

            lvector double v_ex = vec_ld(0, buf_ex);
            lvector double v_hz = vec_ld(0, buf_hz);
            lvector double v_hz_prev = vec_ld(0, buf_hz_prev);

            lvector double v_diff = vec_mulb(v_hz, one_vec, v_hz_prev);
            lvector double v_term = vec_muli(half_vec, v_diff);
            lvector double v_res = vec_mulb(v_ex, one_vec, v_term);

            vec_st(v_res, 0, buf_ex);
            vector_store(buf_ex, &ex[idx], VEC_BYTES);
        } else {
            for (int jj = idx; jj < vec_end_idx; ++jj) {
                int j = jj % ny;
                if (j > 0) {
                    ex[jj] = ex[jj] - 0.5 * (hz[jj] - hz[jj - 1]);
                }
            }
        }
    }

    vector_free(buf_ex);
    vector_free(buf_hz);
    vector_free(buf_hz_prev);
}

__global__ void fdtd2d_kernel3_vec_qwen(int nx, int ny, int t, double *ex, double *ey, double *hz)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();

    int total_elements = nx * ny;
    int per_thread = (total_elements + num_threads - 1) / num_threads;
    int start = thread_id * per_thread;
    int end = min(start + per_thread, total_elements);

    lvector double *buf_hz = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_ex = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_ex_next = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_ey = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_ey_next = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_hz || !buf_ex || !buf_ex_next || !buf_ey || !buf_ey_next) {
        if (buf_hz) vector_free(buf_hz);
        if (buf_ex) vector_free(buf_ex);
        if (buf_ex_next) vector_free(buf_ex_next);
        if (buf_ey) vector_free(buf_ey);
        if (buf_ey_next) vector_free(buf_ey_next);
        return;
    }

    lvector double point7_vec = (lvector double)vec_svbcast(0.7);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int idx = start; idx < end; idx += SIMD_LEN) {
        int vec_end_idx = min(idx + SIMD_LEN, end);
        
        if (idx + SIMD_LEN <= end) {
            int i = idx / ny;
            int j = idx % ny;
            
            if (i < nx - 1 && j < ny - 1 && j + SIMD_LEN <= ny - 1) {
                vector_load(&hz[i * ny + j], buf_hz, VEC_BYTES);
                vector_load(&ex[i * ny + j], buf_ex, VEC_BYTES);
                vector_load(&ex[i * ny + (j + 1)], buf_ex_next, VEC_BYTES);
                vector_load(&ey[i * ny + j], buf_ey, VEC_BYTES);
                vector_load(&ey[(i + 1) * ny + j], buf_ey_next, VEC_BYTES);

                lvector double v_hz = vec_ld(0, buf_hz);
                lvector double v_ex = vec_ld(0, buf_ex);
                lvector double v_ex_next = vec_ld(0, buf_ex_next);
                lvector double v_ey = vec_ld(0, buf_ey);
                lvector double v_ey_next = vec_ld(0, buf_ey_next);

                lvector double v_ex_diff = vec_mulb(v_ex_next, one_vec, v_ex);
                lvector double v_ey_diff = vec_mulb(v_ey_next, one_vec, v_ey);
                lvector double v_sum = vec_mula(v_ex_diff, one_vec, v_ey_diff);
                lvector double v_term = vec_muli(point7_vec, v_sum);
                lvector double v_res = vec_mulb(v_hz, one_vec, v_term);

                vec_st(v_res, 0, buf_hz);
                vector_store(buf_hz, &hz[i * ny + j], VEC_BYTES);
            } else {
                for (int jj = idx; jj < vec_end_idx; ++jj) {
                    int ii = jj / ny;
                    int jjj = jj % ny;
                    if (ii < nx - 1 && jjj < ny - 1) {
                        hz[ii * ny + jjj] = hz[ii * ny + jjj] - 0.7 * (ex[ii * ny + (jjj + 1)] - ex[ii * ny + jjj] + ey[(ii + 1) * ny + jjj] - ey[ii * ny + jjj]);
                    }
                }
            }
        } else {
            for (int jj = idx; jj < vec_end_idx; ++jj) {
                int ii = jj / ny;
                int jjj = jj % ny;
                if (ii < nx - 1 && jjj < ny - 1) {
                    hz[ii * ny + jjj] = hz[ii * ny + jjj] - 0.7 * (ex[ii * ny + (jjj + 1)] - ex[ii * ny + jjj] + ey[(ii + 1) * ny + jjj] - ey[ii * ny + jjj]);
                }
            }
        }
    }

    vector_free(buf_hz);
    vector_free(buf_ex);
    vector_free(buf_ex_next);
    vector_free(buf_ey);
    vector_free(buf_ey_next);
}