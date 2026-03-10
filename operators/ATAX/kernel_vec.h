
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void atax_kernel1_vec_qwen(int nx, int ny, double *A, double *x, double *tmp)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nx;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_tmp = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_x || !buf_tmp) {
        if (buf_a) vector_free(buf_a);
        if (buf_x) vector_free(buf_x);
        if (buf_tmp) vector_free(buf_tmp);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = start_idx; i < end_idx; i++) {
        int j = 0;
        lvector double sum_vec = zero_vec;

        for (j = 0; j <= ny - SIMD_LEN; j += SIMD_LEN) {
            vector_load(&A[i * ny + j], buf_a, VEC_BYTES);
            vector_load(&x[j], buf_x, VEC_BYTES);

            lvector double va = vec_ld(0, buf_a);
            lvector double vx = vec_ld(0, buf_x);

            lvector double prod = vec_muli(va, vx);
            sum_vec = vec_mula(sum_vec, zero_vec, prod);
        }

        double sum_val = sum_f64(sum_vec);

        for (; j < ny; j++) {
            sum_val += A[i * ny + j] * x[j];
        }

        tmp[i] = sum_val;
    }

    vector_free(buf_a);
    vector_free(buf_x);
    vector_free(buf_tmp);
}

__global__ void atax_kernel2_vec_qwen(int nx, int ny, double *A, double *y, double *tmp)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ny;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread;
    int end_idx = (thread_id + 1) * elements_per_thread;

    if (thread_id < remainder) {
        start_idx += thread_id;
        end_idx = start_idx + elements_per_thread + 1;
    } else {
        start_idx += remainder;
        end_idx = start_idx + elements_per_thread;
    }

    for (int j = start_idx; j < end_idx; j++) {
        y[j] = 0;
    }

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_tmp = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_tmp || !buf_y) {
        if (buf_a) vector_free(buf_a);
        if (buf_tmp) vector_free(buf_tmp);
        if (buf_y) vector_free(buf_y);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = 0; i < nx; i++) {
        int j = 0;
        for (j = start_idx; j <= end_idx - SIMD_LEN && j <= ny - SIMD_LEN; j += SIMD_LEN) {
            vector_load(&A[i * ny + j], buf_a, VEC_BYTES);
            vector_load(&y[j], buf_y, VEC_BYTES);

            lvector double va = vec_ld(0, buf_a);
            lvector double vy = vec_ld(0, buf_y);

            double tmp_val = tmp[i];
            lvector double tmp_vec = (lvector double)vec_svbcast(tmp_val);

            lvector double prod = vec_muli(va, tmp_vec);
            lvector double sum = vec_mula(vy, zero_vec, prod);

            vec_st(sum, 0, buf_y);
            vector_store(buf_y, &y[j], VEC_BYTES);
        }

        for (; j < end_idx && j < ny; j++) {
            y[j] += A[i * ny + j] * tmp[i];
        }
    }

    vector_free(buf_a);
    vector_free(buf_tmp);
    vector_free(buf_y);
}