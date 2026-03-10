
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void deriche_kernel4_vec_qwen(int w, int h, double a5, double a6, double b1, double b2,
                            double *imgOut, double *y1)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int j_start = (h * thread_id) / group_size;
    int j_end   = (h * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) j_end = h;

    for (int j = j_start; j < j_end; j++) {
        for (int i = 0; i < w; i++) {
            int idx = i*h + j;
            y1[idx] = a5 * imgOut[idx] + a6 * (i == 0 ? 0.0 : imgOut[(i-1)*h + j]) + b1 * (i == 0 ? 0.0 : y1[(i-1)*h + j]) + b2 * (i < 2 ? 0.0 : y1[(i-2)*h + j]);
        }
    }
}

__global__ void deriche_kernel5_vec_qwen(int w, int h, double a7, double a8,
                             double b1, double b2,
                             double *imgOut, double *y2)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int j_start = (h * thread_id) / group_size;
    int j_end   = (h * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) j_end = h;

    for (int j = j_start; j < j_end; j++) {
        for (int i = w-1; i >= 0; i--) {
            int idx = i*h + j;
            y2[idx] = a7 * (i == w - 1 ? 0.0 : imgOut[(i+1)*h + j]) + a8 * (i >= w - 2 ? 0.0 : imgOut[(i+2)*h + j]) + b1 * (i == w - 1 ? 0.0 : y2[(i+1)*h + j]) + b2 * (i >= w - 2 ? 0.0 : y2[(i+2)*h + j]);
        }
    }
}

__global__ void deriche_kernel6_vec_qwen(int w, int h, double c2,
                             double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = w * h;
    int elems_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elems_per_thread +
                    (thread_id < remainder ? thread_id : remainder);
    int end_idx   = (thread_id + 1) * elems_per_thread +
                    ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);

    lvector double *buf_y1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y1 || !buf_y2 || !buf_out) {
        if (buf_y1) vector_free(buf_y1);
        if (buf_y2) vector_free(buf_y2);
        if (buf_out) vector_free(buf_out);
        return;
    }

    lvector double c2_vec = (lvector double)vec_svbcast(c2);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    int idx;
    for (idx = start_idx; idx + SIMD_LEN <= end_idx; idx += SIMD_LEN) {
        vector_load(&y1[idx], buf_y1, VEC_BYTES);
        vector_load(&y2[idx], buf_y2, VEC_BYTES);

        lvector double v_y1 = vec_ld(0, buf_y1);
        lvector double v_y2 = vec_ld(0, buf_y2);
        lvector double v_sum = vec_mula(v_y1, one_vec, v_y2);
        lvector double v_out = vec_muli(c2_vec, v_sum);

        vec_st(v_out, 0, buf_out);
        vector_store(buf_out, &imgOut[idx], VEC_BYTES);
    }

    for (; idx < end_idx; ++idx) {
        imgOut[idx] = c2 * (y1[idx] + y2[idx]);
    }

    vector_free(buf_y1);
    vector_free(buf_y2);
    vector_free(buf_out);
}