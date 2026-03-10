
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void deriche_kernel1_vec_qwen(int w, int h, double a1, double a2, double b1, double b2,
                            double *imgIn, double *y1)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int i_start = (w * thread_id) / group_size;
    int i_end   = (w * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) i_end = w;

    lvector double *buf_in = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_in || !buf_out) {
        if (buf_in) vector_free(buf_in);
        if (buf_out) vector_free(buf_out);
        return;
    }

    lvector double a1_vec = (lvector double)vec_svbcast(a1);
    lvector double a2_vec = (lvector double)vec_svbcast(a2);
    lvector double b1_vec = (lvector double)vec_svbcast(b1);
    lvector double b2_vec = (lvector double)vec_svbcast(b2);

    for (int i = i_start; i < i_end; i++) {
        double ym1_scalar = 0.0;
        double ym2_scalar = 0.0;
        double xm1_scalar = 0.0;

        for (int j = 0; j < h; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, h);

            if (j + SIMD_LEN <= h) {
                vector_load(&imgIn[i*h + j], buf_in, VEC_BYTES);
                lvector double v_in = vec_ld(0, buf_in);

                lvector double v_xm1 = (lvector double)vec_svbcast(xm1_scalar);
                lvector double v_ym1 = (lvector double)vec_svbcast(ym1_scalar);
                lvector double v_ym2 = (lvector double)vec_svbcast(ym2_scalar);

                lvector double v_term1 = vec_muli(a1_vec, v_in);
                lvector double v_term2 = vec_muli(a2_vec, v_xm1);
                lvector double v_term3 = vec_muli(b1_vec, v_ym1);
                lvector double v_term4 = vec_muli(b2_vec, v_ym2);

                lvector double v_res = vec_mula(v_term1, v_term2, v_term3);
                v_res = vec_mula(v_res, v_term4, (lvector double)vec_svbcast(0.0));

                vec_st(v_res, 0, buf_out);
                vector_store(buf_out, &y1[i*h + j], VEC_BYTES);

                xm1_scalar = imgIn[i*h + j + SIMD_LEN - 1];
                ym2_scalar = ym1_scalar;
                ym1_scalar = y1[i*h + j + SIMD_LEN - 1];
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    int idx = i*h + jj;
                    y1[idx] = a1 * imgIn[idx] + a2 * xm1_scalar + b1 * ym1_scalar + b2 * ym2_scalar;
                    xm1_scalar = imgIn[idx];
                    ym2_scalar = ym1_scalar;
                    ym1_scalar = y1[idx];
                }
            }
        }
    }

    vector_free(buf_in);
    vector_free(buf_out);
}

__global__ void deriche_kernel2_vec_qwen(int w, int h, double a3, double a4, double b1, double b2,
                            double *imgIn, double *y2)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int i_start = (w * thread_id) / group_size;
    int i_end   = (w * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) i_end = w;

    lvector double *buf_in = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_in || !buf_out) {
        if (buf_in) vector_free(buf_in);
        if (buf_out) vector_free(buf_out);
        return;
    }

    lvector double a3_vec = (lvector double)vec_svbcast(a3);
    lvector double a4_vec = (lvector double)vec_svbcast(a4);
    lvector double b1_vec = (lvector double)vec_svbcast(b1);
    lvector double b2_vec = (lvector double)vec_svbcast(b2);

    for (int i = i_start; i < i_end; i++) {
        double yp1_scalar = 0.0, yp2_scalar = 0.0;
        double xp1_scalar = 0.0, xp2_scalar = 0.0;

        for (int j = h - SIMD_LEN; j >= -SIMD_LEN; j -= SIMD_LEN) {
            int vec_start_j = max(j, 0);
            int vec_end_j = j + SIMD_LEN;
            if (vec_start_j >= h) break;

            if (j >= 0 && vec_end_j <= h) {
                vector_load(&imgIn[i*h + vec_start_j], buf_in, VEC_BYTES);
                lvector double v_in = vec_ld(0, buf_in);

                lvector double v_xp1 = (lvector double)vec_svbcast(xp1_scalar);
                lvector double v_xp2 = (lvector double)vec_svbcast(xp2_scalar);
                lvector double v_yp1 = (lvector double)vec_svbcast(yp1_scalar);
                lvector double v_yp2 = (lvector double)vec_svbcast(yp2_scalar);

                lvector double v_term1 = vec_muli(a3_vec, v_xp1);
                lvector double v_term2 = vec_muli(a4_vec, v_xp2);
                lvector double v_term3 = vec_muli(b1_vec, v_yp1);
                lvector double v_term4 = vec_muli(b2_vec, v_yp2);

                lvector double v_res = vec_mula(v_term1, v_term2, v_term3);
                v_res = vec_mula(v_res, v_term4, (lvector double)vec_svbcast(0.0));

                vec_st(v_res, 0, buf_out);
                vector_store(buf_out, &y2[i*h + vec_start_j], VEC_BYTES);

                xp2_scalar = xp1_scalar;
                xp1_scalar = imgIn[i*h + vec_start_j];
                yp2_scalar = yp1_scalar;
                yp1_scalar = y2[i*h + vec_start_j + SIMD_LEN - 1];
            } else {
                for (int jj = h - 1; jj >= vec_start_j; --jj) {
                    int idx = i*h + jj;
                    y2[idx] = a3 * xp1_scalar + a4 * xp2_scalar + b1 * yp1_scalar + b2 * yp2_scalar;
                    xp2_scalar = xp1_scalar;
                    xp1_scalar = imgIn[idx];
                    yp2_scalar = yp1_scalar;
                    yp1_scalar = y2[idx];
                }
                break;
            }
        }
    }

    vector_free(buf_in);
    vector_free(buf_out);
}

__global__ void deriche_kernel3_vec_qwen(int w, int h, double c1,
                            double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = w * h;
    int elems_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elems_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx   = (thread_id + 1) * elems_per_thread + ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);

    lvector double *buf_y1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y1 || !buf_y2 || !buf_out) {
        if (buf_y1) vector_free(buf_y1);
        if (buf_y2) vector_free(buf_y2);
        if (buf_out) vector_free(buf_out);
        return;
    }

    lvector double c1_vec = (lvector double)vec_svbcast(c1);

    for (int idx = start_idx; idx < end_idx; idx += SIMD_LEN) {
        int vec_end_idx = min(idx + SIMD_LEN, end_idx);

        if (idx + SIMD_LEN <= end_idx) {
            vector_load(&y1[idx], buf_y1, VEC_BYTES);
            vector_load(&y2[idx], buf_y2, VEC_BYTES);

            lvector double v_y1 = vec_ld(0, buf_y1);
            lvector double v_y2 = vec_ld(0, buf_y2);

            lvector double v_sum = vec_mula(v_y1, (lvector double)vec_svbcast(1.0), v_y2);
            lvector double v_res = vec_muli(c1_vec, v_sum);

            vec_st(v_res, 0, buf_out);
            vector_store(buf_out, &imgOut[idx], VEC_BYTES);
        } else {
            for (int jj = idx; jj < vec_end_idx; ++jj) {
                imgOut[jj] = c1 * (y1[jj] + y2[jj]);
            }
        }
    }

    vector_free(buf_y1);
    vector_free(buf_y2);
    vector_free(buf_out);
}