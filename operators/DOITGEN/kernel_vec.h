
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void doitgen_kernel1_vec_qwen(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    lvector double *buf_sum = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c4 = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_sum || !buf_a || !buf_c4) {
        if (buf_sum) vector_free(buf_sum);
        if (buf_a) vector_free(buf_a);
        if (buf_c4) vector_free(buf_c4);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end_i = min(i + SIMD_LEN, end_idx);
        if (i + SIMD_LEN <= end_idx) {
            vector_load(&sum[r * (nq * np) + i], buf_sum, VEC_BYTES);
            lvector double vsum = vec_ld(0, buf_sum);
            vsum = zero_vec;
            vec_st(vsum, 0, buf_sum);
            vector_store(buf_sum, &sum[r * (nq * np) + i], VEC_BYTES);
        } else {
            for (int ii = i; ii < vec_end_i; ++ii) {
                sum[r * (nq * np) + ii] = (double)0.0;
            }
        }
    }

    for (int s = 0; s < np; ++s) {
        for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
            int vec_end_i = min(i + SIMD_LEN, end_idx);
            if (i + SIMD_LEN <= end_idx) {
                vector_load(&sum[r * (nq * np) + i], buf_sum, VEC_BYTES);
                lvector double vsum = vec_ld(0, buf_sum);

                lvector double vtemp = zero_vec;
                for (int ii = 0; ii < SIMD_LEN; ++ii) {
                    int p = (i + ii) % np;
                    int q = (i + ii) / np;
                    double a_val = A[r * (nq * np) + q * np + s];
                    double c4_val = C4[s * np + p];
                    vtemp = vec_mula(vtemp, (lvector double)vec_svbcast(1.0), (lvector double)vec_svbcast(a_val * c4_val));
                }

                vsum = vec_mula(vsum, (lvector double)vec_svbcast(1.0), vtemp);
                vec_st(vsum, 0, buf_sum);
                vector_store(buf_sum, &sum[r * (nq * np) + i], VEC_BYTES);
            } else {
                for (int ii = i; ii < vec_end_i; ++ii) {
                    int p = ii % np;
                    int q = ii / np;
                    sum[r * (nq * np) + ii] += A[r * (nq * np) + q * np + s] * C4[s * np + p];
                }
            }
        }
    }

    vector_free(buf_sum);
    vector_free(buf_a);
    vector_free(buf_c4);
}

__global__ void doitgen_kernel2_vec_qwen(int r, int nq, int np, double *sum, double *A, double *C4)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = np * nq;
    int elements_per_thread = total_elements / group_size;
    int extra_elements = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + min(thread_id, extra_elements);
    int end_idx = start_idx + elements_per_thread + (thread_id < extra_elements ? 1 : 0);

    lvector double *buf_sum = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_sum || !buf_a) {
        if (buf_sum) vector_free(buf_sum);
        if (buf_a) vector_free(buf_a);
        return;
    }

    for (int i = start_idx; i < end_idx; i += SIMD_LEN) {
        int vec_end_i = min(i + SIMD_LEN, end_idx);
        if (i + SIMD_LEN <= end_idx) {
            vector_load(&sum[r * (nq * np) + i], buf_sum, VEC_BYTES);
            lvector double vsum = vec_ld(0, buf_sum);
            vec_st(vsum, 0, buf_a);
            vector_store(buf_a, &A[r * (nq * np) + i], VEC_BYTES);
        } else {
            for (int ii = i; ii < vec_end_i; ++ii) {
                A[r * (nq * np) + ii] = sum[r * (nq * np) + ii];
            }
        }
    }

    vector_free(buf_sum);
    vector_free(buf_a);
}