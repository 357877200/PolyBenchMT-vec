
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void jacobi1D_kernel1_vec_qwen(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    int end_idx;
    if (thread_id < remainder) {
        end_idx = start_idx + elements_per_thread;
    } else {
        end_idx = start_idx + elements_per_thread - 1;
    }

    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_b) {
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        return;
    }

    lvector double coeff_vec = (lvector double)vec_svbcast(0.33333);

    for (int i = start_idx; i <= end_idx; i += SIMD_LEN) {
        if (i + SIMD_LEN <= end_idx) {
            vector_load(&A[i - 1], buf_a, VEC_BYTES);
            vector_load(&A[i], buf_a + 1, VEC_BYTES);
            vector_load(&A[i + 1], buf_a + 2, VEC_BYTES);

            lvector double va_prev = vec_ld(0, buf_a);
            lvector double va_curr = vec_ld(0, buf_a + 1);
            lvector double va_next = vec_ld(0, buf_a + 2);

            lvector double vsum = vec_mula(va_prev, (lvector double)vec_svbcast(1.0), va_curr);
            vsum = vec_mula(vsum, (lvector double)vec_svbcast(1.0), va_next);
            lvector double vres = vec_muli(vsum, coeff_vec);

            vec_st(vres, 0, buf_b);
            vector_store(buf_b, &B[i], VEC_BYTES);
        } else {
            for (int ii = i; ii <= end_idx; ++ii) {
                B[ii] = 0.33333f * (A[ii - 1] + A[ii] + A[ii + 1]);
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_b);
}

__global__ void jacobi1D_kernel2_vec_qwen(int n, double *A, double *B)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int elements_per_thread = (n - 2) / total_threads;
    int remainder = (n - 2) % total_threads;

    int start_idx = 1;
    if (thread_id < remainder) {
        start_idx += thread_id * (elements_per_thread + 1);
    } else {
        start_idx += remainder * (elements_per_thread + 1) + (thread_id - remainder) * elements_per_thread;
    }

    int end_idx = (thread_id < remainder) ? start_idx + elements_per_thread : start_idx + elements_per_thread - 1;
    end_idx = (end_idx < (n - 1)) ? end_idx : (n - 1);

    for (int i = start_idx; i <= end_idx; ++i) {
        A[i] = B[i];
    }
}