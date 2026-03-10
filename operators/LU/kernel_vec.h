
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void lu_kernel1_vec_qwen(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = n - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    double tmp = A[k * n + k];

    /* 分配向量缓冲区 */
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a) {
        return;
    }

    /* 常量向量初始化 */
    lvector double tmp_vec = (lvector double)vec_svbcast(tmp);

    /* 向量处理部分 */
    int j_vec_start = start_idx;
    int j_vec_end = end_idx - ((end_idx - start_idx) % SIMD_LEN);

    for (int j = j_vec_start; j < j_vec_end; j += SIMD_LEN) {
        /* 加载数据 */
        vector_load(&A[k * n + j], buf_a, VEC_BYTES);
        lvector double va = vec_ld(0, buf_a);

        /* 计算 va / tmp_vec */
        lvector double vres = vm_fdivd16(va, tmp_vec);

        /* 存储结果 */
        vec_st(vres, 0, buf_a);
        vector_store(buf_a, &A[k * n + j], VEC_BYTES);
    }

    /* 标量尾部处理 */
    for (int j = j_vec_end; j < end_idx; ++j) {
        A[k * n + j] = A[k * n + j] / tmp;
    }

    vector_free(buf_a);
}

__global__ void lu_kernel2_vec_qwen(int n, int k, double *A)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = (n - k - 1);
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = (tid < remainder)
                        ? tid * (elements_per_thread + 1) + k + 1
                        : remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread + k + 1;
    int end_idx = start_idx + elements_per_thread + (tid < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    /* 分配向量缓冲区 */
    lvector double *buf_row_i = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_row_k = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_ik = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_row_i || !buf_row_k || !buf_ik) {
        if (buf_row_i) vector_free(buf_row_i);
        if (buf_row_k) vector_free(buf_row_k);
        if (buf_ik) vector_free(buf_ik);
        return;
    }

    /* 常量向量初始化 */
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int i = k + 1; i < n; ++i) {
        double aik_scalar = A[i * n + k];
        lvector double aik_vec = (lvector double)vec_svbcast(aik_scalar);

        /* 向量处理部分 */
        int j_vec_start = start_idx;
        int j_vec_end = end_idx - ((end_idx - start_idx) % SIMD_LEN);

        for (int j = j_vec_start; j < j_vec_end; j += SIMD_LEN) {
            /* 加载行i和行k的数据 */
            vector_load(&A[i * n + j], buf_row_i, VEC_BYTES);
            vector_load(&A[k * n + j], buf_row_k, VEC_BYTES);

            lvector double vi = vec_ld(0, buf_row_i);
            lvector double vk = vec_ld(0, buf_row_k);

            /* 计算 vi - aik * vk */
            lvector double v_mul = vec_muli(aik_vec, vk);
            lvector double v_res = vec_mulb(vi, (lvector double)vec_svbcast(1.0), v_mul);

            /* 存储结果 */
            vec_st(v_res, 0, buf_row_i);
            vector_store(buf_row_i, &A[i * n + j], VEC_BYTES);
        }

        /* 标量尾部处理 */
        for (int j = j_vec_end; j < end_idx; ++j) {
            A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
        }
    }

    vector_free(buf_row_i);
    vector_free(buf_row_k);
    vector_free(buf_ik);
}