
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void gramschmidt_kernel1_vec_qwen(int ni, int nj, int k, double *a, double *r, double *q)
{
    int thread_id = get_thread_id();

    if (thread_id == 0) {
        lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
        if (!buf_a) {
            return;
        }

        double nrm = 0.0;
        int i;
        for (i = 0; i + SIMD_LEN <= ni; i += SIMD_LEN) {
            vector_load(&a[i * nj + k], buf_a, VEC_BYTES);
            lvector double va = vec_ld(0, buf_a);
            lvector double vmul = vec_muli(va, va);
            nrm += sum_f64(vmul);
        }

        for (; i < ni; ++i) {
            double val = a[i * nj + k];
            nrm += val * val;
        }

        r[k * nj + k] = sqrt(nrm);

        vector_free(buf_a);
    }
}

__global__ void gramschmidt_kernel2_vec_qwen(int ni, int nj, int k, double *a, double *r, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = start_idx + elements_per_thread + (thread_id < remainder ? 1 : 0);

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_q = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_q) {
        if (buf_a) vector_free(buf_a);
        if (buf_q) vector_free(buf_q);
        return;
    }

    lvector double one_vec = (lvector double)vec_svbcast(1.0);
    lvector double rkk_vec = (lvector double)vec_svbcast(r[k * nj + k]);
    lvector double eps_vec = (lvector double)vec_svbcast(1e-6);

    int i;
    for (i = start_idx; i + SIMD_LEN <= end_idx; i += SIMD_LEN) {
        vector_load(&a[i * nj + k], buf_a, VEC_BYTES);
        lvector double va = vec_ld(0, buf_a);
        lvector double vdiv = vm_fdivd16(va, vec_mula(rkk_vec, one_vec, eps_vec));
        vec_st(vdiv, 0, buf_q);
        vector_store(buf_q, &q[i * nj + k], VEC_BYTES);
    }

    for (; i < end_idx; ++i) {
        q[i * nj + k] = a[i * nj + k] / (r[k * nj + k] + 1e-6);
    }

    vector_free(buf_a);
    vector_free(buf_q);
}

__global__ void gramschmidt_kernel3_vec_qwen(int ni, int nj, int k, double *a, double *r, double *q)
{
    int thread_id = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = nj - k - 1;
    if (total_elements <= 0) {
        return;
    }
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = start_idx + elements_per_thread + (thread_id < remainder ? 1 : 0);
    if (start_idx >= end_idx) {
        return;
    }

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_q = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_r = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_q || !buf_r) {
        if (buf_a) vector_free(buf_a);
        if (buf_q) vector_free(buf_q);
        if (buf_r) vector_free(buf_r);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int j = start_idx; j < end_idx; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, end_idx);
        if (j + SIMD_LEN <= end_idx) {
            vec_st(zero_vec, 0, buf_r);
            vector_store(buf_r, &r[k * nj + j], VEC_BYTES);
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                r[k * nj + jj] = 0.0;
            }
        }
    }

    for (int i = 0; i < ni; ++i) {
        lvector double qik_vec = (lvector double)vec_svbcast(q[i * nj + k]);
        for (int j = start_idx; j + SIMD_LEN <= end_idx; j += SIMD_LEN) {
            vector_load(&r[k * nj + j], buf_r, VEC_BYTES);
            vector_load(&a[i * nj + j], buf_a, VEC_BYTES);
            lvector double vr = vec_ld(0, buf_r);
            lvector double va = vec_ld(0, buf_a);
            lvector double vmul = vec_muli(qik_vec, va);
            lvector double vnew = vec_mula(vr, one_vec, vmul);
            vec_st(vnew, 0, buf_r);
            vector_store(buf_r, &r[k * nj + j], VEC_BYTES);
        }

        for (int j = (end_idx - (end_idx % SIMD_LEN)); j < end_idx; ++j) {
            r[k * nj + j] += q[i * nj + k] * a[i * nj + j];
        }
    }

    for (int i = 0; i < ni; ++i) {
        lvector double qik_vec = (lvector double)vec_svbcast(q[i * nj + k]);
        for (int j = start_idx; j + SIMD_LEN <= end_idx; j += SIMD_LEN) {
            vector_load(&r[k * nj + j], buf_r, VEC_BYTES);
            vector_load(&a[i * nj + j], buf_a, VEC_BYTES);
            lvector double vr = vec_ld(0, buf_r);
            lvector double va = vec_ld(0, buf_a);
            lvector double vmul = vec_muli(qik_vec, vr);
            lvector double vnew = vec_mulb(va, one_vec, vmul);
            vec_st(vnew, 0, buf_a);
            vector_store(buf_a, &a[i * nj + j], VEC_BYTES);
        }

        for (int j = (end_idx - (end_idx % SIMD_LEN)); j < end_idx; ++j) {
            a[i * nj + j] -= q[i * nj + k] * r[k * nj + j];
        }
    }

    vector_free(buf_a);
    vector_free(buf_q);
    vector_free(buf_r);
}