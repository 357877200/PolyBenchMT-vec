
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void mm2_kernel1_vec_qwen(int ni, int nj, int nk, double alpha, double *tmp,
                            double *A, double *B)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
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

    lvector double *buf_tmp = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_b = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_tmp || !buf_a || !buf_b || !buf_res) {
        if (buf_tmp) vector_free(buf_tmp);
        if (buf_a) vector_free(buf_a);
        if (buf_b) vector_free(buf_b);
        if (buf_res) vector_free(buf_res);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);
    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);

    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / nj;
        int j = idx % nj;
        tmp[i * nj + j] = 0;
    }

    for (int k = 0; k < nk; k++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / nj;
            int j = idx % nj;

            if (j + SIMD_LEN <= nj) {
                vector_load(&tmp[i*nj + j], buf_tmp, VEC_BYTES);
                vector_load(&A[i*nk + k], buf_a, sizeof(double));
                vector_load(&B[k*nj + j], buf_b, VEC_BYTES);

                lvector double vtmp = vec_ld(0, buf_tmp);
                lvector double va = vec_ld(0, buf_a);
                lvector double vb = vec_ld(0, buf_b);
                
                lvector double vres = vec_muli(va, vb);
                vres = vec_muli(vres, alpha_vec);
                vres = vec_mula(vtmp, (lvector double)vec_svbcast(1.0), vres);

                vec_st(vres, 0, buf_res);
                vector_store(buf_res, &tmp[i*nj + j], VEC_BYTES);
            } else {
                double a_val = A[i * nk + k];
                double b_val = B[k * nj + j];
                tmp[i * nj + j] += alpha * a_val * b_val;
            }
        }
    }

    vector_free(buf_tmp);
    vector_free(buf_a);
    vector_free(buf_b);
    vector_free(buf_res);
}

__global__ void mm2_kernel2_vec_qwen(int ni, int nj, int nl, double beta, double *tmp,
                            double *C, double *D)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
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

    lvector double *buf_d = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_tmp = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_d || !buf_tmp || !buf_c || !buf_res) {
        if (buf_d) vector_free(buf_d);
        if (buf_tmp) vector_free(buf_tmp);
        if (buf_c) vector_free(buf_c);
        if (buf_res) vector_free(buf_res);
        return;
    }

    lvector double beta_vec = (lvector double)vec_svbcast(beta);

    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / nl;
        int j = idx % nl;
        D[i * nl + j] *= beta;
    }

    for (int k = 0; k < nj; k++) {
        for (int idx = start_idx; idx < end_idx; idx++) {
            int i = idx / nl;
            int j = idx % nl;

            if (j + SIMD_LEN <= nl) {
                vector_load(&D[i*nl + j], buf_d, VEC_BYTES);
                vector_load(&tmp[i*nj + k], buf_tmp, sizeof(double));
                vector_load(&C[k*nl + j], buf_c, VEC_BYTES);

                lvector double vd = vec_ld(0, buf_d);
                lvector double vtmp = vec_ld(0, buf_tmp);
                lvector double vc = vec_ld(0, buf_c);
                
                lvector double vres = vec_muli(vtmp, vc);
                vres = vec_mula(vd, (lvector double)vec_svbcast(1.0), vres);

                vec_st(vres, 0, buf_res);
                vector_store(buf_res, &D[i*nl + j], VEC_BYTES);
            } else {
                double tmp_val = tmp[i * nj + k];
                double c_val = C[k * nl + j];
                D[i * nl + j] += tmp_val * c_val;
            }
        }
    }

    vector_free(buf_d);
    vector_free(buf_tmp);
    vector_free(buf_c);
    vector_free(buf_res);
}