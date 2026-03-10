
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void mm3_kernel1_vec_qwen(int ni, int nj, int nk, int nl, int nm, double *A, double *B, double *E)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    lvector double *buf_e = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_e || !buf_a) {
        if (buf_e) vector_free(buf_e);
        if (buf_a) vector_free(buf_a);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nj;
        int j = idx % nj;
        if (j % SIMD_LEN == 0 && j + SIMD_LEN <= nj) {
            vec_st(zero_vec, 0, buf_e);
            vector_store(buf_e, &E[i * nj + j], VEC_BYTES);
        } else if (j % SIMD_LEN == 0) {
            for (int jj = j; jj < nj; ++jj) {
                E[i * nj + jj] = 0.0;
            }
        }
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nj;
            int j = idx % nj;
            if (i < ni && j < nj) {
                if (j % SIMD_LEN == 0 && j + SIMD_LEN <= nj) {
                    vector_load(&E[i * nj + j], buf_e, VEC_BYTES);
                    lvector double ve = vec_ld(0, buf_e);
                    lvector double va = (lvector double)vec_svbcast(A[i * nk + k]);

                    vector_load(&B[k * nj + j], buf_a, VEC_BYTES);
                    lvector double vb = vec_ld(0, buf_a);

                    lvector double vprod = vec_muli(va, vb);
                    lvector double vnew = vec_mula(ve, zero_vec, vprod);

                    vec_st(vnew, 0, buf_e);
                    vector_store(buf_e, &E[i * nj + j], VEC_BYTES);
                } else {
                    for (int jj = j; jj < nj; ++jj) {
                        E[i * nj + jj] += A[i * nk + k] * B[k * nj + jj];
                    }
                }
            }
        }
    }

    vector_free(buf_e);
    vector_free(buf_a);
}

__global__ void mm3_kernel2_vec_qwen(int ni, int nj, int nk, int nl, int nm, double *C, double *D, double *F)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = nj * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    lvector double *buf_f = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_f || !buf_c) {
        if (buf_f) vector_free(buf_f);
        if (buf_c) vector_free(buf_c);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        if (j % SIMD_LEN == 0 && j + SIMD_LEN <= nl) {
            vec_st(zero_vec, 0, buf_f);
            vector_store(buf_f, &F[i * nl + j], VEC_BYTES);
        } else if (j % SIMD_LEN == 0) {
            for (int jj = j; jj < nl; ++jj) {
                F[i * nl + jj] = 0.0;
            }
        }
    }

    for (int k = 0; k < nm; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < nj && j < nl) {
                if (j % SIMD_LEN == 0 && j + SIMD_LEN <= nl) {
                    vector_load(&F[i * nl + j], buf_f, VEC_BYTES);
                    lvector double vf = vec_ld(0, buf_f);
                    lvector double vc = (lvector double)vec_svbcast(C[i * nm + k]);

                    vector_load(&D[k * nl + j], buf_c, VEC_BYTES);
                    lvector double vd = vec_ld(0, buf_c);

                    lvector double vprod = vec_muli(vc, vd);
                    lvector double vnew = vec_mula(vf, zero_vec, vprod);

                    vec_st(vnew, 0, buf_f);
                    vector_store(buf_f, &F[i * nl + j], VEC_BYTES);
                } else {
                    for (int jj = j; jj < nl; ++jj) {
                        F[i * nl + jj] += C[i * nm + k] * D[k * nl + jj];
                    }
                }
            }
        }
    }

    vector_free(buf_f);
    vector_free(buf_c);
}

__global__ void mm3_kernel3_vec_qwen(int ni, int nj, int nk, int nl, int nm, double *E, double *F, double *G)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nl;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    lvector double *buf_g = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_e = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_g || !buf_e) {
        if (buf_g) vector_free(buf_g);
        if (buf_e) vector_free(buf_e);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int i = idx / nl;
        int j = idx % nl;
        if (j % SIMD_LEN == 0 && j + SIMD_LEN <= nl) {
            vec_st(zero_vec, 0, buf_g);
            vector_store(buf_g, &G[i * nl + j], VEC_BYTES);
        } else if (j % SIMD_LEN == 0) {
            for (int jj = j; jj < nl; ++jj) {
                G[i * nl + jj] = 0.0;
            }
        }
    }

    for (int k = 0; k < nj; ++k) {
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int i = idx / nl;
            int j = idx % nl;
            if (i < ni && j < nl) {
                if (j % SIMD_LEN == 0 && j + SIMD_LEN <= nl) {
                    vector_load(&G[i * nl + j], buf_g, VEC_BYTES);
                    lvector double vg = vec_ld(0, buf_g);
                    lvector double ve = (lvector double)vec_svbcast(E[i * nj + k]);

                    vector_load(&F[k * nl + j], buf_e, VEC_BYTES);
                    lvector double vf = vec_ld(0, buf_e);

                    lvector double vprod = vec_muli(ve, vf);
                    lvector double vnew = vec_mula(vg, zero_vec, vprod);

                    vec_st(vnew, 0, buf_g);
                    vector_store(buf_g, &G[i * nl + j], VEC_BYTES);
                } else {
                    for (int jj = j; jj < nl; ++jj) {
                        G[i * nl + jj] += E[i * nj + k] * F[k * nl + jj];
                    }
                }
            }
        }
    }

    vector_free(buf_g);
    vector_free(buf_e);
}