
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void gemm_kernel_vec_qwen(int ni, int nj, int nk, double alpha, double beta, double *a, double *b,
                                double *c)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = ni * nj;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end = start + elements_per_thread + (thread_id < remainder ? 1 : 0);

    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_c) {
        return;
    }

    lvector double beta_vec = (lvector double)vec_svbcast(beta);

    for (int idx = start; idx < end; idx += SIMD_LEN) {
        if (idx + SIMD_LEN <= end) {
            vector_load(&c[idx], buf_c, VEC_BYTES);
            lvector double vc = vec_ld(0, buf_c);
            vc = vec_muli(vc, beta_vec);
            vec_st(vc, 0, buf_c);
            vector_store(buf_c, &c[idx], VEC_BYTES);
        } else {
            for (int jj = idx; jj < end; ++jj) {
                c[jj] *= beta;
            }
        }
    }

    for (int k = 0; k < nk; ++k) {
        for (int idx = start; idx < end; idx += SIMD_LEN) {
            if (idx + SIMD_LEN <= end) {
                vector_load(&c[idx], buf_c, VEC_BYTES);
                lvector double vc = vec_ld(0, buf_c);
                for (int vec_idx = 0; vec_idx < SIMD_LEN; ++vec_idx) {
                    int flat_idx = idx + vec_idx;
                    int i = flat_idx / nj;
                    int j = flat_idx % nj;
                    double a_val = a[i * nk + k];
                    double b_val = b[k * nj + j];
                    double prod = alpha * a_val * b_val;

                    lvector double prod_vec = (lvector double)vec_svbcast(prod);
                    vc = vec_mula(vc, (lvector double)vec_svbcast(1.0), prod_vec);
                }
                vec_st(vc, 0, buf_c);
                vector_store(buf_c, &c[idx], VEC_BYTES);
            } else {
                for (int jj = idx; jj < end; ++jj) {
                    int i = jj / nj;
                    int j = jj % nj;
                    c[jj] += alpha * a[i * nk + k] * b[k * nj + j];
                }
            }
        }
    }

    vector_free(buf_c);
}