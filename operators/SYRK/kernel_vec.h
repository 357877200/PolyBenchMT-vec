
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void syrk_kernel_vec_qwen(int ni, int nj, double alpha, double beta, double *a, double *c)
{
    int tid = get_thread_id();
    int num_threads = get_group_size();
    int total_elements = ni;
    int elements_per_thread = total_elements / num_threads;
    int remainder = total_elements % num_threads;

    int start_idx, end_idx;
    if (tid < remainder) {
        start_idx = tid * (elements_per_thread + 1);
        end_idx = start_idx + (elements_per_thread + 1);
    } else {
        start_idx = remainder * (elements_per_thread + 1) + (tid - remainder) * elements_per_thread;
        end_idx = start_idx + elements_per_thread;
    }

    lvector double *buf_a = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_c = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a || !buf_c) {
        if (buf_a) vector_free(buf_a);
        if (buf_c) vector_free(buf_c);
        return;
    }

    lvector double alpha_vec = (lvector double)vec_svbcast(alpha);
    lvector double beta_vec = (lvector double)vec_svbcast(beta);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < ni; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, ni);

            if (j + SIMD_LEN <= ni) {
                vector_load(&c[i * ni + j], buf_c, VEC_BYTES);
                lvector double vc = vec_ld(0, buf_c);
                vc = vec_muli(vc, beta_vec);

                for (int k = 0; k < nj; ++k) {
                    vector_load(&a[i * nj + k], buf_a, VEC_BYTES);
                    lvector double va = vec_ld(0, buf_a);
                    lvector double temp = vec_muli(alpha_vec, va);

                    vector_load(&a[j * nj + k], buf_a, VEC_BYTES);
                    lvector double vb = vec_ld(0, buf_a);
                    temp = vec_muli(temp, vb);

                    vc = vec_mula(vc, one_vec, temp);
                }

                vec_st(vc, 0, buf_c);
                vector_store(buf_c, &c[i * ni + j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    c[i * ni + jj] *= beta;
                    for (int k = 0; k < nj; ++k) {
                        c[i * ni + jj] += alpha * a[i * nj + k] * a[j * nj + k];
                    }
                }
            }
        }
    }

    vector_free(buf_a);
    vector_free(buf_c);
}