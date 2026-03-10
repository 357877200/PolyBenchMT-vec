
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void covar_kernel1_vec_qwen(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int j_start = (m * thread_id) / group_size;
    int j_end = (m * (thread_id + 1)) / group_size;

    if (thread_id == group_size - 1) {
        j_end = m;
    }

    lvector double *buf_mean = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_data = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_mean || !buf_data) {
        if (buf_mean) vector_free(buf_mean);
        if (buf_data) vector_free(buf_data);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);
    lvector double n_vec = (lvector double)vec_svbcast((double)n);

    for (int j = j_start; j < j_end; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, j_end);

        if (j + SIMD_LEN <= j_end) {
            vec_st(zero_vec, 0, buf_mean);
            vector_store(buf_mean, &mean[j], VEC_BYTES);
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                mean[jj] = 0.0;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = j_start; j < j_end; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, j_end);

            if (j + SIMD_LEN <= j_end) {
                vector_load(&mean[j], buf_mean, VEC_BYTES);
                vector_load(&data[i*m + j], buf_data, VEC_BYTES);

                lvector double vmean = vec_ld(0, buf_mean);
                lvector double vdata = vec_ld(0, buf_data);

                vmean = vec_mula(vmean, (lvector double)vec_svbcast(1.0), vdata);

                vec_st(vmean, 0, buf_mean);
                vector_store(buf_mean, &mean[j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    mean[jj] += data[i*m + jj];
                }
            }
        }
    }

    for (int j = j_start; j < j_end; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, j_end);

        if (j + SIMD_LEN <= j_end) {
            vector_load(&mean[j], buf_mean, VEC_BYTES);
            lvector double vmean = vec_ld(0, buf_mean);
            vmean = vm_fdivd16(vmean, n_vec);
            vec_st(vmean, 0, buf_mean);
            vector_store(buf_mean, &mean[j], VEC_BYTES);
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                mean[jj] /= (double)n;
            }
        }
    }

    vector_free(buf_mean);
    vector_free(buf_data);
}

__global__ void covar_kernel2_vec_qwen(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = m * n;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    lvector double *buf_data = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_mean = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_data || !buf_mean) {
        if (buf_data) vector_free(buf_data);
        if (buf_mean) vector_free(buf_mean);
        return;
    }

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j = idx % m;
        int i = idx / m;

        if (i < n && j < m) {
            if (j + SIMD_LEN <= m && idx + (m-j) <= end_idx) {
                vector_load(&mean[j], buf_mean, VEC_BYTES);
                vector_load(&data[i*m + j], buf_data, VEC_BYTES);

                lvector double vmean = vec_ld(0, buf_mean);
                lvector double vdata = vec_ld(0, buf_data);

                vdata = vec_mulb(vdata, (lvector double)vec_svbcast(1.0), vmean);

                vec_st(vdata, 0, buf_data);
                vector_store(buf_data, &data[i*m + j], VEC_BYTES);

                idx += (SIMD_LEN - 1);
            } else {
                data[i*m + j] -= mean[j];
            }
        }
    }

    vector_free(buf_data);
    vector_free(buf_mean);
}

__global__ void covar_kernel3_vec_qwen(int m, int n, double *symmat, double *data)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    int total_elements = m * m;
    int elements_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elements_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx = (thread_id + 1) * elements_per_thread + (thread_id + 1 < remainder ? thread_id + 1 : remainder);

    lvector double *buf_symmat = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_data1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_data2 = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_symmat || !buf_data1 || !buf_data2) {
        if (buf_symmat) vector_free(buf_symmat);
        if (buf_data1) vector_free(buf_data1);
        if (buf_data2) vector_free(buf_data2);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int idx = start_idx; idx < end_idx; ++idx) {
        int j1 = idx % m;
        int j2 = idx / m;

        if (j1 <= j2) {
            if (j1 + SIMD_LEN <= m && j2 + SIMD_LEN <= m && idx + (m-j1) <= end_idx) {
                vec_st(zero_vec, 0, buf_symmat);
                vector_store(buf_symmat, &symmat[j1*m + j2], VEC_BYTES);

                lvector double vacc = vec_ld(0, buf_symmat);

                for (int i = 0; i < n; ++i) {
                    vector_load(&data[i*m + j1], buf_data1, VEC_BYTES);
                    vector_load(&data[i*m + j2], buf_data2, VEC_BYTES);

                    lvector double vdata1 = vec_ld(0, buf_data1);
                    lvector double vdata2 = vec_ld(0, buf_data2);

                    vacc = vec_mula(vdata1, vdata2, vacc);
                }

                vec_st(vacc, 0, buf_symmat);
                vector_store(buf_symmat, &symmat[j1*m + j2], VEC_BYTES);

                for (int k = 0; k < SIMD_LEN; ++k) {
                    if (j1 + k <= j2) {
                        symmat[(j2 + k)*m + (j1 + k)] = symmat[(j1 + k)*m + (j2 + k)];
                    }
                }

                idx += (SIMD_LEN - 1);
            } else {
                symmat[j1*m + j2] = 0.0;
                for (int i = 0; i < n; ++i) {
                    symmat[j1*m + j2] += data[i*m + j1] * data[i*m + j2];
                }
                symmat[j2*m + j1] = symmat[j1*m + j2];
            }
        }
    }

    vector_free(buf_symmat);
    vector_free(buf_data1);
    vector_free(buf_data2);
}