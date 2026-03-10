
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void corr_kernel1_vec_qwen(int m, int n, double *mean, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);

    lvector double *buf_mean = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_mean) return;

    for (int j = start_j; j < end_j; j++) {
        mean[j] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = start_j; j < end_j; j += SIMD_LEN) {
            int vec_end_j = min(j + SIMD_LEN, end_j);

            if (j + SIMD_LEN <= end_j) {
                vector_load(&mean[j], buf_mean, VEC_BYTES);
                lvector double vmean = vec_ld(0, buf_mean);
                vector_load(&data[i * m + j], buf_mean, VEC_BYTES);
                lvector double vdata = vec_ld(0, buf_mean);

                lvector double vres = vec_mula(vmean, (lvector double)vec_svbcast(1.0), vdata);

                vec_st(vres, 0, buf_mean);
                vector_store(buf_mean, &mean[j], VEC_BYTES);
            } else {
                for (int jj = j; jj < vec_end_j; ++jj) {
                    mean[jj] += data[i * m + jj];
                }
            }
        }
    }

    for (int j = start_j; j < end_j; j++) {
        mean[j] /= (double)FLOAT_N;
    }

    vector_free(buf_mean);
}

__global__ void corr_kernel2_vec_qwen(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int items_per_thread = (m + total_threads - 1) / total_threads;
    int start_j = thread_id * items_per_thread;
    int end_j = min(start_j + items_per_thread, m);

    lvector double *buf_mean = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_std = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_data = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_mean || !buf_std || !buf_data) {
        if (buf_mean) vector_free(buf_mean);
        if (buf_std) vector_free(buf_std);
        if (buf_data) vector_free(buf_data);
        return;
    }

    lvector double eps_vec = (lvector double)vec_svbcast(EPS);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    for (int j = start_j; j < end_j; j++) {
        std[j] = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = data[i * m + j] - mean[j];
            std[j] += diff * diff;
        }
        std[j] /= (FLOAT_N);
        std[j] = sqrt(std[j]);
        if (std[j] <= EPS) {
            std[j] = 1.0;
        }
    }

    vector_free(buf_mean);
    vector_free(buf_std);
    vector_free(buf_data);
}

__global__ void corr_kernel3_vec_qwen(int m, int n, double *mean, double *std, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int total_elements = n * m;
    int items_per_thread = (total_elements + total_threads - 1) / total_threads;
    int start_idx = thread_id * items_per_thread;
    int end_idx = min(start_idx + items_per_thread, total_elements);

    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / m;
        int j = idx % m;

        if ((i < n) && (j < m)) {
            data[i * m + j] -= mean[j];
            data[i * m + j] /= (sqrt(FLOAT_N) * std[j]);
        }
    }
}

__global__ void corr_kernel4_vec_qwen(int m, int n, double *symmat, double *data)
{
    int thread_id = get_thread_id();
    int total_threads = get_group_size();

    int items_per_thread = ((m - 1) + total_threads - 1) / total_threads;
    int start_j1 = thread_id * items_per_thread;
    int end_j1 = min(start_j1 + items_per_thread, m - 1);

    lvector double *buf_data1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_data2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_symmat = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_data1 || !buf_data2 || !buf_symmat) {
        if (buf_data1) vector_free(buf_data1);
        if (buf_data2) vector_free(buf_data2);
        if (buf_symmat) vector_free(buf_symmat);
        return;
    }

    for (int j1 = start_j1; j1 < end_j1; j1++) {
        symmat[j1 * m + j1] = 1.0;

        for (int j2 = (j1 + 1); j2 < m; j2++) {
            symmat[j1 * m + j2] = 0.0;
            for (int i = 0; i < n; i++) {
                symmat[j1 * m + j2] += data[i * m + j1] * data[i * m + j2];
            }
            symmat[j2 * m + j1] = symmat[j1 * m + j2];
        }
    }

    vector_free(buf_data1);
    vector_free(buf_data2);
    vector_free(buf_symmat);
}