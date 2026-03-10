
#define SIMD_LEN    16
#define VEC_BYTES   128

__global__ void ludcmp_kernel1_vec_qwen(int n, int k, double *A)
{
    int thread_id = get_thread_id();
    if (thread_id != 0) return;

    lvector double *buf_a_k = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_p = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_w = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a_k || !buf_a_p || !buf_w) {
        if (buf_a_k) vector_free(buf_a_k);
        if (buf_a_p) vector_free(buf_a_p);
        if (buf_w) vector_free(buf_w);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int j = 0; j < k; j++) {
        if (j + SIMD_LEN <= k) {
            vector_load(&A[k * n + j], buf_w, VEC_BYTES);
            lvector double vw = vec_ld(0, buf_w);

            for (int p = 0; p < j; p += SIMD_LEN) {
                int vec_end_p = min(p + SIMD_LEN, j);
                if (p + SIMD_LEN <= j) {
                    vector_load(&A[k * n + p], buf_a_k, VEC_BYTES);
                    vector_load(&A[p * n + j], buf_a_p, VEC_BYTES);
                    lvector double va_k = vec_ld(0, buf_a_k);
                    lvector double va_p = vec_ld(0, buf_a_p);
                    vw = vec_mulb(vw, va_k, vec_muli(va_k, va_p));
                } else {
                    for (int pp = p; pp < vec_end_p; ++pp) {
                        vw = vec_mulb(vw, (lvector double)vec_svbcast(A[k * n + pp]), (lvector double)vec_svbcast(A[pp * n + j]));
                    }
                }
            }
            vec_st(vw, 0, buf_w);
            vector_store(buf_w, &A[k * n + j], VEC_BYTES);
        } else {
            for (int jj = j; jj < k; ++jj) {
                double w = A[k * n + jj];
                for (int p = 0; p < jj; p++)
                    w -= A[k * n + p] * A[p * n + jj];
                A[k * n + jj] = w / A[jj * n + jj];
            }
        }
    }

    vector_free(buf_a_k);
    vector_free(buf_a_p);
    vector_free(buf_w);
}

__global__ void ludcmp_kernel2_vec_qwen(int n, int k, double *A)
{
    int thread_id = get_thread_id();
    if (thread_id != 0) return;

    lvector double *buf_a_k = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_a_p = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_w = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a_k || !buf_a_p || !buf_w) {
        if (buf_a_k) vector_free(buf_a_k);
        if (buf_a_p) vector_free(buf_a_p);
        if (buf_w) vector_free(buf_w);
        return;
    }

    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    for (int j = k; j < n; j++) {
        if (j + SIMD_LEN <= n) {
            vector_load(&A[k * n + j], buf_w, VEC_BYTES);
            lvector double vw = vec_ld(0, buf_w);

            for (int p = 0; p < k; p += SIMD_LEN) {
                int vec_end_p = min(p + SIMD_LEN, k);
                if (p + SIMD_LEN <= k) {
                    vector_load(&A[k * n + p], buf_a_k, VEC_BYTES);
                    vector_load(&A[p * n + j], buf_a_p, VEC_BYTES);
                    lvector double va_k = vec_ld(0, buf_a_k);
                    lvector double va_p = vec_ld(0, buf_a_p);
                    vw = vec_mulb(vw, va_k, vec_muli(va_k, va_p));
                } else {
                    for (int pp = p; pp < vec_end_p; ++pp) {
                        vw = vec_mulb(vw, (lvector double)vec_svbcast(A[k * n + pp]), (lvector double)vec_svbcast(A[pp * n + j]));
                    }
                }
            }
            vec_st(vw, 0, buf_w);
            vector_store(buf_w, &A[k * n + j], VEC_BYTES);
        } else {
            for (int jj = j; jj < n; ++jj) {
                double w = A[k * n + jj];
                for (int p = 0; p < k; p++)
                    w -= A[k * n + p] * A[p * n + jj];
                A[k * n + jj] = w;
            }
        }
    }

    vector_free(buf_a_k);
    vector_free(buf_a_p);
    vector_free(buf_w);
}

__global__ void ludcmp_kernel3_vec_qwen(int n, int i, double *A, double *b, double *y)
{
    int thread_id = get_thread_id();
    if (thread_id != 0) return;

    lvector double *buf_a_i = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_w = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a_i || !buf_y || !buf_w) {
        if (buf_a_i) vector_free(buf_a_i);
        if (buf_y) vector_free(buf_y);
        if (buf_w) vector_free(buf_w);
        return;
    }

    lvector double vb = (lvector double)vec_svbcast(b[i]);
    lvector double vw = vb;

    for (int j = 0; j < i; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, i);
        if (j + SIMD_LEN <= i) {
            vector_load(&A[i * n + j], buf_a_i, VEC_BYTES);
            vector_load(&y[j], buf_y, VEC_BYTES);
            lvector double va_i = vec_ld(0, buf_a_i);
            lvector double vy = vec_ld(0, buf_y);
            vw = vec_mulb(vw, va_i, vec_muli(va_i, vy));
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                vw = vec_mulb(vw, (lvector double)vec_svbcast(A[i * n + jj]), (lvector double)vec_svbcast(y[jj]));
            }
        }
    }

    double w = sum_f64(vw);
    y[i] = w;

    vector_free(buf_a_i);
    vector_free(buf_y);
    vector_free(buf_w);
}

__global__ void ludcmp_kernel4_vec_qwen(int n, int i, double *A, double *x, double *y)
{
    int thread_id = get_thread_id();
    if (thread_id != 0) return;

    lvector double *buf_a_i = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_x = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_w = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_a_i || !buf_x || !buf_w) {
        if (buf_a_i) vector_free(buf_a_i);
        if (buf_x) vector_free(buf_x);
        if (buf_w) vector_free(buf_w);
        return;
    }

    lvector double vy = (lvector double)vec_svbcast(y[i]);
    lvector double vw = vy;

    for (int j = i + 1; j < n; j += SIMD_LEN) {
        int vec_end_j = min(j + SIMD_LEN, n);
        if (j + SIMD_LEN <= n) {
            vector_load(&A[i * n + j], buf_a_i, VEC_BYTES);
            vector_load(&x[j], buf_x, VEC_BYTES);
            lvector double va_i = vec_ld(0, buf_a_i);
            lvector double vx = vec_ld(0, buf_x);
            vw = vec_mulb(vw, va_i, vec_muli(va_i, vx));
        } else {
            for (int jj = j; jj < vec_end_j; ++jj) {
                vw = vec_mulb(vw, (lvector double)vec_svbcast(A[i * n + jj]), (lvector double)vec_svbcast(x[jj]));
            }
        }
    }

    double w = sum_f64(vw);
    x[i] = w / A[i * n + i];

    vector_free(buf_a_i);
    vector_free(buf_x);
    vector_free(buf_w);
}