#define ELEMS_PER_PART 1024

__global__ void gramschmidt_kernel1_cache_llm(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    if (tid == 0) {
        double nrm = 0.0;
        int total_elements = ni;
        int base = total_elements / gsz;
        int rem = total_elements % gsz;

        double* cache_a = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

        for (int start = 0; start < ni; ) {
            int batch = min(ELEMS_PER_PART, ni - start);
            scalar_load(&a[start * nj + k], cache_a, batch * sizeof(double));
            
            for (int i = 0; i < batch; ++i) {
                nrm += cache_a[i] * cache_a[i];
            }
            start += batch;
        }

        r[k * nj + k] = sqrt(nrm);
        scalar_free(cache_a);
    }
}

__global__ void gramschmidt_kernel2_cache_llm(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int total_elements = ni;
    int base = total_elements / gsz;
    int rem = total_elements % gsz;

    int start = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end = start + ((tid < rem) ? (base + 1) : base);

    double* cache_a = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_q = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double r_kk = r[k * nj + k];

    for (int idx = start; idx < end; ) {
        int batch = min(ELEMS_PER_PART, end - idx);
        
        scalar_load(&a[idx * nj + k], cache_a, batch * sizeof(double));
        
        for (int i = 0; i < batch; ++i) {
            cache_q[i] = cache_a[i] / r_kk;
        }
        
        scalar_store(cache_q, &q[idx * nj + k], batch * sizeof(double));
        idx += batch;
    }

    scalar_free(cache_a);
    scalar_free(cache_q);
}

__global__ void gramschmidt_kernel3_cache_llm(int ni, int nj, int k, double *a, double *r, double *q)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int total_elements = nj - k - 1;
    if (total_elements <= 0) return;

    int base = total_elements / gsz;
    int rem = total_elements % gsz;

    int start_j = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_j = start_j + ((tid < rem) ? (base + 1) : base);
    if (start_j >= end_j) return;

    int actual_j_start = k + 1 + start_j;
    int actual_j_end = k + 1 + end_j;

    double* cache_r = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_a = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_q = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int j_start = actual_j_start; j_start < actual_j_end; ) {
        int j_batch = min(ELEMS_PER_PART, actual_j_end - j_start);
        
        for (int j = 0; j < j_batch; ++j) {
            cache_r[j] = 0.0;
        }

        for (int i_start = 0; i_start < ni; i_start += ELEMS_PER_PART) {
            int i_batch = min(ELEMS_PER_PART, ni - i_start);
            
            scalar_load(&q[i_start * nj + k], cache_q, i_batch * sizeof(double));
            
            for (int j = 0; j < j_batch; ++j) {
                int actual_j = j_start + j;
                scalar_load(&a[i_start * nj + actual_j], cache_a, i_batch * sizeof(double));
                
                double sum = 0.0;
                for (int i = 0; i < i_batch; ++i) {
                    sum += cache_q[i] * cache_a[i];
                }
                cache_r[j] += sum;
            }
        }

        scalar_store(cache_r, &r[k * nj + j_start], j_batch * sizeof(double));

        for (int i_start = 0; i_start < ni; i_start += ELEMS_PER_PART) {
            int i_batch = min(ELEMS_PER_PART, ni - i_start);
            
            scalar_load(&q[i_start * nj + k], cache_q, i_batch * sizeof(double));
            
            for (int j = 0; j < j_batch; ++j) {
                int actual_j = j_start + j;
                scalar_load(&a[i_start * nj + actual_j], cache_a, i_batch * sizeof(double));
                
                double r_kj = cache_r[j];
                for (int i = 0; i < i_batch; ++i) {
                    cache_a[i] -= cache_q[i] * r_kj;
                }
                
                scalar_store(cache_a, &a[i_start * nj + actual_j], i_batch * sizeof(double));
            }
        }

        j_start += j_batch;
    }

    scalar_free(cache_r);
    scalar_free(cache_a);
    scalar_free(cache_q);
}