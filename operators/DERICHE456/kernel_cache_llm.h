#define ELEMS_PER_PART 1024

__global__ void deriche_kernel4_cache_llm(int w, int h, double a5, double a6,
                                          double b1, double b2,
                                          double *imgOut, double *y1)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int j_start = (h * tid) / gsz;
    int j_end   = (h * (tid + 1)) / gsz;
    if (tid == gsz - 1) j_end = h;

    // 缓存整列
    double* cache_img = (double*)scalar_malloc(w * sizeof(double));
    double* cache_y   = (double*)scalar_malloc(w * sizeof(double));

    for (int j = j_start; j < j_end; ++j) {
        // 载入整列（按 i 从 0 到 w-1）
        for (int i = 0; i < w; ++i) {
            cache_img[i] = imgOut[i * h + j];
        }

        double tm1 = 0.0;
        double ym1 = 0.0, ym2 = 0.0;

        for (int i = 0; i < w; ++i) {
            cache_y[i] = a5 * cache_img[i] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = cache_img[i];
            ym2 = ym1;
            ym1 = cache_y[i];
        }

        // 写回整列
        for (int i = 0; i < w; ++i) {
            y1[i * h + j] = cache_y[i];
        }
    }

    scalar_free(cache_img);
    scalar_free(cache_y);
}

__global__ void deriche_kernel5_cache_llm(int w, int h, double a7, double a8,
                                          double b1, double b2,
                                          double *imgOut, double *y2)
{
    int tid = get_thread_id();
    int gsz = get_group_size();

    int j_start = (h * tid) / gsz;
    int j_end   = (h * (tid + 1)) / gsz;
    if (tid == gsz - 1) j_end = h;

    double* cache_img = (double*)scalar_malloc(w * sizeof(double));
    double* cache_y   = (double*)scalar_malloc(w * sizeof(double));

    for (int j = j_start; j < j_end; ++j) {
        // 加载整列
        for (int i = 0; i < w; ++i) {
            cache_img[i] = imgOut[i * h + j];
        }

        double tp1 = 0.0, tp2 = 0.0;
        double yp1 = 0.0, yp2 = 0.0;

        for (int i = w - 1; i >= 0; --i) {
            cache_y[i] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;
            tp1 = cache_img[i];
            yp2 = yp1;
            yp1 = cache_y[i];
        }

        // 写回整列
        for (int i = 0; i < w; ++i) {
            y2[i * h + j] = cache_y[i];
        }
    }

    scalar_free(cache_img);
    scalar_free(cache_y);
}

__global__ void deriche_kernel6_cache_llm(int w, int h, double c2,
                             double *y1, double *y2, double *imgOut)
{
    int gsz = get_group_size();
    int tid = get_thread_id();

    int total_elements = w * h;
    int base = total_elements / gsz;
    int rem = total_elements % gsz;

    int start_idx = (tid < rem) ? tid * (base + 1) : rem * (base + 1) + (tid - rem) * base;
    int end_idx = start_idx + ((tid < rem) ? (base + 1) : base);

    double* cache_y1 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_y2 = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));
    double* cache_out = (double*)scalar_malloc(ELEMS_PER_PART * sizeof(double));

    for (int idx = start_idx; idx < end_idx; ) {
        int batch_size = min(ELEMS_PER_PART, end_idx - idx);
        
        scalar_load(&y1[idx], cache_y1, batch_size * sizeof(double));
        scalar_load(&y2[idx], cache_y2, batch_size * sizeof(double));
        
        for (int bi = 0; bi < batch_size; ++bi) {
            cache_out[bi] = c2 * (cache_y1[bi] + cache_y2[bi]);
        }
        
        scalar_store(cache_out, &imgOut[idx], batch_size * sizeof(double));
        
        idx += batch_size;
    }

    scalar_free(cache_y1);
    scalar_free(cache_y2);
    scalar_free(cache_out);
}