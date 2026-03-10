#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void deriche_kernel1_vec_qwen(int w, int h, double a1, double a2, double b1, double b2,
                            double *imgIn, double *y1)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按列分块
    int i_start = (w * thread_id) / group_size;
    int i_end   = (w * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) i_end = w;

    // 常量向量初始化
    lvector double va1 = (lvector double)vec_svbcast(a1);
    lvector double va2 = (lvector double)vec_svbcast(a2);
    lvector double vb1 = (lvector double)vec_svbcast(b1);
    lvector double vb2 = (lvector double)vec_svbcast(b2);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    // 分配向量缓冲区
    lvector double *buf_in = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_in || !buf_out) {
        if (buf_in) vector_free(buf_in);
        if (buf_out) vector_free(buf_out);
        return;
    }

    for (int i = i_start; i < i_end; i++) {
        // 标量实现 - 强数据依赖无法向量化
        double ym1 = 0.0;
        double ym2 = 0.0;
        double xm1 = 0.0;
        for (int j = 0; j < h; j++) {
            int idx = i*h + j;
            y1[idx] = a1 * imgIn[idx] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = imgIn[idx];
            ym2 = ym1;
            ym1 = y1[idx];
        }
    }

    vector_free(buf_in);
    vector_free(buf_out);
}

__global__ void deriche_kernel2_vec_qwen(int w, int h, double a3, double a4, double b1, double b2,
                            double *imgIn, double *y2)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按列分块
    int i_start = (w * thread_id) / group_size;
    int i_end   = (w * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) i_end = w;

    // 常量向量初始化
    lvector double va3 = (lvector double)vec_svbcast(a3);
    lvector double va4 = (lvector double)vec_svbcast(a4);
    lvector double vb1 = (lvector double)vec_svbcast(b1);
    lvector double vb2 = (lvector double)vec_svbcast(b2);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    // 分配向量缓冲区
    lvector double *buf_in = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_in || !buf_out) {
        if (buf_in) vector_free(buf_in);
        if (buf_out) vector_free(buf_out);
        return;
    }

    for (int i = i_start; i < i_end; i++) {
        // 标量实现 - 强数据依赖无法向量化
        double yp1 = 0.0, yp2 = 0.0;
        double xp1 = 0.0, xp2 = 0.0;
        for (int j = h-1; j >= 0; j--) {
            int idx = i*h + j;
            y2[idx] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = imgIn[idx];
            yp2 = yp1;
            yp1 = y2[idx];
        }
    }

    vector_free(buf_in);
    vector_free(buf_out);
}

__global__ void deriche_kernel3_vec_qwen(int w, int h, double c1,
                            double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按总元素分块
    int total_elements = w * h;
    int elems_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elems_per_thread + (thread_id < remainder ? thread_id : remainder);
    int end_idx   = (thread_id + 1) * elems_per_thread + ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);

    // 常量向量初始化
    lvector double vc1 = (lvector double)vec_svbcast(c1);
    lvector double one_vec = (lvector double)vec_svbcast(1.0);

    // 分配向量缓冲区
    lvector double *buf_y1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_out = (lvector double *)vector_malloc(VEC_BYTES);
    if (!buf_y1 || !buf_y2 || !buf_out) {
        if (buf_y1) vector_free(buf_y1);
        if (buf_y2) vector_free(buf_y2);
        if (buf_out) vector_free(buf_out);
        return;
    }

    for (int idx = start_idx; idx < end_idx; ) {
        int vec_end = min(idx + SIMD_LEN, end_idx);
        
        if (vec_end - idx == SIMD_LEN) {
            // 向量化处理
            vector_load(&y1[idx], buf_y1, VEC_BYTES);
            vector_load(&y2[idx], buf_y2, VEC_BYTES);
            
            lvector double vy1 = vec_ld(0, buf_y1);
            lvector double vy2 = vec_ld(0, buf_y2);
            
            // y1 + y2
            lvector double vsum = vec_mula(vy1, one_vec, vy2);
            // c1 * (y1 + y2)
            lvector double vres = vec_muli(vc1, vsum);
            
            vec_st(vres, 0, buf_out);
            vector_store(buf_out, &imgOut[idx], VEC_BYTES);
            
            idx += SIMD_LEN;
        } else {
            // 标量尾部处理
            for (int i = idx; i < vec_end; i++) {
                imgOut[i] = c1 * (y1[i] + y2[i]);
            }
            idx = vec_end;
        }
    }

    vector_free(buf_y1);
    vector_free(buf_y2);
    vector_free(buf_out);
}
#define SIMD_LEN 16
#define VEC_BYTES 128

__global__ void deriche_kernel4_vec_qwen(int w, int h, double a5, double a6, double b1, double b2,
                            double *imgOut, double *y1)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按行分块
    int j_start = (h * thread_id) / group_size;
    int j_end   = (h * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) j_end = h;

    // 常量向量初始化
    lvector double va5 = (lvector double)vec_svbcast(a5);
    lvector double va6 = (lvector double)vec_svbcast(a6);
    lvector double vb1 = (lvector double)vec_svbcast(b1);
    lvector double vb2 = (lvector double)vec_svbcast(b2);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    // 分配向量缓冲区
    lvector double *buf_imgOut = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_tmp = (lvector double *)vector_malloc(VEC_BYTES);
    
    if (!buf_imgOut || !buf_y1 || !buf_tmp) {
        if (buf_imgOut) vector_free(buf_imgOut);
        if (buf_y1) vector_free(buf_y1);
        if (buf_tmp) vector_free(buf_tmp);
        return;
    }

    for (int j = j_start; j < j_end; j++) {
        // 标量实现 - 存在强数据依赖，难以向量化
        double tm1 = 0.0;
        double ym1 = 0.0, ym2 = 0.0;
        for (int i = 0; i < w; i++) {
            int idx = i*h + j;
            y1[idx] = a5 * imgOut[idx] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = imgOut[idx];
            ym2 = ym1;
            ym1 = y1[idx];
        }
    }

    vector_free(buf_imgOut);
    vector_free(buf_y1);
    vector_free(buf_tmp);
}

__global__ void deriche_kernel5_vec_qwen(int w, int h, double a7, double a8,
                             double b1, double b2,
                             double *imgOut, double *y2)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按行分块
    int j_start = (h * thread_id) / group_size;
    int j_end   = (h * (thread_id + 1)) / group_size;
    if (thread_id == group_size - 1) j_end = h;

    // 常量向量初始化
    lvector double va7 = (lvector double)vec_svbcast(a7);
    lvector double va8 = (lvector double)vec_svbcast(a8);
    lvector double vb1 = (lvector double)vec_svbcast(b1);
    lvector double vb2 = (lvector double)vec_svbcast(b2);
    lvector double zero_vec = (lvector double)vec_svbcast(0.0);

    // 分配向量缓冲区
    lvector double *buf_imgOut = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_tmp = (lvector double *)vector_malloc(VEC_BYTES);
    
    if (!buf_imgOut || !buf_y2 || !buf_tmp) {
        if (buf_imgOut) vector_free(buf_imgOut);
        if (buf_y2) vector_free(buf_y2);
        if (buf_tmp) vector_free(buf_tmp);
        return;
    }

    for (int j = j_start; j < j_end; j++) {
        // 标量实现 - 存在强数据依赖，难以向量化
        double tp1 = 0.0, tp2 = 0.0;
        double yp1 = 0.0, yp2 = 0.0;
        for (int i = w-1; i >= 0; i--) {
            int idx = i*h + j;
            y2[idx] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;  
            tp1 = imgOut[idx];  
            yp2 = yp1;  
            yp1 = y2[idx];  
        }
    }

    vector_free(buf_imgOut);
    vector_free(buf_y2);
    vector_free(buf_tmp);
}

__global__ void deriche_kernel6_vec_qwen(int w, int h, double c2,
                             double *y1, double *y2, double *imgOut)
{
    int thread_id = get_thread_id();
    int group_size = get_group_size();

    // 按总元素分块
    int total_elements = w * h;
    int elems_per_thread = total_elements / group_size;
    int remainder = total_elements % group_size;

    int start_idx = thread_id * elems_per_thread +
                    (thread_id < remainder ? thread_id : remainder);
    int end_idx   = (thread_id + 1) * elems_per_thread +
                    ((thread_id + 1) < remainder ? (thread_id + 1) : remainder);

    // 常量向量初始化
    lvector double vc2 = (lvector double)vec_svbcast(c2);
    lvector double *buf_y1 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_y2 = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_imgOut = (lvector double *)vector_malloc(VEC_BYTES);
    lvector double *buf_res = (lvector double *)vector_malloc(VEC_BYTES);

    if (!buf_y1 || !buf_y2 || !buf_imgOut || !buf_res) {
        if (buf_y1) vector_free(buf_y1);
        if (buf_y2) vector_free(buf_y2);
        if (buf_imgOut) vector_free(buf_imgOut);
        if (buf_res) vector_free(buf_res);
        return;
    }

    for (int idx = start_idx; idx < end_idx; ) {
        int vec_end = min(idx + SIMD_LEN, end_idx);
        int vec_len = vec_end - idx;

        if (vec_len == SIMD_LEN) {
            // 向量化处理
            vector_load(&y1[idx], buf_y1, VEC_BYTES);
            vector_load(&y2[idx], buf_y2, VEC_BYTES);

            lvector double vy1 = vec_ld(0, buf_y1);
            lvector double vy2 = vec_ld(0, buf_y2);

            // y1 + y2
            lvector double vsum = vec_mula(vy1, (lvector double)vec_svbcast(1.0), vy2);
            // c2 * (y1 + y2)
            lvector double vres = vec_muli(vsum, vc2);

            vec_st(vres, 0, buf_res);
            vector_store(buf_res, &imgOut[idx], VEC_BYTES);

            idx += SIMD_LEN;
        } else {
            // 标量尾部处理
            for (int i = idx; i < vec_end; i++) {
                imgOut[i] = c2 * (y1[i] + y2[i]);
            }
            idx = vec_end;
        }
    }

    vector_free(buf_y1);
    vector_free(buf_y2);
    vector_free(buf_imgOut);
    vector_free(buf_res);
}