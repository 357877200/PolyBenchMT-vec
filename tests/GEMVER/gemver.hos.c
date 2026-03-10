#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h"          // percentDiff()、getCurrentTimeMicros()等

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/*************************************
 * CPU reference kernels
 ************************************/
void gemver_cpu1(int n, double alpha, double beta, double *A, double *u1, double *v1, double *u2, double *v2)
{
    // kernel1: A = A + u1*v1' + u2*v2'
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }
}

void gemver_cpu2(int n, double alpha, double beta, double *A, double *x, double *y, double *z)
{
    // kernel2: x = beta * A' * y + z
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            x[i] = x[i] + beta * A[j * n + i] * y[j];
        }
    }
    
    for (int i = 0; i < n; i++) {
        x[i] = x[i] + z[i];
    }
}

void gemver_cpu3(int n, double alpha, double beta, double *A, double *w, double *x)
{
    // kernel3: w = alpha * A * x
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            w[i] = w[i] + alpha * A[i * n + j] * x[j];
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int n, double *alpha, double *beta, double *A, double *u1, double *v1, 
                double *u2, double *v2, double *w, double *x, double *y, double *z)
{
    *alpha = 43532;
    *beta = 12313;

    for (int i = 0; i < n; i++) {
        u1[i] = i;
        u2[i] = (i + 1) / (double)n / 2.0;
        v1[i] = (i + 1) / (double)n / 4.0;
        v2[i] = (i + 1) / (double)n / 6.0;
        y[i] = (i + 1) / (double)n / 8.0;
        z[i] = (i + 1) / (double)n / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;

        for (int j = 0; j < n; j++) {
            A[i * n + j] = ((double)i * j) / n;
        }
    }
}

int check_result(int n, double *w_host, double *w_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        if (percentDiff(w_host[i], w_dev[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "w diff @[%d] : H=%.6f  D=%.6f\n",
                        i, w_host[i], w_dev[i]);
            }
            errNum++;
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)n;
        fprintf(stderr, "Non-Matching CPU-DSP Outputs Beyond Error Threshold of %4.2f Percent: %d (%.2lf%%)\n",
                PERCENT_DIFF_ERROR_THRESHOLD, errNum, fail_percent);
    }
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int n,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/GEMVER/gemver_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            n*n, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i]-before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp/1e6, tCpu/1e6);
    fclose(fp);
}

/*************************************
 * main
 ************************************/
int main(int argc, char **argv)
{
    /* -------------------- 默认参数 -------------------- */
    int    clusterId   = 1;
    int    n           = 64;
    int    nthreads    = 1;
    char  *devProgram  = "operators/GEMVER/gemver.dev.dat";
    char  *kernel1     = "gemver_kernel1";
    char  *kernel2     = "gemver_kernel2";
    char  *kernel3     = "gemver_kernel3";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                    { n          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1")  || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel2")  || !strcmp(argv[i], "-k2"))  { kernel2    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel3")  || !strcmp(argv[i], "-k3"))  { kernel3    = argv[++i]; }
    }

    /* -------------------- 参数合法性 ------------------ */
    if (clusterId < 0 || clusterId > 3) { fprintf(stderr, "invalid clusterId\n"); return 2; }
    if (nthreads <= 0)                 { fprintf(stderr, "invalid nthreads\n");  return 2; }
    if (access(devProgram, F_OK))      { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    /* -------------------- 打开设备 -------------------- */
    int retc;
    retc = hthread_dev_open(clusterId);  if (retc) { fprintf(stderr, "dev open fail\n"); return retc; }
    retc = hthread_dat_load(clusterId, devProgram);
    if (retc) { fprintf(stderr, "load dat fail\n"); return retc; }

    int avail = hthread_get_avail_threads(clusterId);
    if (nthreads > avail) {
        fprintf(stderr, "thread overflow: avail %d, ask %d\n", avail, nthreads);
        hthread_dat_unload(clusterId); hthread_dev_close(clusterId); return 2;
    }

    /* -------------------- 内存分配 -------------------- */
    size_t sizeA   = (size_t)n * n * sizeof(double);
    size_t sizeVec = (size_t)n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double  *A_d  = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RW);
    double  *u1_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RO);
    double  *v1_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RO);
    double  *u2_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RO);
    double  *v2_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RO);
    double  *w_d  = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    double  *x_d  = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    double  *y_d  = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RO);
    double  *z_d  = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RO);

    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after3  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !u1_d || !v1_d || !u2_d || !v2_d || !w_d || !x_d || !y_d || !z_d ||
        !before1 || !after1 || !before2 || !after2 || !before3 || !after3) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h  = (double *)malloc(sizeA);
    double *u1_h = (double *)malloc(sizeVec);
    double *v1_h = (double *)malloc(sizeVec);
    double *u2_h = (double *)malloc(sizeVec);
    double *v2_h = (double *)malloc(sizeVec);
    double *w_h  = (double *)malloc(sizeVec);
    double *x_h  = (double *)malloc(sizeVec);
    double *y_h  = (double *)malloc(sizeVec);
    double *z_h  = (double *)malloc(sizeVec);
    
    if (!A_h || !u1_h || !v1_h || !u2_h || !v2_h || !w_h || !x_h || !y_h || !z_h) {
        fprintf(stderr, "malloc host fail\n"); return 1;
    }

    /* init data */
    double alpha, beta;
    init_array(n, &alpha, &beta, A_h, u1_h, v1_h, u2_h, v2_h, w_h, x_h, y_h, z_h);
    
    memcpy(A_d, A_h, sizeA);
    memcpy(u1_d, u1_h, sizeVec);
    memcpy(v1_d, v1_h, sizeVec);
    memcpy(u2_d, u2_h, sizeVec);
    memcpy(v2_d, v2_h, sizeVec);
    memcpy(w_d, w_h, sizeVec);
    memcpy(x_d, x_h, sizeVec);
    memcpy(y_d, y_h, sizeVec);
    memcpy(z_d, z_h, sizeVec);
    
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);
    memset(before3, 0, sizeHot); memset(after3, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args1[10];
    args1[0] = (uint64_t)n;
    args1[1] = *(uint64_t*)&alpha;  // 浮点数转换为uint64_t
    args1[2] = *(uint64_t*)&beta;
    args1[3] = (uint64_t)A_d;
    args1[4] = (uint64_t)v1_d;
    args1[5] = (uint64_t)v2_d;
    args1[6] = (uint64_t)u1_d;
    args1[7] = (uint64_t)u2_d;
    args1[8] = (uint64_t)before1;
    args1[9] = (uint64_t)after1;

    uint64_t args2[9];
    args2[0] = (uint64_t)n;
    args2[1] = *(uint64_t*)&alpha;
    args2[2] = *(uint64_t*)&beta;
    args2[3] = (uint64_t)A_d;
    args2[4] = (uint64_t)x_d;
    args2[5] = (uint64_t)y_d;
    args2[6] = (uint64_t)z_d;
    args2[7] = (uint64_t)before2;
    args2[8] = (uint64_t)after2;

    uint64_t args3[8];
    args3[0] = (uint64_t)n;
    args3[1] = *(uint64_t*)&alpha;
    args3[2] = *(uint64_t*)&beta;
    args3[3] = (uint64_t)A_d;
    args3[4] = (uint64_t)x_d;
    args3[5] = (uint64_t)w_d;
    args3[6] = (uint64_t)before3;
    args3[7] = (uint64_t)after3;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1=0, tDsp2=0, tDsp3=0, tCpu1=0, tCpu2=0, tCpu3=0;
    uint64_t st, ed;
    
    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel1, 3, 7, args1);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();  tDsp1 = ed - st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel2, 3, 6, args2);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();  tDsp2 = ed - st;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel3, 3, 5, args3);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();  tDsp3 = ed - st;

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    st = getCurrentTimeMicros();
    gemver_cpu1(n, alpha, beta, A_h, u1_h, v1_h, u2_h, v2_h);
    ed = getCurrentTimeMicros();  tCpu1 = ed - st;

    st = getCurrentTimeMicros();
    gemver_cpu2(n, alpha, beta, A_h, x_h, y_h, z_h);
    ed = getCurrentTimeMicros();  tCpu2 = ed - st;

    st = getCurrentTimeMicros();
    gemver_cpu3(n, alpha, beta, A_h, w_h, x_h);
    ed = getCurrentTimeMicros();  tCpu3 = ed - st;

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(n, w_h, w_d);
    if (err != 0) {
        fprintf(stderr, "GEMVER test FAILED!\n");
    } else {
        save_data("GEMVER", n, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("GEMVER", n, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        save_data("GEMVER", n, before3, after3, tDsp3, tCpu3,
                  clusterId, devProgram, nthreads, kernel3);
        printf("WallTime GEMVER_kernel1 (DSP/CPU): %fs / %fs\n",
                tDsp1/1e6, tCpu1/1e6);
        printf("WallTime GEMVER_kernel2 (DSP/CPU): %fs / %fs\n",
                tDsp2/1e6, tCpu2/1e6);
        printf("WallTime GEMVER_kernel3 (DSP/CPU): %fs / %fs\n",
                tDsp3/1e6, tCpu3/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d); hthread_free(u1_d); hthread_free(v1_d); hthread_free(u2_d); hthread_free(v2_d);
    hthread_free(w_d); hthread_free(x_d); hthread_free(y_d); hthread_free(z_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);
    hthread_free(before3); hthread_free(after3);

    free(A_h); free(u1_h); free(v1_h); free(u2_h); free(v2_h);
    free(w_h); free(x_h); free(y_h); free(z_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}