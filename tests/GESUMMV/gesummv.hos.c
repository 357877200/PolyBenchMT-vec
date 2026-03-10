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
 * CPU reference kernel
 ************************************/
void gesummv_cpu(int n, double alpha, double beta, 
                 double *A, double *B, double *tmp, double *x, double *y)
{
    int i, j;

    for (i = 0; i < n; i++) {
        tmp[i] = 0;
        y[i] = 0;
        for (j = 0; j < n; j++) {
            tmp[i] = A[i * n + j] * x[j] + tmp[i];
            y[i] = B[i * n + j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int n, double *A, double *B, double *x)
{
    int i, j;

    for (i = 0; i < n; i++) {
        x[i] = ((double)i) / n;

        for (j = 0; j < n; j++) {
            A[i * n + j] = ((double)i * j) / n;
            B[i * n + j] = ((double)i * j) / n;
        }
    }
}

int check_result(int n, double *y_host, double *y_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        if (percentDiff(y_host[i], y_dev[i]) > PERCENT_DIFF_ERROR_THRESHOLD) {
            if (errNum < 10) {
                fprintf(stderr, "y diff @[%d] : H=%.4f  D=%.4f\n",
                        i, y_host[i], y_dev[i]);
            }
            errNum++;
        }
    }
    if (errNum) fprintf(stderr, "Total errors : %d\n", errNum);
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
    FILE *fp = fopen("tests/GESUMMV/gesummv_events.txt", "a");
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
    double alpha       = 43532.0;
    double beta        = 12313.0;
    char  *devProgram  = "operators/GESUMMV/gesummv.dev.dat";
    char  *kernel      = "gesummv_kernel";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                    { n          = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel")   || !strcmp(argv[i], "-k"))   { kernel     = argv[++i]; }
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
    size_t sizeB   = (size_t)n * n * sizeof(double);
    size_t sizeTmp = (size_t)n * sizeof(double);
    size_t sizeX   = (size_t)n * sizeof(double);
    size_t sizeY   = (size_t)n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double   *A_d     = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO);
    double   *B_d     = (double *)hthread_malloc(clusterId, sizeB, HT_MEM_RO);
    double   *tmp_d   = (double *)hthread_malloc(clusterId, sizeTmp, HT_MEM_WO);
    double   *x_d     = (double *)hthread_malloc(clusterId, sizeX, HT_MEM_RO);
    double   *y_d     = (double *)hthread_malloc(clusterId, sizeY, HT_MEM_WO);

    uint64_t *before = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !B_d || !tmp_d || !x_d || !y_d || !before || !after) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h   = (double *)malloc(sizeA);
    double *B_h   = (double *)malloc(sizeB);
    double *tmp_h = (double *)malloc(sizeTmp);
    double *x_h   = (double *)malloc(sizeX);
    double *y_h   = (double *)malloc(sizeY);
    if (!A_h || !B_h || !tmp_h || !x_h || !y_h) { 
        fprintf(stderr, "malloc host fail\n"); return 1; 
    }

    /* init data */
    init_array(n, A_h, B_h, x_h);
    memset(tmp_h, 0, sizeTmp);
    memset(y_h, 0, sizeY);
    
    memcpy(A_d, A_h, sizeA);
    memcpy(B_d, B_h, sizeB);
    memcpy(tmp_d, tmp_h, sizeTmp);
    memcpy(x_d, x_h, sizeX);
    memcpy(y_d, y_h, sizeY);
    memset(before, 0, sizeHot);
    memset(after, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args[10];
    args[0] = (uint64_t)n;
    args[1] = (uint64_t)doubleToRawBits(alpha);  
    args[2] = (uint64_t)doubleToRawBits(beta); 
    args[3] = (uint64_t)A_d;
    args[4] = (uint64_t)B_d;
    args[5] = (uint64_t)tmp_d;
    args[6] = (uint64_t)x_d;
    args[7] = (uint64_t)y_d;
    args[8] = (uint64_t)before;
    args[9] = (uint64_t)after;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp = 0, tCpu = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel, 3, 7, args);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();
    tDsp = ed - st;

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    st = getCurrentTimeMicros();
    gesummv_cpu(n, alpha, beta, A_h, B_h, tmp_h, x_h, y_h);
    ed = getCurrentTimeMicros();
    tCpu = ed - st;

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(n, y_h, y_d);
    if (err != 0) {
        fprintf(stderr, "GESUMMV test FAILED!\n");
    } else {
        save_data("GESUMMV", n, before, after, tDsp, tCpu,
                  clusterId, devProgram, nthreads, kernel);
        printf("WallTime GESUMMV (DSP/CPU): %fs / %fs\n", tDsp/1e6, tCpu/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d); hthread_free(B_d); hthread_free(tmp_d);
    hthread_free(x_d); hthread_free(y_d);
    hthread_free(before); hthread_free(after);

    free(A_h); free(B_h); free(tmp_h); free(x_h); free(y_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}