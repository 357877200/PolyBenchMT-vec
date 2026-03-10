#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include "hthread_host.h"
#include "../common/tool.h" // percentDiff(), getCurrentTimeMicros()等

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/*************************************
 * CPU reference kernel
 ************************************/
void syrk_cpu(int ni, int nj, double alpha, double beta, double *A, double *C)
{
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < ni; j++) {
            C[i * ni + j] *= beta;
            for (int k = 0; k < nj; k++) {
                C[i * ni + j] += alpha * A[i * nj + k] * A[j * nj + k];
            }
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_arrays(int ni, int nj, double *alpha, double *beta, double *A, double *C)
{
    *alpha = 32412;
    *beta = 2123;

    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            A[i * nj + j] = ((double)i * j) / ni;
        }
    }

    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < ni; j++) {
            C[i * ni + j] = ((double)i * j) / ni;
        }
    }
}

int check_result(int ni, double *C_host, double *C_dev)
{
    int errNum = 0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < ni; j++) {
            if (percentDiff(C_host[i * ni + j], C_dev[i * ni + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "C diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, C_host[i * ni + j], C_dev[i * ni + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(ni * ni);
        fprintf(stderr, "Non-Matching CPU-DSP Outputs Beyond Error Threshold of %4.2f Percent: %d (%.2lf%%)\n",
                PERCENT_DIFF_ERROR_THRESHOLD, errNum, fail_percent);
    }
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int ni,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/SYRK/syrk_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            ni * ni, program, kernel, nthreads);
    for (int i = 0; i < 26; i++) {
        fprintf(fp, "%lu", after[i] - before[i]);
        if (i != 25) fputc(',', fp);
    }
    fprintf(fp, ",%f,%f\n", tDsp / 1e6, tCpu / 1e6);
    fclose(fp);
}

/*************************************
 * main
 ************************************/
int main(int argc, char **argv)
{
    /* -------------------- 默认参数 -------------------- */
    int    clusterId   = 1;
    int    ni          = 1000; // NI
    int    nj          = 1000; // NJ
    int    nthreads    = 1;
    char  *devProgram  = "operators/SYRK/syrk.dev.dat";
    char  *kernel      = "syrk_kernel";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ni"))                                   { ni        = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nj"))                                   { nj        = atoi(argv[++i]); }
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
    size_t sizeMatC = (size_t)ni * ni * sizeof(double);
    size_t sizeMatA = (size_t)ni * nj * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double  *A_d  = (double *)hthread_malloc(clusterId, sizeMatA, HT_MEM_RO);
    double  *C_d  = (double *)hthread_malloc(clusterId, sizeMatC, HT_MEM_RW);
    uint64_t *before = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !C_d || !before || !after) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h = (double *)malloc(sizeMatA);
    double *C_h = (double *)malloc(sizeMatC);

    if (!A_h || !C_h) {
        fprintf(stderr, "malloc host fail\n"); return 1;
    }

    /* init data */
    double alpha, beta;
    init_arrays(ni, nj, &alpha, &beta, A_h, C_h);
    
    memcpy(A_d, A_h, sizeMatA);
    memcpy(C_d, C_h, sizeMatC);
    
    memset(before, 0, sizeHot); memset(after, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args[8];
    args[0] = (uint64_t)ni;
    args[1] = (uint64_t)nj;
    args[2] = doubleToRawBits(alpha);
    args[3] = doubleToRawBits(beta);
    args[4] = (uint64_t)A_d;
    args[5] = (uint64_t)C_d;
    args[6] = (uint64_t)before;
    args[7] = (uint64_t)after;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp = 0, tCpu = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel, 4, 4, args);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros(); tDsp = ed - st;

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    st = getCurrentTimeMicros();
    syrk_cpu(ni, nj, alpha, beta, A_h, C_h);
    ed = getCurrentTimeMicros(); tCpu = ed - st;

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(ni, C_h, C_d);
    if (err != 0) {
        fprintf(stderr, "SYRK test FAILED!\n");
    } else {
        save_data("SYRK", ni, before, after, tDsp, tCpu,
                  clusterId, devProgram, nthreads, kernel);
        printf("WallTime SYRK_kernel (DSP/CPU): %fs / %fs\n",
               tDsp / 1e6, tCpu / 1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d);
    hthread_free(C_d);
    hthread_free(before);
    hthread_free(after);

    free(A_h);
    free(C_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}