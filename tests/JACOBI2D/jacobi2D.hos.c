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
 * CPU reference kernels
 ************************************/
void jacobi2D_cpu1(int n, double *A, double *B)
{
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + (j - 1)] + A[i * n + (j + 1)] + 
                                  A[(i + 1) * n + j] + A[(i - 1) * n + j]);
        }
    }
}

void jacobi2D_cpu2(int n, double *A, double *B)
{
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            A[i * n + j] = B[i * n + j];
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int n, double *A, double *B)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = ((double)i * (j + 2) + 10) / n;
            B[i * n + j] = ((double)(i - 4) * (j - 1) + 11) / n;
        }
    }
}

int check_result(int n, double *A_host, double *A_dev, double *B_host, double *B_dev)
{
    int errNum = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (percentDiff(A_host[i * n + j], A_dev[i * n + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "A diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, A_host[i * n + j], A_dev[i * n + j]);
                }
                errNum++;
            }
            if (percentDiff(B_host[i * n + j], B_dev[i * n + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "B diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, B_host[i * n + j], B_dev[i * n + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(2 * n * n);
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
    FILE *fp = fopen("tests/JACOBI2D/jacobi2D_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            n * n, program, kernel, nthreads);
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
    int    n           = 128; // N
    int    tsteps      = 20;  // TSTEPS
    int    nthreads    = 1;
    char  *devProgram  = "operators/JACOBI2D/jacobi2D.dev.dat";
    char  *kernel1     = "jacobi2D_kernel1";
    char  *kernel2     = "jacobi2D_kernel2";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--n"))                                    { n         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--tsteps"))                               { tsteps    = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads")  || !strcmp(argv[i], "-t"))   { nthreads   = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--program")  || !strcmp(argv[i], "-p"))   { devProgram = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel1")  || !strcmp(argv[i], "-k1"))  { kernel1    = argv[++i]; }
        else if (!strcmp(argv[i], "--kernel2")  || !strcmp(argv[i], "-k2"))  { kernel2    = argv[++i]; }
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
    size_t sizeMat = (size_t)(n * n * sizeof(double));
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double  *A_d  = (double *)hthread_malloc(clusterId, sizeMat, HT_MEM_RW);
    double  *B_d  = (double *)hthread_malloc(clusterId, sizeMat, HT_MEM_RW);
    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !B_d || !before1 || !after1 || !before2 || !after2) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h = (double *)malloc(sizeMat);
    double *B_h = (double *)malloc(sizeMat);

    if (!A_h || !B_h) {
        fprintf(stderr, "malloc host fail\n"); return 1;
    }

    /* init data */
    init_array(n, A_h, B_h);
    
    memcpy(A_d, A_h, sizeMat);
    memcpy(B_d, B_h, sizeMat);
    
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args1[5];
    args1[0] = (uint64_t)n;
    args1[1] = (uint64_t)A_d;
    args1[2] = (uint64_t)B_d;
    args1[3] = (uint64_t)before1;
    args1[4] = (uint64_t)after1;

    uint64_t args2[5];
    args2[0] = (uint64_t)n;
    args2[1] = (uint64_t)A_d;
    args2[2] = (uint64_t)B_d;
    args2[3] = (uint64_t)before2;
    args2[4] = (uint64_t)after2;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1 = 0, tDsp2 = 0, tCpu1 = 0, tCpu2 = 0;
    uint64_t st, ed;

    for (int t = 0; t < tsteps; t++) {
        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel1, 1, 4, args1);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp1 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel2, 1, 4, args2);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp2 += ed - st;
    }

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    for (int t = 0; t < tsteps; t++) {
        st = getCurrentTimeMicros();
        jacobi2D_cpu1(n, A_h, B_h);
        ed = getCurrentTimeMicros(); tCpu1 += ed - st;

        st = getCurrentTimeMicros();
        jacobi2D_cpu2(n, A_h, B_h);
        ed = getCurrentTimeMicros(); tCpu2 += ed - st;
    }

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(n, A_h, A_d, B_h, B_d);
    if (err != 0) {
        fprintf(stderr, "JACOBI2D test FAILED!\n");
    } else {
        save_data("JACOBI2D", n, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("JACOBI2D", n, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        printf("WallTime JACOBI2D_kernel1 (DSP/CPU): %fs / %fs\n",
               tDsp1 / 1e6, tCpu1 / 1e6);
        printf("WallTime JACOBI2D_kernel2 (DSP/CPU): %fs / %fs\n",
               tDsp2 / 1e6, tCpu2 / 1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d); hthread_free(B_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);

    free(A_h); free(B_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}