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
void gramschmidt_cpu1(int ni, int nj, int k, double *A, double *R)
{
    int i;
    double nrm = 0;
    for (i = 0; i < ni; i++) {
        nrm += A[i * nj + k] * A[i * nj + k];
    }
    R[k * nj + k] = sqrt(nrm);
}

void gramschmidt_cpu2(int ni, int nj, int k, double *A, double *R, double *Q)
{
    int i;
    for (i = 0; i < ni; i++) {
        Q[i * nj + k] = A[i * nj + k] / R[k * nj + k];
    }
}

void gramschmidt_cpu3(int ni, int nj, int k, double *A, double *R, double *Q)
{
    int i, j;
    for (j = k + 1; j < nj; j++) {
        R[k * nj + j] = 0;
        for (i = 0; i < ni; i++) {
            R[k * nj + j] += Q[i * nj + k] * A[i * nj + j];
        }
        for (i = 0; i < ni; i++) {
            A[i * nj + j] = A[i * nj + j] - Q[i * nj + k] * R[k * nj + j];
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int ni, int nj, double *A, double *R, double *Q)
{
    int i, j;

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            A[i * nj + j] = ((double)i * j) / ni;
            Q[i * nj + j] = ((double)i * (j + 1)) / nj;
        }
    }

    for (i = 0; i < nj; i++) {
        for (j = 0; j < nj; j++) {
            R[i * nj + j] = ((double)i * (j + 2)) / nj;
        }
    }
}

int check_result(int ni, int nj, double *A_host, double *A_dev)
{
    int errNum = 0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            if (percentDiff(A_host[i * nj + j], A_dev[i * nj + j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "A diff @[%d][%d] : H=%.6f  D=%.6f\n",
                            i, j, A_host[i * nj + j], A_dev[i * nj + j]);
                }
                errNum++;
            }
        }
    }
    if (errNum) {
        double fail_percent = (100.0 * errNum) / (double)(ni * nj);
        fprintf(stderr, "Non-Matching CPU-DSP Outputs Beyond Error Threshold of %4.2f Percent: %d (%.2lf%%)\n",
                PERCENT_DIFF_ERROR_THRESHOLD, errNum, fail_percent);
    }
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int ni, int nj,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/GRAMSCHM/gramschmidt_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            ni*nj, program, kernel, nthreads);
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
    int    ni          = 1024; // NI
    int    nj          = 1024; // NJ
    int    nthreads    = 1;
    char  *devProgram  = "operators/GRAMSCHM/gramschmidt.dev.dat";
    char  *kernel1     = "gramschmidt_kernel1";
    char  *kernel2     = "gramschmidt_kernel2";
    char  *kernel3     = "gramschmidt_kernel3";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ni"))                                   { ni         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nj"))                                   { nj         = atoi(argv[++i]); }
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
    size_t sizeA = (size_t)ni * nj * sizeof(double);
    size_t sizeR = (size_t)nj * nj * sizeof(double);
    size_t sizeQ = (size_t)ni * nj * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double  *A_d  = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RW);
    double  *R_d  = (double *)hthread_malloc(clusterId, sizeR, HT_MEM_RW);
    double  *Q_d  = (double *)hthread_malloc(clusterId, sizeQ, HT_MEM_RW);
    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before3 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after3  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !R_d || !Q_d || !before1 || !after1 || !before2 || !after2 || !before3 || !after3) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h = (double *)malloc(sizeA);
    double *R_h = (double *)malloc(sizeR);
    double *Q_h = (double *)malloc(sizeQ);

    if (!A_h || !R_h || !Q_h) {
        fprintf(stderr, "malloc host fail\n"); return 1;
    }

    /* init data */
    init_array(ni, nj, A_h, R_h, Q_h);
    
    memcpy(A_d, A_h, sizeA);
    memcpy(R_d, R_h, sizeR);
    memcpy(Q_d, Q_h, sizeQ);
    
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);
    memset(before3, 0, sizeHot); memset(after3, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args1[8];
    args1[0] = (uint64_t)ni;
    args1[1] = (uint64_t)nj;
    args1[2] = 0; // k, to be set in loop
    args1[3] = (uint64_t)A_d;
    args1[4] = (uint64_t)R_d;
    args1[5] = (uint64_t)Q_d;
    args1[6] = (uint64_t)before1;
    args1[7] = (uint64_t)after1;

    uint64_t args2[8];
    args2[0] = (uint64_t)ni;
    args2[1] = (uint64_t)nj;
    args2[2] = 0; // k, to be set in loop
    args2[3] = (uint64_t)A_d;
    args2[4] = (uint64_t)R_d;
    args2[5] = (uint64_t)Q_d;
    args2[6] = (uint64_t)before2;
    args2[7] = (uint64_t)after2;

    uint64_t args3[8];
    args3[0] = (uint64_t)ni;
    args3[1] = (uint64_t)nj;
    args3[2] = 0; // k, to be set in loop
    args3[3] = (uint64_t)A_d;
    args3[4] = (uint64_t)R_d;
    args3[5] = (uint64_t)Q_d;
    args3[6] = (uint64_t)before3;
    args3[7] = (uint64_t)after3;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1 = 0, tDsp2 = 0, tDsp3 = 0, tCpu1 = 0, tCpu2 = 0, tCpu3 = 0;
    uint64_t st, ed;

    for (int k = 0; k < nj; k++) {
        args1[2] = (uint64_t)k;
        args2[2] = (uint64_t)k;
        args3[2] = (uint64_t)k;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel1, 3, 3, args1);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp1 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel2, 3, 3, args2);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp2 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel3, 3, 3, args3);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros(); tDsp3 += ed - st;
    }

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    for (int k = 0; k < nj; k++) {
        st = getCurrentTimeMicros();
        gramschmidt_cpu1(ni, nj, k, A_h, R_h);
        ed = getCurrentTimeMicros(); tCpu1 += ed - st;

        st = getCurrentTimeMicros();
        gramschmidt_cpu2(ni, nj, k, A_h, R_h, Q_h);
        ed = getCurrentTimeMicros(); tCpu2 += ed - st;

        st = getCurrentTimeMicros();
        gramschmidt_cpu3(ni, nj, k, A_h, R_h, Q_h);
        ed = getCurrentTimeMicros(); tCpu3 += ed - st;
    }

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(ni, nj, A_h, A_d);
    if (err != 0) {
        fprintf(stderr, "GRAMSCHM test FAILED!\n");
    } else {
        save_data("GRAMSCHM", ni, nj, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("GRAMSCHM", ni, nj, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        save_data("GRAMSCHM", ni, nj, before3, after3, tDsp3, tCpu3,
                  clusterId, devProgram, nthreads, kernel3);
        printf("WallTime GRAMSCHM_kernel1 (DSP/CPU): %fs / %fs\n",
               tDsp1/1e6, tCpu1/1e6);
        printf("WallTime GRAMSCHM_kernel2 (DSP/CPU): %fs / %fs\n",
               tDsp2/1e6, tCpu2/1e6);
        printf("WallTime GRAMSCHM_kernel3 (DSP/CPU): %fs / %fs\n",
               tDsp3/1e6, tCpu3/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d); hthread_free(R_d); hthread_free(Q_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);
    hthread_free(before3); hthread_free(after3);

    free(A_h); free(R_h); free(Q_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}