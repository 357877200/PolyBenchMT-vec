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
#ifndef M_PI
#define M_PI 3.14159
#endif

/*************************************
 * CPU reference kernels
 ************************************/
void doitgen_cpu1(int r, int nq, int np, double *A, double *C4, double *sum)
{
    // kernel1: 对每个r值计算矩阵乘法 sum = A * C4
        for (int q_idx = 0; q_idx < nq; q_idx++) {
            for (int p_idx = 0; p_idx < np; p_idx++) {
                sum[r * nq * np + q_idx * np + p_idx] = 0.0;
                for (int s = 0; s < np; s++) {
                    sum[r * nq * np + q_idx * np + p_idx] += 
                        A[r * nq * np + q_idx * np + s] * C4[s * np + p_idx];
                }
            }
        }
}

void doitgen_cpu2(int r, int nq, int np, double *A, double *sum)
{
    // kernel2: 将结果复制回A
        for (int q_idx = 0; q_idx < nq; q_idx++) {
            for (int p_idx = 0; p_idx < np; p_idx++) {
                A[r * nq * np + q_idx * np + p_idx] = sum[r * nq * np + q_idx * np + p_idx];
            }
        }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int nr, int nq, int np, double *A, double *C4)
{
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nq; j++) {
            for (int k = 0; k < np; k++) {
                A[i * nq * np + j * np + k] = ((double)i * j + k) / np;
            }
        }
    }
    
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < np; j++) {
            C4[i * np + j] = ((double)i * j) / np;
        }
    }
}

int check_result(int nr, int nq, int np, double *A_host, double *A_dev)
{
    int errNum = 0;
    for (int r = 0; r < nr; r++) {
        for (int q = 0; q < nq; q++) {
            for (int p = 0; p < np; p++) {
                int idx = r * nq * np + q * np + p;
                if (percentDiff(A_host[idx], A_dev[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                    if (errNum < 10) {
                        fprintf(stderr, "A diff @[%d][%d][%d] : H=%.4f  D=%.4f\n",
                                r, q, p, A_host[idx], A_dev[idx]);
                    }
                    errNum++;
                }
            }
        }
    }
    if (errNum) fprintf(stderr, "Total errors : %d\n", errNum);
    return errNum;
}

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int nr, int nq, int np,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/DOITGEN/doitgen_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            nr*nq*np, program, kernel, nthreads);
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
    int    nr          = 16;
    int    nq          = 16;
    int    np          = 16;
    int    nthreads    = 1;
    char  *devProgram  = "operators/DOITGEN/doitgen.dev.dat";
    char  *kernel1     = "doitgen_kernel1";
    char  *kernel2     = "doitgen_kernel2";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nr"))                                   { nr         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nq"))                                   { nq         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--np"))                                   { np         = atoi(argv[++i]); }
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
    size_t sizeA   = (size_t)nr * nq * np * sizeof(double);
    size_t sizeC4  = (size_t)np * np * sizeof(double);
    size_t sizeSum = (size_t)nr * nq * np * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double  *A_d   = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RW);
    double  *C4    = (double *)hthread_malloc(clusterId, sizeC4, HT_MEM_RO);
    double  *sum   = (double *)hthread_malloc(clusterId, sizeSum, HT_MEM_RW);

    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !C4 || !sum || !before1 || !after1 || !before2 || !after2) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h = (double *)malloc(sizeA);
    double *C4_h = (double *)malloc(sizeC4);
    double *sum_h = (double *)malloc(sizeSum);
    if (!A_h || !C4_h || !sum_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    /* init data */
    init_array(nr, nq, np, A_h, C4_h);
    memcpy(A_d, A_h, sizeA);
    memcpy(C4, C4_h, sizeC4);
    memset(sum, 0, sizeSum);
    memset(sum_h, 0, sizeSum);
    memset(before1, 0, sizeHot); memset(after1, 0, sizeHot);
    memset(before2, 0, sizeHot); memset(after2, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args1[8];
    args1[0] = (uint64_t)nr;
    args1[1] = (uint64_t)nq;
    args1[2] = (uint64_t)np;
    args1[3] = (uint64_t)sum;
    args1[4] = (uint64_t)A_d;
    args1[5] = (uint64_t)C4;
    args1[6] = (uint64_t)before1;
    args1[7] = (uint64_t)after1;

    uint64_t args2[8];
    args2[0] = (uint64_t)nr;
    args2[1] = (uint64_t)nq;
    args2[2] = (uint64_t)np;
    args2[3] = (uint64_t)sum;
    args2[4] = (uint64_t)A_d;
    args2[5] = (uint64_t)C4;
    args2[6] = (uint64_t)before2;
    args2[7] = (uint64_t)after2;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp1=0, tDsp2=0, tCpu1=0, tCpu2=0;
    uint64_t st, ed;
    for (int r = 0; r < nr; r++) {
        args1[0] = (uint64_t)r;
        args2[0] = (uint64_t)r;
        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel1, 3, 5, args1);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros();  tDsp1 += ed - st;

        st = getCurrentTimeMicros();
        hthread_group_exec(groupId, kernel2, 3, 5, args2);
        hthread_group_wait(groupId);
        ed = getCurrentTimeMicros();  tDsp2 += ed - st;
    }


    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    for (int r = 0; r < nr; r++) {
    st = getCurrentTimeMicros();
    doitgen_cpu1(r, nq, np, A_h, C4_h, sum_h);
    ed = getCurrentTimeMicros();  tCpu1 += ed - st;

    st = getCurrentTimeMicros();
    doitgen_cpu2(r, nq, np, A_h, sum_h);
    ed = getCurrentTimeMicros();  tCpu2 += ed - st;
}
    /* -------------------- 校验结果 ------------------- */
    int err = check_result(nr, nq, np, A_h, A_d);
    if (err != 0) {
        fprintf(stderr, "DOITGEN test FAILED!\n");
    } else {
        save_data("DOITGEN", nr, nq, np, before1, after1, tDsp1, tCpu1,
                  clusterId, devProgram, nthreads, kernel1);
        save_data("DOITGEN", nr, nq, np, before2, after2, tDsp2, tCpu2,
                  clusterId, devProgram, nthreads, kernel2);
        printf("WallTime DOITGEN_kernel1 (DSP/CPU): %fs / %fs\n",
                tDsp1/1e6, tCpu1/1e6);
        printf("WallTime DOITGEN_kernel2 (DSP/CPU): %fs / %fs\n",
                tDsp2/1e6, tCpu2/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d); hthread_free(C4); hthread_free(sum);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);

    free(A_h); free(C4_h); free(sum_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}