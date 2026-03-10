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
void gemm_cpu(int ni, int nj, int nk, double alpha, double beta, 
              double *A, double *B, double *C)
{
    int i, j, k;
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            C[i * nj + j] *= beta;
            
            for (k = 0; k < nk; ++k) {
                C[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
            }
        }
    }
}

/*************************************
 * data initialisation / check
 ************************************/
void init_array(int ni, int nj, int nk, double *A, double *B, double *C)
{
    int i, j;
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nk; j++) {
            A[i * nk + j] = ((double)i * j) / ni;
        }
    }
    
    for (i = 0; i < nk; i++) {
        for (j = 0; j < nj; j++) {
            B[i * nj + j] = ((double)i * j) / ni;
        }
    }
    
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            C[i * nj + j] = ((double)i * j) / ni;
        }
    }
}

int check_result(int ni, int nj, double *C_host, double *C_dev)
{
    int errNum = 0;
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            int idx = i * nj + j;
            if (percentDiff(C_host[idx], C_dev[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10) {
                    fprintf(stderr, "C diff @[%d][%d] : H=%.4f  D=%.4f\n",
                            i, j, C_host[idx], C_dev[idx]);
                }
                errNum++;
            }
        }
    }
    if (errNum) fprintf(stderr, "Total errors : %d\n", errNum);
    return errNum;
}

// int check_result(int ni, int nj, double *C_host, double *C_dev)
// {
//     int errNum = 0;

//     // 新增：打开一个文件用于保存数据
//     FILE *fp_out = fopen("tests/GEMM/gemm_results_compare.txt", "w");
//     if (!fp_out) {
//         perror("fopen gemm_results_compare.txt");
//         // 如果文件打开失败，继续检查逻辑
//     }

//     for (int i = 0; i < ni; i++) {
//         for (int j = 0; j < nj; j++) {
//             int idx = i * nj + j;

//             // 新增：写入CPU和DSP值到文件，每行一个元素
//             if (fp_out) {
//                 fprintf(fp_out, "%.15f %.15f\n", C_host[idx], C_dev[idx]);
//             }

//             // 原有误差检测代码
//             if (percentDiff(C_host[idx], C_dev[idx]) > PERCENT_DIFF_ERROR_THRESHOLD) {
//                 if (errNum < 10) {
//                     fprintf(stderr,
//                             "C diff @[%d][%d] : H=%.4f  D=%.4f\n",
//                             i, j, C_host[idx], C_dev[idx]);
//                 }
//                 errNum++;
//             }
//         }
//     }

//     if (fp_out) fclose(fp_out); // 新增：关闭文件

//     if (errNum)
//         fprintf(stderr, "Total errors : %d\n", errNum);
//     return errNum;
// }

/*************************************
 * save perf-counter data
 ************************************/
static void save_data(const char *bench,
                      int ni, int nj, int nk,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel)
{
    FILE *fp = fopen("tests/GEMM/gemm_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            ni*nj*nk, program, kernel, nthreads);
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
    int    clusterId   = 0;
    int    ni          = 64;
    int    nj          = 64;
    int    nk          = 64;
    int    nthreads    = 1;
    double alpha =32412.0f, beta=2123.0f;
    char  *devProgram  = "operators/GEMM/gemm.dev.dat";
    char  *kernel      = "gemm_kernel";

    /* -------------------- 解析命令行 ------------------ */
    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if      (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c"))  { clusterId  = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ni"))                                   { ni         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nj"))                                   { nj         = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--nk"))                                   { nk         = atoi(argv[++i]); }
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
    size_t sizeA   = (size_t)ni * nk * sizeof(double);
    size_t sizeB   = (size_t)nk * nj * sizeof(double);
    size_t sizeC   = (size_t)ni * nj * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    /* device */
    double   *A_d   = (double *)hthread_malloc(clusterId, sizeA, HT_MEM_RO);
    double   *B_d   = (double *)hthread_malloc(clusterId, sizeB, HT_MEM_RO);
    double   *C_d   = (double *)hthread_malloc(clusterId, sizeC, HT_MEM_RW);

    uint64_t *before = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    if (!A_d || !B_d || !C_d || !before || !after) {
        fprintf(stderr, "device malloc failed\n"); return 1;
    }

    /* host */
    double *A_h = (double *)malloc(sizeA);
    double *B_h = (double *)malloc(sizeB);
    double *C_h = (double *)malloc(sizeC);
    if (!A_h || !B_h || !C_h) { fprintf(stderr, "malloc host fail\n"); return 1; }

    /* init data */
    init_array(ni, nj, nk, A_h, B_h, C_h);
    memcpy(A_d, A_h, sizeA);
    memcpy(B_d, B_h, sizeB);
    memcpy(C_d, C_h, sizeC);
    memset(before, 0, sizeHot);
    memset(after, 0, sizeHot);

    /* -------------------- kernel 参数 ----------------- */
    uint64_t args[10];
    args[0] = (uint64_t)ni;
    args[1] = (uint64_t)nj;
    args[2] = (uint64_t)nk;
    args[3] = (uint64_t)doubleToRawBits(alpha);  
    args[4] = (uint64_t)doubleToRawBits(beta); 
    args[5] = (uint64_t)A_d;
    args[6] = (uint64_t)B_d;
    args[7] = (uint64_t)C_d;
    args[8] = (uint64_t)before;
    args[9] = (uint64_t)after;

    /* -------------------- DSP 执行 -------------------- */
    int groupId = hthread_group_create(clusterId, nthreads);
    if (groupId == -1) { fprintf(stderr, "group create fail\n"); return 2; }

    uint64_t tDsp = 0, tCpu = 0;
    uint64_t st, ed;

    st = getCurrentTimeMicros();
    hthread_group_exec(groupId, kernel, 5, 5, args);
    hthread_group_wait(groupId);
    ed = getCurrentTimeMicros();
    tDsp = ed - st;

    hthread_group_destroy(groupId);

    /* -------------------- CPU 执行 -------------------- */
    st = getCurrentTimeMicros();
    gemm_cpu(ni, nj, nk, alpha, beta, A_h, B_h, C_h);
    ed = getCurrentTimeMicros();
    tCpu = ed - st;

    /* -------------------- 校验结果 ------------------- */
    int err = check_result(ni, nj, C_h, C_d);
    if (err != 0) {
        fprintf(stderr, "GEMM test FAILED!\n");
    } else {
        save_data("GEMM", ni, nj, nk, before, after, tDsp, tCpu,
                  clusterId, devProgram, nthreads, kernel);
        printf("WallTime GEMM (DSP/CPU): %fs / %fs\n", tDsp/1e6, tCpu/1e6);
    }

    /* -------------------- 资源释放 -------------------- */
    hthread_free(A_d); hthread_free(B_d); hthread_free(C_d);
    hthread_free(before); hthread_free(after);

    free(A_h); free(B_h); free(C_h);

    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return err ? 1 : 0;
}