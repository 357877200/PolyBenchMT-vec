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


void ludcmp_cpu1(int n, int i, double *A) {
    double w;
    for (int j = 0; j < i; j++) {
        w = A[i * n + j];
        for (int k = 0; k < j; k++)
            w -= A[i * n + k] * A[k * n + j];
        A[i * n + j] = w / A[j * n + j];
    }
}

void ludcmp_cpu2(int n, int i, double *A) {
    double w;
    for (int j = i; j < n; j++) {
        w = A[i * n + j];
        for (int k = 0; k < i; k++)
            w -= A[i * n + k] * A[k * n + j];
        A[i * n + j] = w;
    }
}

void ludcmp_cpu3(int n, int i, double *A, double *b, double *y) {
    double w = b[i];
    for (int j = 0; j < i; j++)
        w -= A[i * n + j] * y[j];
    y[i] = w;
}

void ludcmp_cpu4(int n, int i, double *A, double *x, double *y) {
    double w = y[i];
    for (int j = i + 1; j < n; j++)
        w -= A[i * n + j] * x[j];
    x[i] = w / A[i * n + i];
}

/*************************************
 * init data
 ************************************/
void init_array(int n, double *A, double *b, double *x, double *y) {
    double fn = (double)n;
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        y[i] = 0.0;
        b[i] = (i+1)/fn/2.0 + 4.0;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++)
            A[i * n + j] = (double)(-j % n) / n + 1.0;
        for (int j = i + 1; j < n; j++)
            A[i * n + j] = 0.0;
        A[i * n + i] = 1.0;
    }
    // 半正定化
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    memset(B, 0, (size_t)n * n * sizeof(double));
    for (int t = 0; t < n; t++)
        for (int r = 0; r < n; r++)
            for (int s = 0; s < n; s++)
                B[r * n + s] += A[r * n + t] * A[s * n + t];
    memcpy(A, B, (size_t)n * n * sizeof(double));
    free(B);
}

/*************************************
 * check
 ************************************/
int check_result(int n, double *ref, double *test) {
    int errNum = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (percentDiff(ref[i*n+j], test[i*n+j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                if (errNum < 10)
                    fprintf(stderr, "Diff @[%d][%d]: H=%.6f D=%.6f\n", i, j, ref[i*n+j], test[i*n+j]);
                errNum++;
            }
    return errNum;
}

/*************************************
 * save perf data
 ************************************/
static void save_data(const char *bench,
                      int n,
                      uint64_t *before, uint64_t *after,
                      uint64_t tDsp, uint64_t tCpu,
                      int clusterId, const char *program,
                      int nthreads, const char *kernel) {
    FILE *fp = fopen("tests/LUDCMP/ludcmp_events.txt", "a");
    if (!fp) { perror("fopen"); return; }
    fprintf(fp, "%s,%d,%d,%s,%s,%d,", bench, clusterId,
            n, program, kernel, nthreads);
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
int main(int argc, char **argv) {
    int clusterId = 1;
    int n = 128;
    int nthreads = 1;
    char *devProgram = "operators/LUDCMP/ludcmp.dev.dat";
    char *kernel1 = "ludcmp_kernel1"; // L 部分
    char *kernel2 = "ludcmp_kernel2"; // U 部分
    char *kernel3 = "ludcmp_kernel3"; // 前向代入
    char *kernel4 = "ludcmp_kernel4"; // 后向代入

    for (int i = 1; i < argc; i++) {
        if (i+1 >= argc) break;
        if (!strcmp(argv[i], "--clusterId") || !strcmp(argv[i], "-c")) clusterId = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n")) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-t")) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--program") || !strcmp(argv[i], "-p")) devProgram = argv[++i];
        else if (!strcmp(argv[i], "--kernel1") || !strcmp(argv[i], "-k1")) kernel1 = argv[++i];
        else if (!strcmp(argv[i], "--kernel2") || !strcmp(argv[i], "-k2")) kernel2 = argv[++i];
        else if (!strcmp(argv[i], "--kernel3") || !strcmp(argv[i], "-k3")) kernel3 = argv[++i];
        else if (!strcmp(argv[i], "--kernel4") || !strcmp(argv[i], "-k4")) kernel4 = argv[++i];
    }

    if (access(devProgram, F_OK)) { fprintf(stderr, "%s not found\n", devProgram); return 2; }

    hthread_dev_open(clusterId);
    hthread_dat_load(clusterId, devProgram);

    size_t sizeMat = (size_t)n * n * sizeof(double);
    size_t sizeVec = (size_t)n * sizeof(double);
    size_t sizeHot = 26 * sizeof(uint64_t);

    double *A_d = (double *)hthread_malloc(clusterId, sizeMat, HT_MEM_RW);
    double *b_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    double *x_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    double *y_d = (double *)hthread_malloc(clusterId, sizeVec, HT_MEM_RW);
    uint64_t *before1 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after1  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *before2 = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);
    uint64_t *after2  = (uint64_t *)hthread_malloc(clusterId, sizeHot, HT_MEM_RW);

    double *A_h = (double *)malloc(sizeMat);
    double *b_h = (double *)malloc(sizeVec);
    double *x_h = (double *)malloc(sizeVec);
    double *y_h = (double *)malloc(sizeVec);

    init_array(n, A_h, b_h, x_h, y_h);
    memcpy(A_d, A_h, sizeMat);
    memcpy(b_d, b_h, sizeVec);
    memcpy(x_d, x_h, sizeVec);
    memcpy(y_d, y_h, sizeVec);

    int groupId = hthread_group_create(clusterId, nthreads);
    int barrier_id = hthread_barrier_create(clusterId);

    uint64_t tDsp1 = 0, tDsp2 = 0, tDsp3 = 0, tDsp4 = 0;
    uint64_t tCpu1 = 0, tCpu2 = 0, tCpu3 = 0, tCpu4 = 0;
    uint64_t st, ed;

    // DSP 核1: L 部分
    st = getCurrentTimeMicros();
    for (int i = 0; i < n; i++) {
        uint64_t args1_k1[] = { (uint64_t)n, (uint64_t)i, (uint64_t)A_d,
                                (uint64_t)before1, (uint64_t)after1 };
        hthread_group_exec(groupId, kernel1, 2, 1, args1_k1);
        hthread_group_wait(groupId);
    }
    ed = getCurrentTimeMicros(); tDsp1 += ed - st;

    // DSP 核2: U 部分
    st = getCurrentTimeMicros();
    for (int i = 0; i < n; i++) {
        uint64_t args1_k2[] = { (uint64_t)n, (uint64_t)i, (uint64_t)A_d,
                                (uint64_t)before1, (uint64_t)after1 };
        hthread_group_exec(groupId, kernel2, 2, 1, args1_k2);
        hthread_group_wait(groupId);
    }
    ed = getCurrentTimeMicros(); tDsp2 += ed - st;

    // DSP 核3: 前向代入
    st = getCurrentTimeMicros();
    for (int i = 0; i < n; i++) {
        uint64_t args3[] = { (uint64_t)n, (uint64_t)i, (uint64_t)A_d,
                             (uint64_t)b_d, (uint64_t)y_d,
                             (uint64_t)before2, (uint64_t)after2 };
        hthread_group_exec(groupId, kernel3, 2, 3, args3);
        hthread_group_wait(groupId);
    }
    ed = getCurrentTimeMicros(); tDsp3 += ed - st;

    // DSP 核4: 后向代入
    st = getCurrentTimeMicros();
    for (int i = n - 1; i >= 0; i--) {
        uint64_t args4[] = { (uint64_t)n, (uint64_t)i, (uint64_t)A_d,
                             (uint64_t)x_d, (uint64_t)y_d,
                             (uint64_t)before2, (uint64_t)after2 };
        hthread_group_exec(groupId, kernel4, 2, 3, args4);
        hthread_group_wait(groupId);
    }
    ed = getCurrentTimeMicros(); tDsp4 += ed - st;

    hthread_group_destroy(groupId);

    // CPU baseline
    st = getCurrentTimeMicros();
    for (int i = 0; i < n; i++)
        ludcmp_cpu1(n, i, A_h);
    ed = getCurrentTimeMicros(); tCpu1 += ed - st;

    st = getCurrentTimeMicros();
    for (int i = 0; i < n; i++)
        ludcmp_cpu2(n, i, A_h);
    ed = getCurrentTimeMicros(); tCpu2 += ed - st;

    st = getCurrentTimeMicros();
    for (int i = 0; i < n; i++)
        ludcmp_cpu3(n, i, A_h, b_h, y_h);
    ed = getCurrentTimeMicros(); tCpu3 += ed - st;

    st = getCurrentTimeMicros();
    for (int i = n - 1; i >= 0; i--)
        ludcmp_cpu4(n, i, A_h, x_h, y_h);
    ed = getCurrentTimeMicros(); tCpu4 += ed - st;

    int errA = check_result(n, A_h, A_d);

    if (errA == 0) {
        printf("WallTime ludcmp_kernel1 (DSP/CPU): %fs / %fs\n", tDsp1/1e6, tCpu1/1e6);
        printf("WallTime ludcmp_kernel2 (DSP/CPU): %fs / %fs\n", tDsp2/1e6, tCpu2/1e6);
        printf("WallTime ludcmp_kernel3 (DSP/CPU): %fs / %fs\n", tDsp3/1e6, tCpu3/1e6);
        printf("WallTime ludcmp_kernel4 (DSP/CPU): %fs / %fs\n", tDsp4/1e6, tCpu4/1e6);
    } else {
        fprintf(stderr, "ludcmp result mismatch!\n");
    }

    free(A_h); free(b_h); free(x_h); free(y_h);
    hthread_free(A_d); hthread_free(b_d); hthread_free(x_d); hthread_free(y_d);
    hthread_free(before1); hthread_free(after1);
    hthread_free(before2); hthread_free(after2);
    hthread_barrier_destroy(barrier_id);
    hthread_dat_unload(clusterId);
    hthread_dev_close(clusterId);

    return (errA==0) ? 0 : 1;
}