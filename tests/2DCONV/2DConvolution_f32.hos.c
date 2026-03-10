// 使用的相对路径，在cjk/operators_lib路径下执行文件
// author：cjk  time：2025/7/22
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include "hthread_host.h"
#include "../common/tool.h"

void convolution2D_cpu(int ni, int nj, const float *A, float *B)
{
    // 卷积核系数，与核函数中一致
    float c11 = +0.2, c21 = +0.5, c31 = -0.8;
    float c12 = -0.3, c22 = +0.6, c32 = -0.9;
    float c13 = +0.4, c23 = +0.7, c33 = +0.10;

    // 遍历有效区域，边界不处理
    for (int i = 1; i < ni - 1; ++i)
    {
        for (int j = 1; j < nj - 1; ++j)
        {
            B[i * nj + j] = 
                c11 * A[(i - 1) * nj + (j - 1)] + c21 * A[(i - 1) * nj + j] + c31 * A[(i - 1) * nj + (j + 1)] +
                c12 * A[i * nj + (j - 1)] + c22 * A[i * nj + j] + c32 * A[i * nj + (j + 1)] +
                c13 * A[(i + 1) * nj + (j - 1)] + c23 * A[(i + 1) * nj + j] + c33 * A[(i + 1) * nj + (j + 1)];
        }
    }
}

int check_convolution2D(int ni, int nj, float *B_gold, float *B)
{
    int errNum = 0;
    // 只检查有效区域 (1 to ni-2, 1 to nj-2)
    for (int i = 1; i < ni - 1; ++i)
    {
        for (int j = 1; j < nj - 1; ++j)
        {
            int idx = i * nj + j;
            if (fabs(B[idx] - B_gold[idx]) > 1e-5) // 使用浮点数比较，设置一个容差
            {
                if(errNum<=10)fprintf(stderr, "Data error at (%d,%d): Host=%.2f, Device=%.2f\n",
                        i, j, B_gold[idx], B[idx]);
                errNum++;
            }
            else
            {
                // fprintf(stdout, "Data match at (%d,%d): Host=%.2f, Device=%.2f\n",
                //         i, j, B_gold[idx], B[idx]);
            }
            
        }
    }
    if (errNum == 0)
    {
        fprintf(stdout, "All data matched within tolerance.\n");
    }
    else
    {
        fprintf(stdout, "Total errors: %d\n", errNum);
    }
    return errNum;
}

// 传入的参数根据实际算子需要的参数来调整
void save_data(int ni,int nj,uint64_t *before_hot_data, uint64_t *after_hot_data, uint64_t timeDsp,uint64_t  timeCpu,int clusterId, char *devProgram, int nthreads)
{
    FILE *file = fopen("tests/polybenchMT/2DCONV/2DConvolution_events.txt", "a");
    if (file == NULL)
    {
        perror("Error opening file");
        return;
    }

    fprintf(file, "%d,%d,%s,%d,Difference,", clusterId, ni*nj, devProgram, nthreads);
    for (int eid = 0; eid < 26; eid++)
    {
        fprintf(file, "%lu", after_hot_data[eid] - before_hot_data[eid]);
        if (eid < 25)
            fprintf(file, ",");
    }
    fprintf(file, ",%fs,%fs\n", timeDsp/ 1e6,timeCpu/ 1e6);
    fclose(file);
}


int main(int argc, char **argv)
{
    int retc;
    int clusterId = 1;
    int ni = 50;      // 行数，默认4096
    int nj = 50;      // 列数，默认4096
    char *devProgram = "operators/polybenchMT/2DCONV/2DConvolution.dev.dat";
    int nthreads = 1;
    char *kernel = "convolution2D_kernel_vec_f32";
    char info[256];
    uint64_t timeGold, timeDev;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc) {  // 确保有值跟随在选项后面
            if (strcmp(argv[i], "--clusterId") == 0 || strcmp(argv[i], "-c") == 0) {
                clusterId = atoi(argv[i + 1]);
                i++;  // 跳过已处理的参数值
            } else if (strcmp(argv[i], "--ni") == 0) {
                ni = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--nj") == 0) {
                nj = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--program") == 0 || strcmp(argv[i], "-p") == 0) {
                devProgram = argv[i + 1];
                i++;
            } else if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
                nthreads = atoi(argv[i + 1]);
                i++;
            } else if (strcmp(argv[i], "--kernel") == 0 || strcmp(argv[i], "-k") == 0) {
                kernel = argv[i + 1];
                i++;
            }
        }
    }

    sprintf(info, "clusterId : %d。 ni : %d。 nj : %d。 devProgram : %s。 nthreads : %d。",
            clusterId, ni, nj, devProgram, nthreads);

    if (clusterId < 0 || clusterId > 3) { M_logError("invalid clusterId : %d", clusterId); return 2; }
    if (nthreads <= 0) { M_logError("invalid nthreads : %d", nthreads); return 2; }
    if (fileIsExist(devProgram) != 0) { M_logError("%s : No such file or directory", devProgram); return 2; }
    if (ni < 3 || nj < 3) { M_logError("ni and nj must be at least 3 for convolution (got ni=%d, nj=%d)", ni, nj); return 2; }

    retc = hthread_dev_open(clusterId); M_checkRetC(retc, hthread_dev_open);
    retc = hthread_dat_load(clusterId, devProgram); M_checkRetC(retc, hthread_dat_load);
    int availThreads = hthread_get_avail_threads(clusterId);
    if (nthreads > availThreads) {
        M_logError("number of threads overflow: avail %d, actual %d", availThreads, nthreads);
        retc = hthread_dat_unload(clusterId);
        hthread_dev_close(clusterId);
        return 2;
    }

    size_t bufSize = (size_t)ni * nj * sizeof(float);
    float *A = (float *)hthread_malloc(clusterId, bufSize, HT_MEM_RO); M_checkMalloc(A);
    float *B = (float *)hthread_malloc(clusterId, bufSize, HT_MEM_RW); M_checkMalloc(B);
    float *B_gold = (float *)malloc(bufSize); M_checkMalloc(B_gold);

    uint64_t *before_hot_data = (uint64_t *)hthread_malloc(clusterId, 26 * sizeof(uint64_t), HT_MEM_RW); M_checkMalloc(before_hot_data);
    uint64_t *after_hot_data = (uint64_t *)hthread_malloc(clusterId, 26 * sizeof(uint64_t), HT_MEM_RW); M_checkMalloc(after_hot_data);

    // 初始化数据
    for (size_t i = 0; i < (size_t)ni * nj; i++) {
        A[i] = 1.0 * rand() / RAND_MAX * 6.28;
        B[i] = 0.0;
        B_gold[i] = 0.0;
    }
    for (size_t i = 0; i < 26; i++) before_hot_data[i] = after_hot_data[i] = 0;

    uint64_t args[6];
    args[0] = (uint64_t)ni;
    args[1] = (uint64_t)nj;
    args[2] = (uint64_t)A;
    args[3] = (uint64_t)B;
    args[4] = (uint64_t)before_hot_data;
    args[5] = (uint64_t)after_hot_data;

    timeDev = getCurrentTimeMicros();
    int threadId = hthread_group_create(clusterId, nthreads, kernel, 2, 6, args);
    if (threadId == -1) { M_logError("Failed to create threads with %s", kernel); return 2; }
    hthread_group_wait(threadId);
    timeDev = getCurrentTimeMicros() - timeDev;

    timeGold = getCurrentTimeMicros();
    convolution2D_cpu(ni, nj, A, B_gold);
    timeGold = getCurrentTimeMicros() - timeGold;

    int errNum = check_convolution2D(ni, nj, B_gold, B);
 
    if (errNum != 0)
    {
        fprintf(stderr, "Failed to test 2D Convolution!\n");
        return 1;
    }
    else
    {
        // 保存数据
        save_data(ni ,nj,before_hot_data, after_hot_data, timeDev, timeGold, clusterId, devProgram, nthreads);
        fprintf(stdout, "WallTime convolution2D_kernel : %fs\n", timeDev / 1e6);
        fprintf(stdout, "WallTime convolution2D_cpu    : %fs\n", timeGold / 1e6);
        return 0;
    }

    // 关闭设备端
    hthread_group_destroy(threadId);
    if (A) hthread_free(A);
    if (B) hthread_free(B);
    if (B_gold) free(B_gold);
    retc = hthread_dat_unload(clusterId); M_checkRetC(retc, hthread_dat_unload);
    hthread_dev_close(clusterId); M_checkRetC(retc, hthread_dev_close);
}