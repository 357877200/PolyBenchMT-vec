include make.conf

# 定义 libvm 路径
LIBVM_INCLUDE_PATH := $(LIBVM_ROOT_PATH)/include 
LIBVM_LIB_PATH     := $(LIBVM_ROOT_PATH)/lib

HTHREADS_INCLUDE_PATH := ${HTHREADS_ROOT_PATH}/include
HTHREADS_LIB_PATH := ${HTHREADS_ROOT_PATH}/lib

HOS_CC := gcc
HOS_LDFLAGS := -L${HTHREADS_LIB_PATH} -lhthread_host -lm -lpthread
HOS_CFLAGS := -Wall -I${HTHREADS_INCLUDE_PATH}

DEV_CC_BIN_PATH := ${DEV_CC_ROOT_PATH}/bin
DEV_CC := ${DEV_CC_BIN_PATH}/MT-3000-gcc
DEV_LD := ${DEV_CC_BIN_PATH}/MT-3000-ld
DEV_MAKEDAT := ${DEV_CC_BIN_PATH}/MT-3000-makedat
DEV_CC_INCLUDE_PATH := ${DEV_CC_ROOT_PATH}/include
DEV_CC_LIB_PATH := ${DEV_CC_ROOT_PATH}/lib

# ===================================================================
# ✅ 修改 1: 添加 libvm 头文件路径
# ===================================================================
DEV_INCLUDE_PATH := ${DEV_CC_INCLUDE_PATH} ${HTHREADS_INCLUDE_PATH} ${LIBVM_INCLUDE_PATH}
DEV_INCLUDE_FLAGS := $(foreach i, ${DEV_INCLUDE_PATH}, -I${i})
DEV_WALLFLAGS := -Wall -Wno-attributes -Wno-unused-function
DEV_CFLAGS := $(DEV_WALLFLAGS) -O2 -fenable-m3000 -ffunction-sections \
	-flax-vector-conversions ${DEV_INCLUDE_FLAGS}

# ===================================================================
# ✅ 修改 2: 添加 libvm 库路径，并在链接时加入 -lvm
# ===================================================================
DEV_LIBRARY_PATH := ${DEV_CC_LIB_PATH} ${HTHREADS_LIB_PATH} ${LIBVM_LIB_PATH}
DEV_LDFLAGS := $(foreach i, ${DEV_LIBRARY_PATH}, -L${i})
DEV_LDFLAGS += --gc-sections -T$(HTHREADS_LIB_PATH)/dsp.lds -lhthread_device -lvm \
	${DEV_CC_LIB_PATH}/slib3000.a  \
	${DEV_CC_ROOT_PATH}/lib/vlib3000.a

# 可选：导出主机端库路径（不影响 DSP 编译，仅用于运行 hos 程序）
export LD_LIBRARY_PATH=/thfs3/software/programming_env/mt3000_programming_env/third-party-lib:$LD_LIBRARY_PATH

# -------------------------
# 自动获取设备端源文件(.dev.c)和对应目标
DEV_SRCS := $(shell find operators -name '*.dev.c')
DEV_OBJS := $(DEV_SRCS:.dev.c=.dev.o)
DEV_OUTS := $(DEV_SRCS:.dev.c=.dev.out)
DEV_DATS := $(DEV_SRCS:.dev.c=.dev.dat)

# 自动获取主机端源文件(.hos.c)和对应目标
HOS_SRCS := $(shell find tests -name '*.hos.c')
HOS_BINS := $(HOS_SRCS:.hos.c=)

# 默认目标
.PHONY: ALL
ALL: all_device all_host

# 设备端目标
.PHONY: all_device
all_device: $(DEV_DATS)

# 主机端目标
.PHONY: all_host
all_host: $(HOS_BINS)

# 通用规则：编译 .dev.c -> .dev.o
%.dev.o: %.dev.c
	$(DEV_CC) -c $(DEV_CFLAGS) $< -o $@

# 链接 .dev.o -> .dev.out
%.dev.out: %.dev.o
	$(DEV_LD) $^ $(DEV_LDFLAGS) -o $@

# 生成 .dev.dat
%.dev.dat: %.dev.out
	$(DEV_MAKEDAT) -J $<

# 主机端编译
%: %.hos.c
	$(HOS_CC) -o $@ $< $(HOS_CFLAGS) $(HOS_LDFLAGS)

.PHONY: clean
clean:
	rm -rf $(DEV_OBJS) $(DEV_OUTS) $(DEV_DATS)
	rm -rf $(HOS_BINS)