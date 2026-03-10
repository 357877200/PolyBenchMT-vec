[English](README.md) | 中文说明
# PolyBenchMT-vec

**PolyBenchMT-vec** 是 [PolyBenchMT](https://github.com/Forgoys/PolyBenchMT) 的增强版本。  
PolyBenchMT 本身是原始 [PolyBench](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) 基准测试套件针对 **MT-3000 平台**（基于 hthreads）的移植与优化版本。

本项目在此基础上新增两大关键特性：

1. **显式向量化支持**：  
   所有计算算子（如 `gemm`、`convolution`、`atax` 等）均已添加向量化指令，以充分利用 MT-3000 处理器的 SIMD 单元，显著提升数据级并行性能。

2. **任意规模负载支持**：  
   主机端代码经过重构，**不再依赖编译时固定的数组大小**，而是支持运行时指定任意问题规模，便于开展细粒度的性能扩展性分析。

> ✅ 完全兼容 PolyBenchMT 在 MT-3000 上的执行模型  
> ✅ 在保留 hthreads 多线程并行的同时，新增线程内向量化并行  
> ✅ 适用于评估多核平台上“线程级 + 向量级”混合并行的综合性能

本项目继承了 PolyBenchMT 对 MT-3000 平台的所有优化，并通过集成显式向量化进一步扩展了其性能评估能力。
