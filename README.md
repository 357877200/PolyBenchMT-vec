# PolyBenchMT-vec

**PolyBenchMT-vec** is an enhanced version of [PolyBenchMT](https://github.com/Forgoys/PolyBenchMT).  
PolyBenchMT itself is a port and optimization of the original [PolyBench](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) benchmark suite for the **MT-3000 platform**, built on **hthreads**.

This project introduces two key enhancements:

1. **Explicit Vectorization Support**:  
   All computational kernels (e.g., `gemm`, `convolution`, `atax`) have been augmented with vectorization directives to fully exploit the SIMD units of the MT-3000 processor, significantly improving data-level parallelism and performance.

2. **Arbitrary Workload Scaling**:  
   The host-side code has been refactored to **eliminate compile-time fixed array sizes**. Users can now specify problem dimensions at runtime, enabling fine-grained performance analysis across a wide range of input scales.

> ✅ Fully compatible with PolyBenchMT’s execution model on MT-3000  
> ✅ Preserves hthreads-based multi-threading while adding intra-thread vectorization  
> ✅ Ideal for evaluating hybrid parallelism combining thread-level and vector-level concurrency

PolyBenchMT-vec inherits all platform-specific optimizations from PolyBenchMT for the MT-3000 architecture and extends its capability by integrating explicit vectorization for comprehensive performance evaluation.
