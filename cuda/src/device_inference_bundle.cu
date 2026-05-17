// Single TU bundle for MSVC/CUDA + CMake: kernels.cu calls model_forward (model.cu), which
// calls kv_cache_* (kv_cache.cu). Per-file nvcc --compile leaves those __device__ symbols
// unresolved unless every TU gets -rdc=true and a device link step — VS integration often
// does not. The Makefile avoids this by passing all .cu files on one nvcc link line; we get
// the same effect by including these sources here once.

#include "kv_cache.cu"
#include "model.cu"
#include "kernels.cu"
