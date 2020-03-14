#include "oneflow/customized/kernels/where_kernel.h"

namespace oneflow {

namespace {

template<typename T, typename CondT>
__global__ void CudaWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                          T* out) {
  DoWhere(elem_cnt, cond, lhs, rhs, out);
}

}  // namespace

template<typename T, typename CondT>
struct WhereFunctor<DeviceType::kGPU, T, CondT> {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const {
    CudaWhere<T, CondT>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, cond, lhs, rhs, out);
  }
};

}  // namespace oneflow