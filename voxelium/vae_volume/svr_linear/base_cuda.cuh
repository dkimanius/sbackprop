#pragma once
#ifndef SVR_LINEAR_BASE_CUDA_H
#define SVR_LINEAR_BASE_CUDA_H

#include <torch/script.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

#include "vae_volume/svr_linear/base.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor. In ", __FILE__, ":", __LINE__)

#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CUDA_ERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__
inline
bool thread_index_expand(const size_t dim0, const size_t dim1,
                               size_t &idx0,      size_t &idx1)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    idx0 = index / dim1;
    idx1 = index % dim1;
    return idx0 < dim0;
}

__device__
inline
bool thread_index_expand(const size_t dim0, const size_t dim1, const size_t dim2,
                               size_t &idx0,      size_t &idx1,      size_t &idx2)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    idx0 = index / (dim1 * dim2);
    idx1 = (index / dim2) % dim1;
    idx2 = index % dim2;
    return idx0 < dim0;
}

__device__
inline
bool thread_index_expand(const size_t dim0, const size_t dim1, const size_t dim2, const size_t dim3,
                               size_t &idx0,      size_t &idx1,      size_t &idx2,      size_t &idx3)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    idx0 = index / (dim1 * dim2 * dim3);
    idx1 = (index / (dim2 * dim3)) % dim1;
    idx2 = (index / dim3) % dim2;
    idx3 = index % dim3;
    return idx0 < dim0;
}

template <typename accessor_t>
__device__
inline
size_t accessor_index_collapse(const accessor_t &a, size_t i0, size_t i1)
{
    return i0 * a.stride(0) + i1;
}

template <typename accessor_t>
__device__
inline
size_t accessor_index_collapse(const accessor_t &a, size_t i0, size_t i1, size_t i2)
{
    return i0 * a.stride(0) + i1 * a.stride(1) + i2;
}

template <typename accessor_t>
__device__
inline
size_t accessor_index_collapse(const accessor_t &a, size_t i0, size_t i1, size_t i2, size_t i3)
{
    return i0 * a.stride(0) + i1 * a.stride(1) + i2 * a.stride(2) + i3;
}

#endif // SVR_LINEAR_BASE_CUDA_H