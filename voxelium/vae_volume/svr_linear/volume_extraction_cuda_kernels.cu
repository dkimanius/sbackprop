#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/macros/Macros.h>

#include "vae_volume/svr_linear/base_cuda.cuh"
#include "vae_volume/svr_linear/volume_extraction_cuda_kernels.h"


template <typename scalar_t, typename accscaler_t, bool do_bias>
__global__ void volume_extraction_forward_cuda_kernel(
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> output,
    const int offset,
    const int max_r2
)
{
    size_t b, z, y, x;
    if (thread_index_expand(output.size(0), output.size(1), output.size(2), output.size(3), b, z, y, x))
    {
        const long zp = z - (output.size(1) - 1) / 2;
        const long yp = y - (output.size(2) - 1) / 2;
        const long xp = x;

        if (xp*xp + yp*yp + zp*zp <= max_r2)
        {
            const long i = grid3d_index[z+offset][y+offset][x];

            for (int c = 0; c < 2; c++) // Over real and imaginary
            {
                accscaler_t v(0);
                for (int j = 0; j < input.size(1); j ++)
                    v += weight[i][j][c] * input[b][j];
                if (do_bias)
                    v += bias[i][c];
                output[b][z][y][x][c] = (scalar_t) v;
            }
        }
    }
}

void volume_extraction_forward_cuda(
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const bool do_bias
)
{
    CHECK_CUDA_INPUT(grid3d_index)
    CHECK_CUDA_INPUT(weight)
    CHECK_CUDA_INPUT(bias)
    CHECK_CUDA_INPUT(input)
    CHECK_CUDA_INPUT(output)

    const int offset = (grid3d_index.size(0) - output.size(1)) / 2;

    const int deviceId = input.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));

    const dim3 threads(512);
    const dim3 blocks(
        (
            output.size(0) *
            output.size(1) *
            output.size(2) *
            output.size(3) +
            threads.x - 1
        ) / threads.x
    );

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                input.scalar_type(),
                "volume_extraction_forward_cuda_kernel",
                [&] {
                    using accscalar_t = at::acc_type<scalar_t, true>;
                    volume_extraction_forward_cuda_kernel
                    <scalar_t, accscalar_t, decltype(Bargs)::value...>
                    <<<blocks, threads, 0, stream>>>(
                        /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                        /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*output*/ output.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        /*offset*/ offset,
                        /*max_r2*/ max_r2
                    );
                }
            );
        }
    );

#ifdef DEBUG
    CUDA_ERRCHK(cudaPeekAtLastError());
    CUDA_ERRCHK(cudaDeviceSynchronize());
#endif
}

template <typename scalar_t, bool do_bias, bool do_input_grad, bool do_spectral_weighting>
__global__ void volume_extraction_backward_cuda_kernel(
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input_spectral_weight,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_weight,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    const int offset,
    const int max_r2,
    const size_t grad_weight_numel,
    const size_t grad_bias_numel,
    const size_t grad_input_numel
)
{
    size_t b, z, y, x;
    if (
        thread_index_expand(
            grad_output.size(0),
            grad_output.size(1),
            grad_output.size(2),
            grad_output.size(3),
            b, z, y, x
        )
    )
    {
        const long zp = z - (grad_output.size(1) - 1) / 2;
        const long yp = y - (grad_output.size(2) - 1) / 2;
        const long xp = x;

        scalar_t r = xp*xp + yp*yp + zp*zp;
        if (r <= max_r2)
        {
            if (do_spectral_weighting)
                r = input_spectral_weight[(int) sqrt((float) r)];

            const long i = grid3d_index[z+offset][y+offset][x];
            for (int c = 0; c < 2; c++) // Over real and imaginary
            {
                for (int j = 0; j < input.size(1); j ++)
                {
                    if (do_input_grad)
                        at::native::fastAtomicAdd(
                            grad_input.data(),
                            accessor_index_collapse(grad_input, b, j),
                            grad_input_numel,
                            do_spectral_weighting ?
                                grad_output[b][z][y][x][c] * weight[i][j][c] * r :
                                grad_output[b][z][y][x][c] * weight[i][j][c],
                            true
                        );

                    at::native::fastAtomicAdd(
                        grad_weight.data(),
                        accessor_index_collapse(grad_weight, i, j, c),
                        grad_weight_numel,
                        grad_output[b][z][y][x][c] * input[b][j],
                        true
                    );
                }

                if (do_bias)
                    at::native::fastAtomicAdd(
                        grad_bias.data(),
                        accessor_index_collapse(grad_bias, i, c),
                        grad_bias_numel,
                        grad_output[b][z][y][x][c],
                        true
                    );
            }
        }
    }
}


void volume_extraction_backward_cuda(
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor input_spectral_weight,
    const torch::Tensor input,
    const torch::Tensor grad_output,
    torch::Tensor grad_weight,
    torch::Tensor grad_bias,
    torch::Tensor grad_input,
    const int max_r2,
    const bool do_bias,
    const bool do_input_grad,
    const bool do_spectral_weighting
)
{
    CHECK_CUDA_INPUT(grid3d_index)
    CHECK_CUDA_INPUT(weight)
    CHECK_CUDA_INPUT(bias)
    CHECK_CUDA_INPUT(input_spectral_weight)
    CHECK_CUDA_INPUT(input)
    CHECK_CUDA_INPUT(grad_output)
    CHECK_CUDA_INPUT(grad_weight)
    CHECK_CUDA_INPUT(grad_bias)
    CHECK_CUDA_INPUT(grad_input)

    const int offset = (grid3d_index.size(0) - grad_output.size(1)) / 2;
    
    const int deviceId = input.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));

    const dim3 threads(512);
    const dim3 blocks(
        (
            grad_output.size(0) *
            grad_output.size(1) *
            grad_output.size(2) *
            grad_output.size(3) +
            threads.x - 1
        ) / threads.x
    );

    std::array<bool, 3> bargs={{do_bias, do_input_grad, do_spectral_weighting}};
    dispatch_bools<3>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                input.scalar_type(),
                "volume_extraction_backward_cuda_kernel",
                [&] {
                    volume_extraction_backward_cuda_kernel<scalar_t, decltype(Bargs)::value...>
                    <<<blocks, threads, 0, stream>>>(
                        /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                        /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*input_spectral_weight*/ input_spectral_weight.packed_accessor32
                            <scalar_t, 1, torch::RestrictPtrTraits>(),
                        /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*grad*/ grad_output.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                        /*grad_weight*/ grad_weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*grad_bias*/ grad_bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*grad_input*/ grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*offset*/ offset,
                        /*max_r2*/ max_r2,
                        /*grad_weight_numel*/ grad_weight.numel(),
                        /*grad_bias_numel*/ grad_bias.numel(),
                        /*grad_input_numel*/ grad_input.numel()
                    );
                }
            );
        }
    );

#ifdef DEBUG
    CUDA_ERRCHK(cudaPeekAtLastError());
    CUDA_ERRCHK(cudaDeviceSynchronize());
#endif
}
