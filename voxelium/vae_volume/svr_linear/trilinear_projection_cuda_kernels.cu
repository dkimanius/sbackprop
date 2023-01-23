#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/macros/Macros.h>


#include "vae_volume/svr_linear/base_cuda.cuh"
#include "vae_volume/svr_linear/trilinear_projection_cuda_kernels.h"

#define FORWARD_BLOCK_SIZE 512
#define BACKWARD_BLOCK_SIZE 128

template <typename scalar_t>
__device__ inline scalar_t _helper_cube_interpolation_coordinates(
    scalar_t xp, scalar_t yp, scalar_t zp, const int init_offset,
    int xs[2], int ys[2], int zs[2],
    scalar_t &fx, scalar_t &fy, scalar_t &fz
)
{
    scalar_t conj = 1.; // Complex conjugate
    if (xp < 0) // Hermitian half only
    {
        conj = -1.;
        xp = -xp;
        yp = -yp;
        zp = -zp;
    }

    xs[0] = floor(xp);
    fx = xp - xs[0];
    xs[1] = xs[0] + 1;

    ys[0] = floor(yp);
    fy = yp - ys[0];
    ys[0] += init_offset;
    ys[1] = ys[0] + 1;

    zs[0] = floor(zp);
    fz = zp - zs[0];
    zs[0] += init_offset;
    zs[1] = zs[0] + 1;

    return conj;
}

template <typename scalar_t, typename accscalar_t, bool do_bias>
__global__ void trilinear_projection_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid2d_coord,
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rot_matrix,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output,
    const int max_r2,
    const int init_offset
)
{
    int xs[2], ys[2], zs[2];
    scalar_t xp, yp, zp, fx, fy, fz;
    size_t b, i;

    if (thread_index_expand(input.size(0), grid2d_coord.size(0), b, i))
    {
        scalar_t x(grid2d_coord[i][0]), y(grid2d_coord[i][1]);
        xp = rot_matrix[b][0][0] * x + rot_matrix[b][1][0] * y;
        yp = rot_matrix[b][0][1] * x + rot_matrix[b][1][1] * y;
        zp = rot_matrix[b][0][2] * x + rot_matrix[b][1][2] * y;

        if (xp*xp + yp*yp + zp*zp <= max_r2)
        {
            scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
            );

            for (int c = 0; c < 2; c ++) // Over real and imaginary
            {
                // Order: 0=000 1=001 2=010 3=011 4=100 5=101 6=110 7=111
                accscalar_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                for (int k = 0; k < 8; k ++) // Over local cube vertices
                {
                    long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];

                    for (int j = 0; j < input.size(1); j ++) // Over input vector
                        v[k] += weight[index][j][c] * input[b][j];

                    if (do_bias)
                        v[k] += bias[index][c];
                }

                // Set the interpolated value in the 2D output array
                const accscalar_t dx00 = LIN_INTERP(fx, v[0], v[1]);
                const accscalar_t dx10 = LIN_INTERP(fx, v[2], v[3]);
                const accscalar_t dx01 = LIN_INTERP(fx, v[4], v[5]);
                const accscalar_t dx11 = LIN_INTERP(fx, v[6], v[7]);
                const accscalar_t dxy0 = LIN_INTERP(fy, dx00, dx10);
                const accscalar_t dxy1 = LIN_INTERP(fy, dx01, dx11);
                if (c == 1) // Only flip sign of the imaginary component (complex conjugate)
                    output[b][i][c] = (scalar_t) conj * LIN_INTERP(fz, dxy0, dxy1);
                else
                    output[b][i][c] = (scalar_t) LIN_INTERP(fz, dxy0, dxy1);
            }
        } // If < max_r
    }
}

void trilinear_projection_forward_cuda(
    const torch::Tensor grid2d_coord,
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const torch::Tensor rot_matrix,
    const torch::Tensor input,
    torch::Tensor output,
    const int max_r2,
    const int init_offset,
    const bool do_bias
)
{
    CHECK_CUDA_INPUT(grid2d_coord)
    CHECK_CUDA_INPUT(grid3d_index)
    CHECK_CUDA_INPUT(weight)
    CHECK_CUDA_INPUT(bias)
    CHECK_CUDA_INPUT(rot_matrix)
    CHECK_CUDA_INPUT(input)
    CHECK_CUDA_INPUT(output)

    const int deviceId = input.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));

    const dim3 threads(FORWARD_BLOCK_SIZE);
    const dim3 blocks((input.size(0) * grid2d_coord.size(0) + threads.x - 1) / threads.x);

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                input.scalar_type(),
                "trilinear_projection_forward_cuda_kernel",
                [&] {
                    using accscalar_t = at::acc_type<scalar_t, true>;
                    trilinear_projection_forward_cuda_kernel
                    <scalar_t, accscalar_t, decltype(Bargs)::value...>
                    <<<blocks, threads, 0, stream>>>(
                        /*grid2d_coord*/ grid2d_coord.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                        /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*rot_matrix*/ rot_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        /*output*/ output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                        /*max_r2*/ max_r2,
                        /*init_offset*/ init_offset
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


template <typename scalar_t, typename accscalar_t, bool do_bias, bool do_grad_rot_matrix, bool do_spectral_weighting>
__global__ void trilinear_projection_backward_sparse_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid2d_coord,
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<long, 2, torch::RestrictPtrTraits> grad_weight_index,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rot_matrix,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input_spectral_weight,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weight,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_rot_matrix,
    const int max_r2,
    const int init_offset,
    const size_t grad_input_numel,
    const size_t grad_rot_matrix_numel
)
{
    int xs[2], ys[2], zs[2];
    scalar_t r, xp, yp, zp, fx, fy, fz;

    __shared__ accscalar_t s[BACKWARD_BLOCK_SIZE][2][3]; // Accumulative rotation matrix gradient

    for (int m = 0; m < 2; m++)
        for (int n = 0; n < 3; n++)
            s[threadIdx.y][m][n] = 0.;

    const size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < input.size(0) && i < grid2d_coord.size(0))
    {
        scalar_t x(grid2d_coord[i][0]), y(grid2d_coord[i][1]);
        xp = rot_matrix[b][0][0] * x + rot_matrix[b][1][0] * y;
        yp = rot_matrix[b][0][1] * x + rot_matrix[b][1][1] * y;
        zp = rot_matrix[b][0][2] * x + rot_matrix[b][1][2] * y;

        r = xp*xp + yp*yp + zp*zp;
        if (r <= max_r2)
        {
            if (do_spectral_weighting)
                r = input_spectral_weight[(int) sqrt((float) r)];

            scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
            );

            scalar_t fxs[] = {(scalar_t) 1.0 - fx, fx};
            scalar_t fys[] = {(scalar_t) 1.0 - fy, fy};
            scalar_t fzs[] = {(scalar_t) 1.0 - fz, fz};

            /* 'C' is the current cost,
               a 'voxel' means sum(weight[j]*input[j]),
               'out' is the linear combination of the 8 voxels  */
            scalar_t dC_dout[] = {grad_output[b][i][0], grad_output[b][i][1]};
            dC_dout[1] *= conj; // Make complex conjugate

            // dout_dp means [d(out)/d(xp), d(out)/d(yp), d(out)/d(zp)]
            accscalar_t dout_dp[3][2] = {{0, 0}, {0, 0}, {0, 0}};
            long i_offset = 8 * i;

            for (int k = 0; k < 8; k ++) // Over cube vertices
            {
                const long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];
                grad_weight_index[b][i_offset+k] = index;

                for (int c = 0; c < 2; c ++) // Over real and imaginary
                {
                    accscalar_t voxel_value(0);
                    const scalar_t dC_dvoxel = dC_dout[c] * fzs[k/4] * fys[(k/2)%2] * fxs[k%2];

                    for (int j = 0; j < input.size(1); j ++) // Over input vector
                    {
                        scalar_t wkj = weight[index][j][c];

                        /* voxel = sum(weight[j]*input[j])
                           =>  d(voxel)/d(weight[j]) = input[j]
                               d(voxel)/d(input[j]) = weight[j]  */

                        grad_weight[b][i_offset+k][j][c] = dC_dvoxel * input[b][j];
                        at::native::fastAtomicAdd(
                            grad_input.data(),
                            accessor_index_collapse(grad_input, b, j),
                            grad_input_numel,
                            do_spectral_weighting ? dC_dvoxel * wkj * r : dC_dvoxel * wkj,
                            true
                        );
                    }

                    if (do_bias)
                    {
                        grad_bias[b][i_offset+k][c] = dC_dvoxel;
                    }
                }
            }
        } // If < max_r
    }
}


template <typename scalar_t, typename accscalar_t, bool do_bias, bool do_grad_rot_matrix, bool do_spectral_weighting>
__global__ void trilinear_projection_backward_dense_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grid2d_coord,
    const torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> grid3d_index,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> bias,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> rot_matrix,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input_spectral_weight,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_weight,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_rot_matrix,
    const int max_r2,
    const int init_offset,
    const size_t grad_weight_numel,
    const size_t grad_bias_numel,
    const size_t grad_input_numel,
    const size_t grad_rot_matrix_numel
)
{
    int xs[2], ys[2], zs[2];
    scalar_t r, xp, yp, zp, fx, fy, fz;

    __shared__ accscalar_t s[BACKWARD_BLOCK_SIZE][2][3]; // Accumulative rotation matrix gradient

    for (int m = 0; m < 2; m++)
        for (int n = 0; n < 3; n++)
            s[threadIdx.y][m][n] = 0.;

    const size_t b = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < input.size(0) && i < grid2d_coord.size(0))
    {
        scalar_t x(grid2d_coord[i][0]), y(grid2d_coord[i][1]);
        xp = rot_matrix[b][0][0] * x + rot_matrix[b][1][0] * y;
        yp = rot_matrix[b][0][1] * x + rot_matrix[b][1][1] * y;
        zp = rot_matrix[b][0][2] * x + rot_matrix[b][1][2] * y;

        r = xp*xp + yp*yp + zp*zp;
        if (r <= max_r2)
        {
            if (do_spectral_weighting)
                r = input_spectral_weight[(int) sqrt((float) r)];
            scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
            );

            scalar_t fxs[] = {(scalar_t) 1.0 - fx, fx};
            scalar_t fys[] = {(scalar_t) 1.0 - fy, fy};
            scalar_t fzs[] = {(scalar_t) 1.0 - fz, fz};

            /* 'C' is the current cost,
               a 'voxel' means sum(weight[j]*input[j]),
               'out' is the linear combination of the 8 voxels  */
            scalar_t dC_dout[] = {grad_output[b][i][0], grad_output[b][i][1]};
            dC_dout[1] *= conj; // Make complex conjugate

            // dout_dp means [d(out)/d(xp), d(out)/d(yp), d(out)/d(zp)]
            accscalar_t dout_dp[3][2] = {{0, 0}, {0, 0}, {0, 0}};

            for (int k = 0; k < 8; k++) // over cube vertices
            {
                const long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];

                for (int c = 0; c < 2; c++) // Over real and imaginary
                {
                    accscalar_t voxel_value(0);
                    const scalar_t dC_dvoxel = dC_dout[c] * fzs[k/4] * fys[(k/2)%2] * fxs[k%2];

                    for (int j = 0; j < input.size(1); j ++)
                    {
                        scalar_t wkj = weight[index][j][c];

                        /* voxel = sum(weight[j]*input[j])
                           =>  d(voxel)/d(weight[j]) = input[j]
                               d(voxel)/d(input[j]) = weight[j]  */

                        at::native::fastAtomicAdd(
                            grad_weight.data(),
                            accessor_index_collapse(grad_weight, index, j, c),
                            grad_weight_numel,
                            dC_dvoxel * input[b][j],
                            true
                        );

                        at::native::fastAtomicAdd(
                            grad_input.data(),
                            accessor_index_collapse(grad_input, b, j),
                            grad_input_numel,
                            do_spectral_weighting ? dC_dvoxel * wkj * r : dC_dvoxel * wkj,
                            true
                        );
                    }

                    if (do_bias)
                    {
                        at::native::fastAtomicAdd(
                            grad_bias.data(),
                            accessor_index_collapse(grad_bias, index, c),
                            grad_bias_numel,
                            dC_dvoxel,
                            true
                        );
                    }
                }
            }
        } // If < max_r
    }
}

void trilinear_projection_backward_cuda(
    const torch::Tensor grid2d_coord,
    const torch::Tensor grid3d_index,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor grad_weight_index,
    const torch::Tensor rot_matrix,
    const torch::Tensor input_spectral_weight,
    const torch::Tensor input,
    const torch::Tensor grad_output,
    torch::Tensor grad_weight,
    torch::Tensor grad_bias,
    torch::Tensor grad_input,
    torch::Tensor grad_rot_matrix,
    const int max_r2,
    const int init_offset,
    const bool do_bias,
    const bool do_grad_rot_matrix,
    const bool sparse_grad,
    const bool do_spectral_weighting
)
{
    CHECK_CUDA_INPUT(grid2d_coord)
    CHECK_CUDA_INPUT(grid3d_index)
    CHECK_CUDA_INPUT(weight)
    CHECK_CUDA_INPUT(bias)
    CHECK_CUDA_INPUT(grad_weight_index)
    CHECK_CUDA_INPUT(rot_matrix)
    CHECK_CUDA_INPUT(input_spectral_weight)
    CHECK_CUDA_INPUT(input)
    CHECK_CUDA_INPUT(grad_output)
    CHECK_CUDA_INPUT(grad_input)
    CHECK_CUDA_INPUT(grad_rot_matrix)
    CHECK_CUDA_INPUT(grad_weight)
    CHECK_CUDA_INPUT(grad_bias)

    const int deviceId = input.device().index();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(deviceId);
    CUDA_ERRCHK(cudaSetDevice(deviceId));

    const dim3 threads(1, BACKWARD_BLOCK_SIZE);
    const dim3 blocks(
        (input.size(0) + threads.x - 1) / threads.x,
        (grid2d_coord.size(0) + threads.y - 1) / threads.y
    );

    std::array<bool, 3> bargs={{do_bias, do_grad_rot_matrix, do_spectral_weighting}};
    if (sparse_grad)
    {
        dispatch_bools<3>{}(
            bargs,
            [&](auto...Bargs) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(),
                    "trilinear_projection_backward_sparse_cuda_kernel",
                    [&] {
                        using accscalar_t = at::acc_type<scalar_t, true>;
                        trilinear_projection_backward_sparse_cuda_kernel
                        <scalar_t, accscalar_t, decltype(Bargs)::value...>
                        <<<blocks, threads, 0, stream>>>(
                            /*grid2d_coord*/ grid2d_coord.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                            /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grad_weight_index*/ grad_weight_index.packed_accessor32<long, 2, torch::RestrictPtrTraits>(),
                            /*rot_matrix*/ rot_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*input_spectral_weight*/ input_spectral_weight.packed_accessor32
                                <scalar_t, 1, torch::RestrictPtrTraits>(),
                            /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grad*/ grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*grad_weight*/ grad_weight.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                            /*grad_bias*/ grad_bias.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*grad_input*/ grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grad_rot_matrix*/ grad_rot_matrix.packed_accessor32
                                <scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*max_r2*/ max_r2,
                            /*init_offset*/ init_offset,
                            /*grad_input_numel*/ grad_input.numel(),
                            /*grad_rot_matrix_numel*/ grad_rot_matrix.numel()
                        );
                    }
                );
            }
        );
    }
    else
    {
        dispatch_bools<3>{}(
            bargs,
            [&](auto...Bargs) {
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                    input.scalar_type(),
                    "trilinear_projection_backward_dense_cuda_kernel",
                    [&] {
                        using accscalar_t = at::acc_type<scalar_t, true>;
                        trilinear_projection_backward_dense_cuda_kernel
                        <scalar_t, accscalar_t, decltype(Bargs)::value...>
                        <<<blocks, threads, 0, stream>>>(
                            /*grid2d_coord*/ grid2d_coord.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grid3d_index*/ grid3d_index.packed_accessor64<long, 3, torch::RestrictPtrTraits>(),
                            /*weight*/ weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*bias*/ bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*rot_matrix*/ rot_matrix.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*input_spectral_weight*/ input_spectral_weight.packed_accessor32
                                <scalar_t, 1, torch::RestrictPtrTraits>(),
                            /*input*/ input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grad*/ grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*grad_weight*/ grad_weight.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*grad_bias*/ grad_bias.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grad_input*/ grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                            /*grad_rot_matrix*/ grad_rot_matrix.packed_accessor32
                                <scalar_t, 3, torch::RestrictPtrTraits>(),
                            /*max_r2*/ max_r2,
                            /*init_offset*/ init_offset,
                            /*grad_weight_numel*/ grad_weight.numel(),
                            /*grad_bias_numel*/ grad_bias.numel(),
                            /*grad_input_numel*/ grad_input.numel(),
                            /*grad_rot_matrix_numel*/ grad_rot_matrix.numel()
                        );
                    }
                );
            }
        );
    }

#ifdef DEBUG
    CUDA_ERRCHK(cudaPeekAtLastError());
    CUDA_ERRCHK(cudaDeviceSynchronize());
#endif
}
