#include "vae_volume/svr_linear/trilinear_projection_cpu_kernels.h"

template <typename scalar_t>
inline void _helper_rotate_coordinates(
    const torch::TensorAccessor<scalar_t, 2> rot_matrix,
    const scalar_t x, const scalar_t y,
    scalar_t &xp, scalar_t &yp, scalar_t &zp
)
{
    xp = rot_matrix[0][0] * x + rot_matrix[1][0] * y;
    yp = rot_matrix[0][1] * x + rot_matrix[1][1] * y;
    zp = rot_matrix[0][2] * x + rot_matrix[1][2] * y;
}


template <typename scalar_t>
inline scalar_t _helper_cube_interpolation_coordinates(
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


template <typename scalar_t, bool do_bias>
void trilinear_projection_forward_cpu_kernel(
    const torch::TensorAccessor<scalar_t, 2> grid2d_coord,
    const torch::TensorAccessor<long, 3> grid3d_index,
    const torch::TensorAccessor<scalar_t, 3> weight,
    const torch::TensorAccessor<scalar_t, 2> bias,
    const torch::TensorAccessor<scalar_t, 3> rot_matrix,
    const torch::TensorAccessor<scalar_t, 2> input,
    torch::TensorAccessor<scalar_t, 3> output,
    const int max_r2,
    const int init_offset
)
{
    int xs[2], ys[2], zs[2];
    scalar_t r, xp, yp, zp, fx, fy, fz;
    for (long b = 0; b < input.size(0); b ++)
    {
        for (long i = 0; i < grid2d_coord.size(0); i ++)
        {
            _helper_rotate_coordinates<scalar_t>(
                rot_matrix[b], grid2d_coord[i][0], grid2d_coord[i][1], xp, yp, zp
            );

            if (xp*xp + yp*yp + zp*zp <= max_r2)
            {
                scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                    xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
                );

                for (int c = 0; c < 2; c ++) // Over real and imaginary
                {
                    // Order: 0=000 1=001 2=010 3=011 4=100 5=101 6=110 7=111
                    scalar_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                    for (int k = 0; k < 8; k ++) // Over local cube vertices
                    {
                        long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];

                        for (int j = 0; j < input.size(1); j ++) // Over input vector
                            v[k] += weight[index][j][c] * input[b][j];

                        if (do_bias)
                            v[k] += bias[index][c];
                    }

                    // Set the interpolated value in the 2D output array
                    const scalar_t dx00 = LIN_INTERP(fx, v[0], v[1]);
                    const scalar_t dx10 = LIN_INTERP(fx, v[2], v[3]);
                    const scalar_t dx01 = LIN_INTERP(fx, v[4], v[5]);
                    const scalar_t dx11 = LIN_INTERP(fx, v[6], v[7]);
                    const scalar_t dxy0 = LIN_INTERP(fy, dx00, dx10);
                    const scalar_t dxy1 = LIN_INTERP(fy, dx01, dx11);
                    if (c == 1) // Only flip sign of the imaginary component (complex conjugate)
                        output[b][i][c] = conj * LIN_INTERP(fz, dxy0, dxy1);
                    else
                        output[b][i][c] = LIN_INTERP(fz, dxy0, dxy1);
                }
            } // If < max_r
        } // 2D Point
    } // Batch
}

void trilinear_projection_forward_cpu(
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
    CHECK_CPU_INPUT(grid2d_coord)
    CHECK_CPU_INPUT(grid3d_index)
    CHECK_CPU_INPUT(weight)
    CHECK_CPU_INPUT(bias)
    CHECK_CPU_INPUT(rot_matrix)
    CHECK_CPU_INPUT(input)
    CHECK_CPU_INPUT(output)

    std::array<bool, 1> bargs={{do_bias}};
    dispatch_bools<1>{}(
        bargs,
        [&](auto...Bargs) {
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "trilinear_projection_forward_cpu_kernel",
                [&] {
                    trilinear_projection_forward_cpu_kernel<scalar_t, decltype(Bargs)::value...>(
                        /*grid2d_coord*/ grid2d_coord.accessor<scalar_t, 2>(),
                        /*grid3d_index*/ grid3d_index.accessor<long, 3>(),
                        /*weight*/ weight.accessor<scalar_t, 3>(),
                        /*bias*/ bias.accessor<scalar_t, 2>(),
                        /*rot_matrix*/ rot_matrix.accessor<scalar_t, 3>(),
                        /*input*/ input.accessor<scalar_t, 2>(),
                        /*output*/ output.accessor<scalar_t, 3>(),
                        /*max_r2*/ max_r2,
                        /*init_offset*/ init_offset
                    );
                }
            );
        }
    );
}

template <typename scalar_t, bool do_bias, bool do_grad_rot_matrix, bool do_spectral_weighting>
void trilinear_projection_sparse_backward_cpu_kernel(
    const torch::TensorAccessor<scalar_t, 2> grid2d_coord,
    const torch::TensorAccessor<long, 3> grid3d_index,
    const torch::TensorAccessor<scalar_t, 3> weight,
    const torch::TensorAccessor<scalar_t, 2> bias,
    torch::TensorAccessor<long, 2> grad_weight_index,
    const torch::TensorAccessor<scalar_t, 3> rot_matrix,
    const torch::TensorAccessor<scalar_t, 1> input_spectral_weight,
    const torch::TensorAccessor<scalar_t, 2> input,
    const torch::TensorAccessor<scalar_t, 3> grad_output,
    torch::TensorAccessor<scalar_t, 4> grad_weight,
    torch::TensorAccessor<scalar_t, 3> grad_bias,
    torch::TensorAccessor<scalar_t, 2> grad_input,
    torch::TensorAccessor<scalar_t, 3> grad_rot_matrix,
    const int max_r2,
    const int init_offset
)
{
    int xs[2], ys[2], zs[2];
    scalar_t r, xp, yp, zp, fx, fy, fz;
    for (long b = 0; b < input.size(0); b ++)
    {
        for (long i = 0; i < grid2d_coord.size(0); i ++)
        {
            scalar_t x(grid2d_coord[i][0]), y(grid2d_coord[i][1]);
            _helper_rotate_coordinates<scalar_t>(
                rot_matrix[b], x, y, xp, yp, zp
            );

            r = xp*xp + yp*yp + zp*zp;
            if (r <= max_r2)
            {
                if (do_spectral_weighting)
                    r = input_spectral_weight[(int) std::sqrt(r)];

                scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                    xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
                );

                scalar_t fxs[] = {(scalar_t) 1.0 - fx, (scalar_t) fx};
                scalar_t fys[] = {(scalar_t) 1.0 - fy, (scalar_t) fy};
                scalar_t fzs[] = {(scalar_t) 1.0 - fz, (scalar_t) fz};

                /* 'C' is the current cost,
                   a 'voxel' means sum(weight[j]*input[j]),
                   'out' is the linear combination of the 8 voxels  */
                scalar_t dC_dout[] = {grad_output[b][i][0], conj * grad_output[b][i][1]};

                // dout_dp means [d(out)/d(xp), d(out)/d(yp), d(out)/d(zp)]
                scalar_t dout_dp[3][2] = {{0, 0}, {0, 0}, {0, 0}};
                long i_offset = 8 * i;

                for (int k = 0; k < 8; k ++) // Over cube vertices
                {
                    const long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];
                    grad_weight_index[b][i_offset+k] = index;

                    for (int c = 0; c < 2; c ++) // Over real and imaginary
                    {
                        scalar_t voxel_value(0);
                        const scalar_t dC_dvoxel = dC_dout[c] * fzs[k/4] * fys[(k/2)%2] * fxs[k%2];

                        for (int j = 0; j < input.size(1); j ++)// Over input vector
                        {
                            scalar_t wkj = weight[index][j][c];

                            /* voxel = sum(weight[j]*input[j])
                               =>  d(voxel)/d(weight[j]) = input[j]
                                   d(voxel)/d(input[j]) = weight[j]  */

                            grad_weight[b][i_offset+k][j][c] = dC_dvoxel * input[b][j];
                            grad_input[b][j] += do_spectral_weighting ? dC_dvoxel * wkj * r : dC_dvoxel * wkj;
                        }

                        if (do_bias)
                            grad_bias[b][i_offset+k][c] = dC_dvoxel;
                    }
                }
            } // If < max_r
        } // 2D Point
    } // Batch
}

template <typename scalar_t, bool do_bias, bool do_grad_rot_matrix, bool do_spectral_weighting>
void trilinear_projection_dense_backward_cpu_kernel(
    const torch::TensorAccessor<scalar_t, 2> grid2d_coord,
    const torch::TensorAccessor<long, 3> grid3d_index,
    const torch::TensorAccessor<scalar_t, 3> weight,
    const torch::TensorAccessor<scalar_t, 2> bias,
    const torch::TensorAccessor<scalar_t, 3> rot_matrix,
    const torch::TensorAccessor<scalar_t, 1> input_spectral_weight,
    const torch::TensorAccessor<scalar_t, 2> input,
    const torch::TensorAccessor<scalar_t, 3> grad_output,
    torch::TensorAccessor<scalar_t, 3> grad_weight,
    torch::TensorAccessor<scalar_t, 2> grad_bias,
    torch::TensorAccessor<scalar_t, 2> grad_input,
    torch::TensorAccessor<scalar_t, 3> grad_rot_matrix,
    const int max_r2,
    const int init_offset
)
{
    int xs[2], ys[2], zs[2];
    scalar_t r, xp, yp, zp, fx, fy, fz;
    for (long b = 0; b < input.size(0); b ++)
    {
        for (long i = 0; i < grid2d_coord.size(0); i ++)
        {
            scalar_t x(grid2d_coord[i][0]), y(grid2d_coord[i][1]);
            _helper_rotate_coordinates<scalar_t>(
                rot_matrix[b], x, y, xp, yp, zp
            );

            r = xp*xp + yp*yp + zp*zp;
            if (r <= max_r2)
            {
                if (do_spectral_weighting)
                    r = input_spectral_weight[(int) std::sqrt(r)];

                scalar_t conj = _helper_cube_interpolation_coordinates<scalar_t>(
                    xp, yp, zp, init_offset, xs, ys, zs, fx, fy, fz
                );

                scalar_t fxs[] = {(scalar_t) 1.0 - fx, (scalar_t) fx};
                scalar_t fys[] = {(scalar_t) 1.0 - fy, (scalar_t) fy};
                scalar_t fzs[] = {(scalar_t) 1.0 - fz, (scalar_t) fz};

                /* 'C' is the current cost,
                   a 'voxel' means sum(weight[j]*input[j]),
                   'out' is the linear combination of the 8 voxels  */
                scalar_t dC_dout[] = {grad_output[b][i][0], conj * grad_output[b][i][1]};

                // dout_dp means [d(out)/d(xp), d(out)/d(yp), d(out)/d(zp)]
                scalar_t dout_dp[3][2] = {{0, 0}, {0, 0}, {0, 0}};

                for (int k = 0; k < 8; k++) // over cube vertices
                {
                    const long index = grid3d_index[zs[k/4]] [ys[(k/2)%2]] [xs[k%2]];

                    for (int c = 0; c < 2; c++) // Over real and imaginary
                    {
                        scalar_t voxel_value(0);
                        const scalar_t dC_dvoxel = dC_dout[c] * fzs[k/4] * fys[(k/2)%2] * fxs[k%2];

                        for (int j = 0; j < input.size(1); j ++)
                        {
                            scalar_t wkj = weight[index][j][c];

                            /* voxel = sum(weight[j]*input[j])
                               =>  d(voxel)/d(weight[j]) = input[j]
                                   d(voxel)/d(input[j]) = weight[j]  */

                            grad_weight[index][j][c] += dC_dvoxel * input[b][j];
                            grad_input[b][j] += do_spectral_weighting ? dC_dvoxel * wkj * r : dC_dvoxel * wkj;
                        }

                        if (do_bias)
                            grad_bias[index][c] += dC_dvoxel;
                    }
                }
            } // If < max_r
        } // 2D Point
    } // Batch
}


void trilinear_projection_backward_cpu(
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
    CHECK_CPU_INPUT(grid2d_coord)
    CHECK_CPU_INPUT(grid3d_index)
    CHECK_CPU_INPUT(weight)
    CHECK_CPU_INPUT(bias)
    CHECK_CPU_INPUT(rot_matrix)
    CHECK_CPU_INPUT(input_spectral_weight)
    CHECK_CPU_INPUT(input)
    CHECK_CPU_INPUT(grad_output)
    CHECK_CPU_INPUT(grad_input)
    CHECK_CPU_INPUT(grad_rot_matrix)
    CHECK_CPU_INPUT(grad_weight_index)
    CHECK_CPU_INPUT(grad_weight)
    CHECK_CPU_INPUT(grad_bias)

    std::array<bool, 3> bargs={{do_bias, do_grad_rot_matrix, do_spectral_weighting}};
    if (sparse_grad)
    {
        dispatch_bools<3>{}(
            bargs,
            [&](auto...Bargs) {
                AT_DISPATCH_FLOATING_TYPES(
                    input.scalar_type(),
                    "trilinear_projection_sparse_backward_cpu_kernel",
                    [&] {
                        trilinear_projection_sparse_backward_cpu_kernel<scalar_t, decltype(Bargs)::value...>(
                            /*grid2d_coord*/ grid2d_coord.accessor<scalar_t, 2>(),
                            /*grid3d_index*/ grid3d_index.accessor<long, 3>(),
                            /*weight*/ weight.accessor<scalar_t, 3>(),
                            /*bias*/ bias.accessor<scalar_t, 2>(),
                            /*grad_weight_index*/ grad_weight_index.accessor<long, 2>(),
                            /*rot_matrix*/ rot_matrix.accessor<scalar_t, 3>(),
                            /*input_spectral_weight*/ input_spectral_weight.accessor<scalar_t, 1>(),
                            /*input*/ input.accessor<scalar_t, 2>(),
                            /*grad*/ grad_output.accessor<scalar_t, 3>(),
                            /*grad_weight*/ grad_weight.accessor<scalar_t, 4>(),
                            /*grad_bias*/ grad_bias.accessor<scalar_t, 3>(),
                            /*grad_input*/ grad_input.accessor<scalar_t, 2>(),
                            /*grad_rot_matrix*/ grad_rot_matrix.accessor<scalar_t, 3>(),
                            /*max_r2*/ max_r2,
                            /*init_offset*/ init_offset
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
                AT_DISPATCH_FLOATING_TYPES(
                    input.scalar_type(),
                    "trilinear_projection_dense_backward_cpu_kernel",
                    [&] {
                        trilinear_projection_dense_backward_cpu_kernel<scalar_t, decltype(Bargs)::value...>(
                            /*grid2d_coord*/ grid2d_coord.accessor<scalar_t, 2>(),
                            /*grid3d_index*/ grid3d_index.accessor<long, 3>(),
                            /*weight*/ weight.accessor<scalar_t, 3>(),
                            /*bias*/ bias.accessor<scalar_t, 2>(),
                            /*rot_matrix*/ rot_matrix.accessor<scalar_t, 3>(),
                            /*input_spectral_weight*/ input_spectral_weight.accessor<scalar_t, 1>(),
                            /*input*/ input.accessor<scalar_t, 2>(),
                            /*grad*/ grad_output.accessor<scalar_t, 3>(),
                            /*grad_weight*/ grad_weight.accessor<scalar_t, 3>(),
                            /*grad_bias*/ grad_bias.accessor<scalar_t, 2>(),
                            /*grad_input*/ grad_input.accessor<scalar_t, 2>(),
                            /*grad_rot_matrix*/ grad_rot_matrix.accessor<scalar_t, 3>(),
                            /*max_r2*/ max_r2,
                            /*init_offset*/ init_offset
                        );
                    }
                );
            }
        );
    }
}
