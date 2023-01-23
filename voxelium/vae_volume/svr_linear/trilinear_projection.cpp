#include "vae_volume/svr_linear/trilinear_projection.h"

torch::Tensor trilinear_projection_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rot_matrix,
    torch::Tensor grid2d_coord,
    torch::Tensor grid3d_index,
    const int max_r
)
{
    const int batch_size = input.size(0);
    const bool do_bias = bias.size(0) == weight.size(0);

    CHECK_SIZE_DIM1(grid2d_coord, 2)
    CHECK_SIZE_DIM0(rot_matrix, batch_size)
    CHECK_SIZE_DIM1(rot_matrix, 3)
    CHECK_SIZE_DIM2(rot_matrix, 3)

    auto output = torch::zeros(
        {batch_size, grid2d_coord.size(0), 2},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(true)
    );

    if (input.device().type() == torch::kCPU)
    {
        trilinear_projection_forward_cpu(
            /*grid2d_coord*/ grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*rot_matrix*/ rot_matrix,
            /*input*/ input,
            /*output*/ output,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias
        );
    }
    else if (input.device().type() == torch::kCUDA)
    {
        trilinear_projection_forward_cuda(
            /*grid2d_coord*/ grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*rot_matrix*/ rot_matrix,
            /*input*/ input,
            /*output*/ output,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias
        );
    }
    else
        throw std::logic_error("Support for device not implemented");

    return output;
}


std::vector<torch::Tensor> trilinear_projection_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rot_matrix,
    torch::Tensor input_spectral_weight,
    torch::Tensor grad_output,
    torch::Tensor grid2d_coord,
    torch::Tensor grid3d_index,
    bool sparse_grad,
    const int max_r
)
{
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int points_count = grid2d_coord.size(0);
    const bool do_bias = bias.size(0) == weight.size(0);

    CHECK_SIZE_DIM0(grid2d_coord, points_count)
    CHECK_SIZE_DIM1(grid2d_coord, 2)
    CHECK_SIZE_DIM0(rot_matrix, batch_size)
    CHECK_SIZE_DIM1(rot_matrix, 3)
    CHECK_SIZE_DIM2(rot_matrix, 3)

    CHECK_DTYPE(weight, input.dtype())
    CHECK_DTYPE(bias, input.dtype())
    CHECK_DTYPE(rot_matrix, input.dtype())
    CHECK_DTYPE(grad_output, input.dtype())
    CHECK_DTYPE(grid2d_coord, input.dtype())

    TORCH_CHECK(input_spectral_weight.size(0) == 0 || input_spectral_weight.size(0) > max_r,
        "input_spectral_weight.size(0) bad size (", input_spectral_weight.size(0), " <= ",
        max_r, "). In ", __FILE__, ":", __LINE__)

    torch::Tensor grad_weight, grad_bias, grad_weight_index;

    if (sparse_grad)
    {
        grad_weight = torch::zeros(
            {batch_size, points_count * 8, weight.size(1), 2},
            torch::TensorOptions()
                .dtype(input.dtype())
                .layout(torch::kStrided)
                .device(input.device())
                .requires_grad(false)
        );

        if (do_bias)
            grad_bias = torch::zeros(
                {batch_size, points_count * 8, 2},
                torch::TensorOptions()
                    .dtype(input.dtype())
                    .layout(torch::kStrided)
                    .device(input.device())
                    .requires_grad(false)
            );
       else
            grad_bias = torch::empty(
                {0, 0, 0},
                torch::TensorOptions()
                    .dtype(input.dtype())
                    .layout(torch::kStrided)
                    .device(input.device())
                    .requires_grad(false)
            );

        grad_weight_index = torch::zeros(
            {batch_size, points_count * 8},
            torch::TensorOptions()
                .dtype(torch::kInt64)
                .layout(torch::kStrided)
                .device(input.device())
                .requires_grad(false)
        );
    }
    else
    {
        grad_weight = torch::zeros_like(weight);
        grad_bias = torch::zeros_like(bias);
        grad_weight_index = torch::empty(0,
            torch::TensorOptions()
                .dtype(input.dtype())
                .layout(torch::kStrided)
                .device(input.device())
                .requires_grad(false)
        );
    }

    auto grad_input = torch::zeros(
        {batch_size, input_size},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(false)
    );

    auto grad_rot_matrix = torch::zeros(
        {batch_size, 3, 3},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(false)
    );

    if (input.device().type() == torch::kCPU)
    {
        trilinear_projection_backward_cpu(
            /*grid2d_coord*/grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*grad_weight_index*/ grad_weight_index,
            /*rot_matrix*/ rot_matrix,
            /*input_spectral_weight*/ input_spectral_weight,
            /*input*/ input,
            /*grad_output*/ grad_output,
            /*grad_weight*/ grad_weight,
            /*grad_bias*/ grad_bias,
            /*grad_input*/ grad_input,
            /*grad_rot_matrix*/ grad_rot_matrix,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias,
            /*do_rot_matrix_grad*/ rot_matrix.requires_grad(),
            /*sparse_grad*/ sparse_grad,
            /*do_spectral_weighting*/ input_spectral_weight.size(0) > 0
        );
    }
    else if (input.device().type() == torch::kCUDA)
    {
        trilinear_projection_backward_cuda(
            /*grid2d_coord*/grid2d_coord,
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*grad_weight_index*/ grad_weight_index,
            /*rot_matrix*/ rot_matrix,
            /*input_spectral_weight*/ input_spectral_weight,
            /*input*/ input,
            /*grad_output*/ grad_output,
            /*grad_weight*/ grad_weight,
            /*grad_bias*/ grad_bias,
            /*grad_input*/ grad_input,
            /*grad_rot_matrix*/ grad_rot_matrix,
            /*max_r2*/ (int) max_r * max_r,
            /*init_offset*/ (int) grid3d_index.size(0)/2,
            /*do_bias*/ do_bias,
            /*do_rot_matrix_grad*/ rot_matrix.requires_grad(),
            /*sparse_grad*/ sparse_grad,
            /*do_spectral_weighting*/ input_spectral_weight.size(0) > 0
        );
    }
    else
        throw std::logic_error("Support for device not implemented");

    return {grad_input, grad_weight_index, grad_weight, grad_bias, grad_rot_matrix};
}
