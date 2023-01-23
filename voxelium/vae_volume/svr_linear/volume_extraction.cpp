#include "vae_volume/svr_linear/volume_extraction.h"

torch::Tensor volume_extraction_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grid3d_index,
    const int max_r
)
{
    const int batch_size = input.size(0);
    const int img_side = max_r * 2 + 1;
    const bool do_bias = bias.size(0) == weight.size(0);

    auto output = torch::zeros(
        {batch_size, img_side, img_side, img_side / 2 + 1, 2},
        torch::TensorOptions()
            .dtype(input.dtype())
            .layout(torch::kStrided)
            .device(input.device())
            .requires_grad(true)
    );

    if (input.device().type() == torch::kCPU)
    {
        volume_extraction_forward_cpu(
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*input*/ input,
            /*output*/ output,
            /*max_r2*/ (int) max_r * max_r,
            /*do_bias*/ do_bias
        );
    }
    else if (input.device().type() == torch::kCUDA)
    {
        volume_extraction_forward_cuda(
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*input*/ input,
            /*output*/ output,
            /*max_r2*/ (int) max_r * max_r,
            /*do_bias*/ do_bias
        );
    }
    else
        throw std::logic_error("Support for device not implemented");

    return output;
}


std::vector<torch::Tensor> volume_extraction_backward(
    torch::Tensor input_spectral_weight,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grad_output,
    torch::Tensor grid3d_index
)
{
    const int img_side = grad_output.size(1);
    const int max_r = (img_side - 1) / 2;
    const bool do_bias = bias.size(0) == weight.size(0);

    CHECK_SIZE_DIM0(grad_output, input.size(0))

    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros_like(bias);
    auto grad_input = torch::zeros_like(input);

    TORCH_CHECK(input_spectral_weight.size(0) == 0 || input_spectral_weight.size(0) > max_r,
        "input_spectral_weight.size(0) bad size (", input_spectral_weight.size(0), " <= ",
        max_r, "). In ", __FILE__, ":", __LINE__)

    if (input.device().type() == torch::kCPU)
    {
        volume_extraction_backward_cpu(
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*input_spectral_weight*/ input_spectral_weight,
            /*input*/ input,
            /*grad_output*/ grad_output,
            /*grad_weight*/ grad_weight,
            /*grad_bias*/ grad_bias,
            /*grad_input*/ grad_input,
            /*max_r2*/ (int) max_r * max_r,
            /*do_bias*/ do_bias,
            /*do_input_grad*/ input.requires_grad(),
            /*do_spectral_weighting*/ input_spectral_weight.size(0) > 0
        );
    }
    else if (input.device().type() == torch::kCUDA)
    {
        volume_extraction_backward_cuda(
            /*grid3d_index*/ grid3d_index,
            /*weight*/ weight,
            /*bias*/ bias,
            /*input_spectral_weight*/ input_spectral_weight,
            /*input*/ input,
            /*grad_output*/ grad_output,
            /*grad_weight*/ grad_weight,
            /*grad_bias*/ grad_bias,
            /*grad_input*/ grad_input,
            /*max_r2*/ (int) max_r * max_r,
            /*do_bias*/ do_bias,
            /*do_input_grad*/ input.requires_grad(),
            /*do_spectral_weighting*/ input_spectral_weight.size(0) > 0
        );
    }
    else
        throw std::logic_error("Support for device not implemented");

    return {grad_input, grad_weight, grad_bias};
}
