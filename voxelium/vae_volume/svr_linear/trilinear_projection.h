#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "vae_volume/svr_linear/trilinear_projection_cpu_kernels.h"
#include "vae_volume/svr_linear/trilinear_projection_cuda_kernels.h"

torch::Tensor trilinear_projection_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor rot_matrix,
    torch::Tensor grid2d_coord,
    torch::Tensor grid3d_index,
    const int max_r
);

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
);