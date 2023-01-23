#pragma once
#ifndef SVR_LINEAR_VOLUME_EXTRACTION_H
#define SVR_LINEAR_VOLUME_EXTRACTION_H

#include <vector>
#include <stdexcept>

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "vae_volume/svr_linear/volume_extraction_cpu_kernels.h"
#include "vae_volume/svr_linear/volume_extraction_cuda_kernels.h"

torch::Tensor volume_extraction_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grid3d_index,
    const int max_r
);

std::vector<torch::Tensor> volume_extraction_backward(
    torch::Tensor input_spectral_weight,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grad_output,
    torch::Tensor grid3d_index
);

#endif // SVR_LINEAR_VOLUME_EXTRACTION_H