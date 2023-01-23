#!/usr/bin/env python3

"""
Python API for the sparse volume reconstruction linear layer
"""
import time
from typing import TypeVar, Union

import numpy as np
import torch
import voxelium_svr_linear

from voxelium.base import grid_iterator, sample_gaussian_function
from voxelium.relion import eulerToMatrix

Tensor = TypeVar('torch.tensor')

VOXEL_SPREAD_MARGIN = 3


class SparseVolumeReconstructionLinear(torch.nn.Module):
    def __init__(
            self,
            size,
            input_size,
            dtype=torch.float32,
            bias=True,
            input_spectral_weight=None
    ):
        super(SparseVolumeReconstructionLinear, self).__init__()

        if size % 2 == 0:
            size += 1

        self.size = size
        self.size_x = size // 2 + 1
        self.input_size = input_size

        bz = size
        bz_2 = bz // 2
        grid_indices = np.zeros((bz, bz, bz // 2 + 1), dtype=int) - 1
        max_r2 = (size // 2) ** 2
        i = 0
        # for z, y, x in grid_iterator(bz-1, bz-1, bz_2+1):
        for z, y, x in grid_iterator(bz, bz, bz_2 + 1):
            if (z - bz_2) ** 2 + (y - bz_2) ** 2 + x ** 2 <= max_r2:
                grid_indices[z, y, x] = i
                i += 1

        self.weight_count = i

        # Add margin for pixel spread into voxels
        m = VOXEL_SPREAD_MARGIN
        bz += m * 2
        bz_2 += m
        grid_indices_margin = np.zeros((bz, bz, bz_2 + 1), dtype=int) - 1
        grid_indices_margin[m:-m, m:-m, :-m] = grid_indices
        grid_indices = grid_indices_margin

        for i in range(5):
            self.radial_expansion(grid_indices)

        self.grid3d_index = torch.nn.Parameter(
            torch.tensor(grid_indices, dtype=torch.long), requires_grad=False)

        data_tensor = torch.empty((self.weight_count, input_size, 2), dtype=dtype).normal_()
        self.weight = torch.nn.Parameter(data=data_tensor, requires_grad=True)

        if bias:
            data_tensor = torch.empty((self.weight_count, 2), dtype=dtype).normal_()
            self.bias = torch.nn.Parameter(data=data_tensor, requires_grad=True)
        else:
            self.bias = None

        self.weight.data *= 1e-6
        if self.bias is not None:
            self.bias.data *= 0

        self.input_spectral_weight = torch.nn.Parameter(
            torch.empty(0, dtype=self.weight.dtype) if input_spectral_weight is None else input_spectral_weight
        )
        self.input_spectral_weight.requires_grad = False

    def forward(self, input, max_r=None, grid2d_coord=None, rot_matrices=None, sparse_grad=True):
        if rot_matrices is not None and grid2d_coord is not None:
            return TrilinearProjection.apply(
                input,  # input
                self.weight,  # weight
                self.bias,  # bias
                self.grid3d_index,  # grid3d_index
                rot_matrices,  # rot_matrices
                grid2d_coord,  # grid2d_coord
                max_r,  # max_r
                self.input_spectral_weight,  # input_spectral_weight
                sparse_grad,  # sparse_grad
                False  # testing
            )
        else:
            return VolumeExtraction.apply(
                input,  # input
                self.weight,  # weight
                self.bias,  # bias
                self.grid3d_index,  # grid3d_index
                self.input_spectral_weight,  # input_spectral_weight
                max_r  # max_r
            )

    def set_reference(self, grid_ht: Union[Tensor, np.ndarray]):
        self.weight.data = torch.zeros_like(self.weight.data)
        m = VOXEL_SPREAD_MARGIN
        max_r2 = (self.size // 2) ** 2
        s = (self.size - 1) // 2
        for z, y, x in grid_iterator(self.size, self.size, self.size_x):
            i = self.grid3d_index[z + m, y + m, x]
            if i >= 0 and (z - s) ** 2 + (y - s) ** 2 + x ** 2 <= max_r2:
                self.weight.data[i] = grid_ht[z, y, x]

    @staticmethod
    def radial_expansion(grid):
        assert grid.shape[0] == grid.shape[1] == grid.shape[2] * 2 - 1
        bz = grid.shape[0]
        bz2 = bz // 2
        mask1 = grid == -1

        ls = np.linspace(-bz2, bz2, bz)
        lsx = np.linspace(0, bz2, bz // 2 + 1)
        z, y, x = np.meshgrid(ls, ls, lsx)
        c = np.zeros((int(np.sum(mask1)), 3))
        c[:, 0] = x[mask1]
        c[:, 1] = y[mask1]
        c[:, 2] = z[mask1]

        norm = np.sqrt(np.sum(np.square(c), axis=1))
        c_ = np.round(c / norm[:, None]).astype(int)

        c = c.astype(int)
        c[:, 1:] += bz2
        c_ = c - c_

        g = grid[c_[:, 2], c_[:, 1], c_[:, 0]]
        mask2 = g >= 0

        grid[c[mask2, 2], c[mask2, 1], c[mask2, 0]] = g[mask2]


class TrilinearProjection(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, input, weight, bias, grid3d_index,
            rot_matrices, grid2d_coord, max_r,
            input_spectral_weight=None,
            sparse_grad=True, testing=False
    ):
        assert grid3d_index.shape[0] == grid3d_index.shape[1] == grid3d_index.shape[2] * 2 - 1
        if max_r is None:
            max_r = (grid3d_index.shape[0] - 2 * VOXEL_SPREAD_MARGIN) // 2
        else:
            max_r = min(max_r, (grid3d_index.shape[0] - 2 * VOXEL_SPREAD_MARGIN) // 2)

        input_spectral_weight = torch.empty(0, dtype=input.dtype).to(input.device) \
            if input_spectral_weight is None else input_spectral_weight

        output = voxelium_svr_linear.trilinear_projection_forward(
            input=input,
            weight=weight,
            bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
            rot_matrix=rot_matrices,
            grid2d_coord=grid2d_coord,
            grid3d_index=grid3d_index,
            max_r=max_r
        )

        ctx.save_for_backward(
            input,
            weight,
            bias,
            grid3d_index,
            rot_matrices,
            grid2d_coord,
            input_spectral_weight,
            torch.Tensor([max_r]),
            torch.Tensor([sparse_grad]),
            torch.Tensor([testing])
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, grid3d_index, rot_matrices, grid2d_coord, \
            input_spectral_weight, max_r, sparse_grad, testing \
            = ctx.saved_tensors
        sparse_grad = bool(sparse_grad[0])

        grad_input, grad_weight_index, grad_weight_values, grad_bias_values, grad_rot_matrix = \
            voxelium_svr_linear.trilinear_projection_backward(
                input=input,
                grid2d_grad=grad_output.contiguous(),
                weight=weight,
                bias=torch.empty([0, 0], dtype=weight.dtype).to(weight.device) if bias is None else bias,
                grid3d_index=grid3d_index,
                rot_matrix=rot_matrices,
                grid2d_coord=grid2d_coord,
                input_spectral_weight=input_spectral_weight,
                max_r=max_r[0],
                sparse_grad=sparse_grad
            )

        if sparse_grad:
            grad_weight = torch.sparse_coo_tensor(
                grad_weight_index.contiguous().view(1, -1),
                grad_weight_values.contiguous().view(-1, weight.shape[-2], 2),
                weight.shape
            )
        else:
            grad_weight = grad_weight_values

        if bias is None:
            grad_bias = None
        else:
            if sparse_grad:
                grad_bias = torch.sparse_coo_tensor(
                    grad_weight_index.contiguous().view(1, -1),
                    grad_bias_values.contiguous().view(-1, 2),
                    bias.shape
                )
            else:
                grad_bias = grad_bias_values

        if testing[0] and sparse_grad:
            grad_weight = grad_weight.to_dense()
            if bias is not None:
                grad_bias = grad_bias.to_dense()

        return grad_input, grad_weight, grad_bias, None, grad_rot_matrix, None, None, None, None, None


class VolumeExtraction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, grid3d_index, input_spectral_weight=None, max_r=None):

        assert grid3d_index.shape[0] == grid3d_index.shape[1] == grid3d_index.shape[2] * 2 - 1
        if max_r is None:
            max_r = (grid3d_index.shape[0] - 2 * VOXEL_SPREAD_MARGIN) // 2
        else:
            max_r = min(max_r, (grid3d_index.shape[0] - 2 * VOXEL_SPREAD_MARGIN) // 2)

        input_spectral_weight = torch.empty(0, dtype=input.dtype).to(input.device) \
            if input_spectral_weight is None else input_spectral_weight

        output = voxelium_svr_linear.volume_extraction_forward(
            input=input,
            weight=weight,
            bias=torch.empty(0, dtype=weight.dtype).to(weight.device) if bias is None else bias,
            grid3d_index=grid3d_index,
            max_r=max_r
        )
        ctx.save_for_backward(input, weight, bias, input_spectral_weight, grid3d_index)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, input_spectral_weight, grid3d_index = ctx.saved_tensors

        input_spectral_weight = torch.empty(0, dtype=weight.dtype).to(weight.device) \
            if input_spectral_weight is None else input_spectral_weight

        grad_input, grad_weight, grad_bias = \
            voxelium_svr_linear.volume_extraction_backward(
                input=input,
                weight=weight,
                bias=torch.empty(0, dtype=weight.dtype).to(weight.device) if bias is None else bias,
                grad_output=grad_output,
                grid3d_index=grid3d_index,
                input_spectral_weight=input_spectral_weight
            )

        return grad_input, grad_weight, grad_bias, None, None, None
