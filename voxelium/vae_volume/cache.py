#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import sys
from typing import List, TypeVar, Union, Tuple, Any

import numpy as np
import torch

Tensor = TypeVar('torch.tensor')

from voxelium.base import smooth_square_mask, smooth_circular_mask, get_spectral_indices, dt_symmetrize


class Cache:
    square_masks = {}
    circular_masks = {}
    spectral_indices = {}
    encoder_input_masks = {}
    
    @staticmethod
    def _get_square_mask(image_size: int, thickness: float) -> Tensor:
        return torch.Tensor(
            smooth_square_mask(
                image_size=image_size,
                square_side=image_size - thickness * 2,
                thickness=thickness
            )
        )

    @staticmethod
    def get_square_mask(image_size: int, thickness: float, device: Any = 'cpu') -> Tensor:
        tag = str(image_size) + "_" + str(thickness) + "_" + str(device)
        if tag not in Cache.square_masks:
            Cache.square_masks[tag] = Cache._get_square_mask(image_size, thickness).to(device)
        return Cache.square_masks[tag]

    @staticmethod
    def apply_square_mask(input: Tensor, thickness: float) -> Tensor:
        return input * Cache.get_square_mask(input.shape[-1], thickness, input.device)[None, ...]

    @staticmethod
    def _get_circular_mask(image_size: int, radius: float, thickness: float) -> Tensor:
        return torch.Tensor(
            smooth_circular_mask(
                image_size=image_size,
                radius=radius,
                thickness=thickness
            )
        )

    @staticmethod
    def get_circular_mask(image_size: int, radius: float, thickness: float, device: Any = 'cpu') -> Tensor:
        tag = str(image_size) + "_" + str(radius) + "_" + str(thickness) + "_" + str(device)
        if tag not in Cache.circular_masks:
            Cache.circular_masks[tag] = Cache._get_circular_mask(image_size, radius, thickness).to(device)
        return Cache.circular_masks[tag]

    @staticmethod
    def apply_circular_mask(input: Tensor, radius: float, thickness: float) -> Tensor:
        return input * Cache.get_circular_mask(input.shape[-1], radius, thickness, input.device)[None, ...]

    @staticmethod
    def _get_spectral_indices(
            shape: Union[Tuple[int, int], Tuple[int, int, int]], numpy: bool = False, max_r: int = None
    ) -> Union[Tensor, np.ndarray]:
        if shape[0] != shape[1]:
            out = get_spectral_indices((shape[0], shape[0]))
            out = out[:, shape[0]//2:]
        else:
            out = get_spectral_indices(shape)
        if max_r is not None:
            out[out > max_r] = max_r
        if not numpy:
            out = torch.Tensor(out)
        return out

    @staticmethod
    def get_spectral_indices(
            shape: Union[Tuple[int, int], Tuple[int, int, int]],
            numpy: bool = False,
            device: Any = 'cpu',
            max_r: int = None
    ) -> Union[Tensor, np.ndarray]:
        tag = str(shape) + "_" + str(max_r)
        tag += "_np" if numpy else "_" + str(device)
        if tag not in Cache.spectral_indices:
            Cache.spectral_indices[tag] = Cache._get_spectral_indices(shape, numpy, max_r)
            if not numpy:
                Cache.spectral_indices[tag] = Cache.spectral_indices[tag].to(device)
        return Cache.spectral_indices[tag]

    @staticmethod
    def _get_encoder_input_mask(image_size: int, max_r: int = None) -> Tensor:
        spectral_indices = Cache._get_spectral_indices((image_size, image_size))
        if max_r is None:
            return spectral_indices < image_size // 2 + 1
        else:
            return spectral_indices < max_r

    @staticmethod
    def get_encoder_input_mask(image_size: int, max_r: int = None, device: Any = 'cpu') -> Tensor:
        tag = str(image_size) + "_" + str(max_r) + "_" + str(device)
        if tag not in Cache.encoder_input_masks:
            Cache.encoder_input_masks[tag] = Cache._get_encoder_input_mask(image_size, max_r).to(device)
        return Cache.encoder_input_masks[tag]

    @staticmethod
    def apply_encoder_input_mask(input: Tensor, max_r: int = None, device: Any = 'cpu') -> Tensor:
        return input[None, Cache.get_encoder_input_mask(input.shape[-1], max_r, device)]
