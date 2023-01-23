#!/usr/bin/env python3

"""
Module for the sparse linear layer
"""
import copy
import time
from typing import TypeVar, Union
import numpy as np
import torch

Tensor = TypeVar('torch.tensor')


def make_compact_grid2d(size: int = None, max_r: int = None):
    """
    Makes image grid coordinates and indices.
    Used by sparse project to output projection images out of 3D grids.
    Must provide either max_r or size. If both are given max_r will be ignored.
    For even box size: img_size = max_r * 2 - 2 <=> max_r = floor(img_size / 2) + 1
    For odd box size: img_size = max_r * 2 - 1 <=> max_r = floor(img_size / 2) + 1
    :param size: Size of the grid containing a max_r circle (not including max_r)
    :param max_r: Max radius of circle contained by image grid.
    """
    if max_r is None:
        if size is None:
            raise RuntimeError("Either max_r or size must be given.")
        if size % 2 == 0:
            size += 1
        max_r = (size - 1) // 2
    if size is None:
        if max_r is None:
            raise RuntimeError("Either max_r or size must be given.")
        size = max_r * 2 + 1

    size_2 = size // 2

    # Make xy-plane grid coordinates
    ls = torch.linspace(-size_2, size_2, size)
    lsx = torch.linspace(0, size_2, size_2 + 1)
    y, x = torch.meshgrid(ls, lsx, indexing='ij')
    coord = torch.stack([x, y], 2).view(-1, 2)

    # We need to work with explicit indices, flatten coordinate grid
    radius = torch.sqrt(torch.sum(torch.square(coord), -1))

    # Mask out beyond Nyqvist in 2D grid
    mask = radius <= max_r
    mask = mask.flatten()

    coord = coord[mask].contiguous()

    # import matplotlib
    # import matplotlib.pylab as plt
    # matplotlib.use('TkAgg')
    # plt.plot(coord.data[:, 0].numpy(), coord.data[:, 1].numpy(), '.', alpha=0.3)
    # plt.show()

    coord.require_grad = False

    return coord, mask


def make_grid3d(size: int = None, max_r: int = None):
    """
    Makes volume grid coordinates and indices.
    Used by sparse project to output projection images out of 3D grids.
    Must provide either max_r or size. If both are given max_r will be ignored.
    Note: img_size = max_r * 2 + 1 <=> max_r = floor(img_size / 2)
    :param size: Size of the grid containing a max_r circle
    :param max_r: Max radius of circle contained by image grid.
    """
    if max_r is None:
        if size is None:
            raise RuntimeError("Either max_r or size must be given.")
        if size % 2 == 0:
            size += 1
        max_r = (size - 1) // 2
    if size is None:
        if max_r is None:
            raise RuntimeError("Either max_r or size must be given.")
        size = max_r * 2 + 1

    size_2 = size // 2

    # Make xy-plane grid coordinates
    ls = torch.linspace(-size_2, size_2, size)
    lsx = torch.linspace(0, size_2, size_2 + 1)
    coord = torch.stack(torch.meshgrid(ls, ls, lsx, indexing='ij'), 3)

    # Mask out beyond Nyqvist in 2D grid
    radius = torch.sqrt(torch.sum(torch.square(coord), -1))
    mask = radius <= max_r

    return coord, mask
