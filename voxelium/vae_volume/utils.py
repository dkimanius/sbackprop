#!/usr/bin/env python3

"""
Test module for a training VAE
"""
from glob import glob
import os
import shutil
import sys
from typing import List, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from voxelium.base.star_file import load_star
from voxelium.vae_volume.cache import Cache

import matplotlib.pyplot as plt

Tensor = TypeVar('torch.tensor')

from voxelium.base import smooth_square_mask, smooth_circular_mask, ModelContainer, grid_spectral_average_torch, \
    dt_symmetrize, spectra_to_grid_torch


def cos_step_ascend(begin_ascend, end_ascend, x):
    if x < begin_ascend:
        return 0.
    if x > end_ascend:
        return 1.
    a = begin_ascend
    b = end_ascend - begin_ascend
    return .5 + np.cos(np.pi * (x - a) / b + np.pi) / 2.


def cos_step_descend(begin_descend, end_descend, x):
    if x < begin_descend:
        return 1.
    if x > end_descend:
        return 0.
    a = begin_descend
    b = end_descend - begin_descend
    return .5 + np.cos(np.pi * (x - a) / b) / 2.


def get_kld_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def setup_device(args, verbose=False):
    device = None
    gpu_ids = []
    if args.gpu is not None:
        queried_gpu_ids = args.gpu.split(",")
        for i in range(len(queried_gpu_ids)):
            gpu_id = int(queried_gpu_ids[i].strip())
            try:
                gpu_name = torch.cuda.get_device_name(gpu_id)
            except AssertionError:
                if verbose:
                    print(f'WARNING: GPU with the device id "{gpu_id}" not found.', file=sys.stderr)
                continue
            if verbose:
                print(f'Found device "{gpu_name}"')
            gpu_ids.append(gpu_id)

        if len(gpu_ids) > 0:
            device = "cuda:" + str(gpu_ids[0])
            if verbose:
                print("Running on GPU with device id(s)", *gpu_ids)
        else:
            if verbose:
                print(f'WARNING: no GPUs were found with the specified ids.', file=sys.stderr)

    if len(gpu_ids) == 0:
        gpu_ids = None
        if verbose:
            print("Running on CPU")
        device = torch.device("cpu")

    return device, gpu_ids

def get_gradient_penalty(module: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(*x.shape[:-1], 1, device=x.device)
    interpolates = (alpha * x + (1 - alpha) * y).requires_grad_()
    m_interpolates = module(interpolates)
    grad_output = torch.ones_like(m_interpolates, requires_grad=False)
    gradients, = torch.autograd.grad(
        outputs = m_interpolates,
        inputs = interpolates,
        grad_outputs = grad_output,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )
    gradient_norm = gradients.view(x.shape[0], -1).norm(p=2, dim=-1)
    gradient_penalty = (gradient_norm - 1).square().mean()
    return gradient_penalty

np_dtype_dict = {"float32": np.float32, "float16": np.float16, "float64": np.float64}

def get_np_dtype(dtype_str: str) -> np.dtype:
    return np_dtype_dict[dtype_str]

def find_star_file_in_path(path: str, type: str = "optimiser") -> str:
    if os.path.isfile(os.path.join(path, f"run_{type}.star")):
        return os.path.join(path, f"run_{type}.star")
    files = glob(os.path.join(path, f"*{type}.star"))
    if len(files) > 0:
        files = list.sort(files)
        return files[-1]

    raise FileNotFoundError(f"Could not find '{type}' star-file in path: {path}")

def find_project_root(from_path: str, file_relative_path: str) -> str:
    """
    Searches for the Relion project root starting at from_path and iterate through parent directories
    till file_relative_path is found as a relative sub path or till filesystem root is found, at which
    point a RuntimeException is raise.

    :param from_path: starting search from this path
    :param file_relative_path: searching to find this relative path as a file
    """
    current_path = os.path.abspath(from_path)
    while True:
        trial_path = os.path.join(current_path, file_relative_path)
        if os.path.isfile(trial_path):
            return current_path
        if current_path == os.path.dirname(current_path):  # At filesystem root
            raise RuntimeError(f"Relion project directory could not be found from the subdirectory: {from_path}")
        current_path = os.path.dirname(current_path)

def dump_particles_to_dir(input_path, output_path):
    """
    Load data from path
    :param path: relion job directory or data file
    """
    if os.path.isfile(input_path):
        data_star_path = input_path
        root_search_path = os.path.dirname(os.path.abspath(input_path))
    else:
        data_star_path = os.path.abspath(find_star_file_in_path(input_path, "data"))
        root_search_path = os.path.abspath(input_path)

    data_star_path = os.path.abspath(data_star_path)
    data = load_star(data_star_path)

    if 'optics' not in data:
        raise RuntimeError("Optics groups table not found in data star file")
    if 'particles' not in data:
        raise RuntimeError("Particles table not found in data star file")

    particles = data['particles']
    nr_particles = len(particles['rlnImageName'])
    image_file_paths = set()

    for i in range(nr_particles):
        img_name = particles['rlnImageName'][i]
        img_tokens = img_name.split("@")
        if len(img_tokens) == 2:
            img_path = img_tokens[1]
        elif len(img_tokens) == 1:
            img_path = img_tokens[1]
        else:
            raise RuntimeError(f"Invalid image file name (rlnImageName): {img_name}")
        image_file_paths.add(img_path)

    image_file_paths = list(image_file_paths)

    project_root = find_project_root(root_search_path, image_file_paths[0])

    # Convert image paths to absolute paths
    for i in range(len(image_file_paths)):
        image_file_paths[i] = os.path.abspath(os.path.join(project_root, image_file_paths[i]))
    
    new_project_path = os.path.abspath(output_path)
    destination_image_file_paths = [p.replace(project_root, new_project_path) for p in image_file_paths]

    [os.makedirs(os.path.dirname(p), exist_ok=True) for p in destination_image_file_paths]

    for src, dst in zip(image_file_paths, destination_image_file_paths):
        shutil.copy(src, dst)

    new_star_path = data_star_path.replace(os.path.dirname(data_star_path), new_project_path)
    shutil.copy(data_star_path, new_star_path)

def plot_fscs(output_file, **fscs):
    fig, main_ax = plt.subplots()
    for plot_name in fscs:
        fsc_file = load_star(fscs[plot_name])
        fsc_values = [float(x) for x in fsc_file["fsc"]["rlnFourierShellCorrelation"]]
        main_ax.plot(fsc_values, label=plot_name)
    main_ax.legend()
    main_ax.set_xlabel("1/Angstroms (1/Å)")
    main_ax.set_ylabel("FSC")
    plt.plot()
    plt.savefig(output_file)


# class SpectralStandardMapping(torch.nn.Module):
#     def __init__(self, image_size):
#         super().__init__()
#         self.image_size = image_size
#         self.std_spectra = torch.nn.Parameter(torch.zeros(image_size // 2 + 1))
#         self.std_grid = torch.nn.Parameter(torch.zeros(image_size, image_size))
#
#     def forward(self, input):
#
#
#     def track(self, input):
#         with torch.no_grad():
#             s_idx = Cache.get_spectral_indices(
#                 (self.image_size + 1, self.image_size + 1),
#                 max_r=self.image_size // 2 + 1,
#                 device=input.device
#             )[:, self.image_size // 2:]
#
#             if torch.is_complex(input):
#                 input = torch.view_as_real(input)
#
#             power = torch.mean(torch.sum(torch.square(input), -1), 0)
#             power = grid_spectral_average_torch(power, s_idx)
#             power = power[:, :-1]
#             power[:, -1] = power[:, -2]
#
#             stds = torch.sqrt(power)
#
#             g_idx = Cache.get_spectral_indices(
#                 (self.image_size, self.image_size),
#                 max_r=self.image_size // 2,
#                 device=input.device
#             )
#             g_idx = dt_symmetrize(g_idx, dim=2)[..., self.image_size // 2:]
#             self.std_grid.data = spectra_to_grid_torch(spectra=stds, indices=g_idx)
#             self.std_spectra = stds
