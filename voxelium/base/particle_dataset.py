#!/usr/bin/env python3

"""
Module for managing pytorch dataset of particles.
"""

import os
import shutil
import warnings
from typing import Any, List, Dict

import numpy as np
import mrcfile
from scipy.ndimage import shift
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import tqdm
from voxelium.base.ctf import ContrastTransferFunction

from collections import namedtuple
from multiprocessing import Process, Value, Array

ParticleDatasetImage = namedtuple("ParticleDatasetImage", ["image", "filename", "random_subset"])


class ParticleDataset(Dataset):
    def __init__(self) -> None:
        self.image_file_paths = None

        self.part_random_subset = None
        self.part_rotation = None
        self.part_translation = None
        self.part_defocus = None
        self.part_og_idx = None
        self.part_stack_idx = None
        self.part_image_file_path_idx = None
        self.part_norm_correction = None
        self.part_noise_group_id = None
        self.part_noise_group_idx = None
        self.part_preloaded_image = None
        self.nr_parts = None
        self.nr_noise_groups = None
        self.dtype = np.float32

        self.has_ctf = None
        self.compute_ctf = True

        # Dictionaries mapping optics group id to data
        self.optics_group_stats = []
        self.optics_group_ctfs = []

        self.cache_root = None
        self.shared_cached_file = None
        self.shared_cached_pos = None
        self.shared_cached_size = None
        self.shared_cached_side = None

    def initialize(
            self,
            image_file_paths: List,
            part_random_subset: List,
            part_rotation: List,
            part_translation: List,
            part_defocus: List,
            part_og_idx: List,
            part_stack_idx: List,
            part_image_file_path_idx: List,
            part_norm_correction: List,
            part_noise_group_id: List,
            optics_group_stats: List,
            dtype: np.dtype = np.float32,
    ) -> None:
        self.image_file_paths = image_file_paths
        self.part_random_subset = part_random_subset
        self.part_rotation = part_rotation
        self.part_translation = part_translation
        self.part_defocus = part_defocus
        self.part_og_idx = part_og_idx
        self.part_stack_idx = part_stack_idx
        self.part_image_file_path_idx = part_image_file_path_idx
        self.part_norm_correction = part_norm_correction
        self.part_noise_group_id = part_noise_group_id
        self.optics_group_stats = optics_group_stats
        self.dtype = dtype

        # convert unique ids to indices
        d = defaultdict(count(0).__next__)
        self.part_noise_group_idx = np.array([d[k] for k in self.part_noise_group_id])
        self.nr_noise_groups = len(list(set(self.part_noise_group_idx)))
        self.nr_parts = len(self.part_rotation)

        if np.all(np.isnan(self.part_defocus)):
            self.has_ctf = False
        else:
            self.has_ctf = True
            self.setup_ctfs()

        self.setup_cache_resources()

    def setup_ctfs(self, h_sym: bool = False, compute_ctf: bool = None):
        if self.part_defocus is None:
            return

        if compute_ctf is not None:
            self.compute_ctf = compute_ctf

        for og in self.optics_group_stats:
            if og["voltage"] is not None or \
                    og["spherical_aberration"] is not None or \
                    og["amplitude_contrast"] is not None:
                ctf = ContrastTransferFunction(
                    og["voltage"],
                    og["spherical_aberration"],
                    og["amplitude_contrast"]
                )
            else:
                ctf = None
                warnings.warn(f"WARNING: CTF parameters missing for optics group ID: {id}", RuntimeWarning)

            self.optics_group_ctfs.append(ctf)

    def get_optics_group_stats(self):
        return self.optics_group_stats

    def get_optics_group_ctfs(self):
        return self.optics_group_ctfs

    def get_nr_noise_groups(self):
        return self.nr_noise_groups

    def set_cache_root(self, path):
        self.cache_root = path
        if os.path.isdir(self.cache_root):
            shutil.rmtree(self.cache_root)
        if os.path.isfile(self.cache_root):
            raise RuntimeError("Cache path is a file.")
        else:
            os.makedirs(self.cache_root)

        self.setup_cache_resources()

    def setup_cache_resources(self):
        if self.cache_root is None and self.nr_parts is None:
            return
        self.shared_cached_file = Array('i', [-1] * self.nr_parts)
        self.shared_cached_pos = Array('i', [-1] * self.nr_parts)
        self.shared_cached_size = Array('i', [-1] * self.nr_parts)
        self.shared_cached_side = Array('i', [-1] * self.nr_parts)

    def append_to_cache(self, index, data):
        process_info = torch.utils.data.get_worker_info()

        pid = 0 if process_info is None else process_info.id
        file_path = os.path.join(self.cache_root, "process_" + str(pid) + ".dat")
        if not os.path.isfile(file_path):
            file = open(file_path, 'wb')
        else:
            file = open(file_path, 'ab')

        file.seek(0, os.SEEK_END)  # Seek to end of file
        pos = file.tell()

        self.shared_cached_size[index] = data.size * data.itemsize # Needs array size in bytes, not just array size
        self.shared_cached_pos[index] = pos
        self.shared_cached_file[index] = pid
        self.shared_cached_side[index] = data.shape[-1]

        file.write(data)
        file.close()

    def load_from_cache(self, index):
        pid = self.shared_cached_file[index]
        file_path = os.path.join(self.cache_root, "process_" + str(pid) + ".dat")
        with open(file_path, 'r+b') as file:
            file.seek(self.shared_cached_pos[index], os.SEEK_SET)
            data = file.read(self.shared_cached_size[index])
        data = np.frombuffer(data, dtype=self.dtype)
        side = self.shared_cached_side[index]
        data = data.reshape(side, side)
        return data

    def preload_images(self):
        self.part_preloaded_image = [None for _ in range(len(self.part_rotation))]
        part_index_list = np.arange(len(self.part_rotation))
        unique_file_idx, unique_reverse = np.unique(self.part_image_file_path_idx, return_inverse=True)
        
        pbar = tqdm.tqdm(total=len(self.part_image_file_path_idx), smoothing=0.1)
        for i in range(len(unique_file_idx)):
            file_idx = unique_file_idx[i]
            path = self.image_file_paths[file_idx]
            with mrcfile.mmap(path, 'r') as mrc:
                # Mask out particles with no images in this file stack
                this_file_mask = unique_reverse == file_idx
                this_file_stack_indices = self.part_stack_idx[this_file_mask]
                this_file_index_list = part_index_list[this_file_mask]  # Particles indices with images in this file
                
                # Since this_file_stack_indices indexes into the mmap object, we should make sure
                # it is sorted so we minimize disk accesses
                stack_indices_argsort = np.argsort(this_file_stack_indices)
                for j in range(len(this_file_stack_indices)):
                    k = stack_indices_argsort[j]
                    idx = this_file_index_list[k]
                    self.part_preloaded_image[idx] = mrc.data[this_file_stack_indices[k]].astype(self.dtype) # Take slices of images for this data set
                    pbar.update()
        pbar.close()

    def load_image(self, index) -> ParticleDatasetImage:
        image_file_path_idx = self.part_image_file_path_idx[index]
        image_filename = self.image_file_paths[image_file_path_idx]
        random_subset = self.part_random_subset[index]
        if self.part_preloaded_image is not None and len(self.part_preloaded_image) > 0:
            image = self.part_preloaded_image[index]
        elif self.shared_cached_size[index] is not None and self.shared_cached_size[index] > 0:
            image = self.load_from_cache(index)
        else:
            with mrcfile.mmap(image_filename, 'r') as mrc:
                stack_idx = self.part_stack_idx[index]
                if len(mrc.data.shape) > 2:
                    image = mrc.data[stack_idx].astype(self.dtype)         
                else:
                    image = mrc.data.astype(self.dtype)

            if self.shared_cached_size is not None:
                self.append_to_cache(index, image)

        return ParticleDatasetImage(image=image, filename=image_filename, random_subset=random_subset)

    def __getitem__(self, index):
        dataset_image = self.load_image(index)
        image = torch.Tensor(dataset_image.image.astype(np.float32))
        og_idx = self.part_og_idx[index]
        ng_idx = self.part_noise_group_idx[index]

        rotation = torch.Tensor(self.part_rotation[index])
        translation = torch.Tensor(self.part_translation[index])

        data = {
            "image": image,
            "rotation": rotation,
            "translation": translation,
            "idx": index,
            "optics_group_idx": og_idx,
            "noise_group_idx": ng_idx,
            "random_subset": dataset_image.random_subset,                       
        }
        
        if self.compute_ctf:
            if not self.has_ctf or self.optics_group_ctfs[og_idx] is None:
                data["ctf"] = torch.ones_like(image)
            else:
                data["ctf"] = torch.Tensor(
                    self.optics_group_ctfs[og_idx](
                        self.optics_group_stats[og_idx]["image_size"],
                        self.optics_group_stats[og_idx]["pixel_size"],
                        torch.Tensor([self.part_defocus[index][0]]),
                        torch.Tensor([self.part_defocus[index][1]]),
                        torch.Tensor([self.part_defocus[index][2]])
                    )
                ).squeeze(0)
        
        return data

    def __len__(self):
        return len(self.part_rotation)

    def get_state_dict(self) -> Dict:
        return {
            "type": "ParticleDataset",
            "version": "0.0.1",
            "image_file_paths": self.image_file_paths,
            "part_random_subset": self.part_random_subset,
            "part_rotation": self.part_rotation,
            "part_translation": self.part_translation,
            "part_defocus": self.part_defocus,
            "part_og_idx": self.part_og_idx,
            "part_stack_idx": self.part_stack_idx,
            "part_image_file_path_idx": self.part_image_file_path_idx,
            "part_norm_correction": self.part_norm_correction,
            "part_noise_group_id": self.part_noise_group_id,
            "optics_group_stats": self.optics_group_stats,
        }

    def set_state_dict(self, state_dict):
        if "type" not in state_dict or state_dict["type"] != "ParticleDataset":
            raise TypeError("Input is not an 'ParticleDataset' instance.")

        if "version" not in state_dict:
            raise RuntimeError("ParticleDataset instance lacks version information.")

        if "image_random_subset" not in state_dict:
            warnings.warn(f"The Particle Dataset was saved without random subset information. Setting to empty.", RuntimeWarning)
            state_dict["image_random_subset"] = []

        if state_dict["version"] == "0.0.1":
            self.initialize(
                image_file_paths=state_dict["image_file_paths"],
                part_random_subset=state_dict["part_random_subset"],
                part_rotation=state_dict["part_rotation"],
                part_translation=state_dict["part_translation"],
                part_defocus=state_dict["part_defocus"],
                part_og_idx=state_dict["part_og_idx"],
                part_stack_idx=state_dict["part_stack_idx"],
                part_image_file_path_idx=state_dict["part_image_file_path_idx"],
                part_norm_correction=state_dict["part_norm_correction"],
                part_noise_group_id=state_dict["part_noise_group_id"],
                optics_group_stats=state_dict["optics_group_stats"]
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
