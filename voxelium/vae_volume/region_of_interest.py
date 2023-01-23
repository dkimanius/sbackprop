#!/usr/bin/env python3

"""
Container for dataset analysis results
"""

import os
import shutil
import sys
import numpy as np
from typing import Dict, List, TypeVar, Tuple, Any

from voxelium.vae_volume.cache import Cache
from voxelium.vae_volume.svr_linear import make_grid3d

Tensor = TypeVar('torch.tensor')

import torch
import torch.nn.functional as F

from voxelium.base import euler_to_matrix, matrix_to_quaternion, ContrastTransferFunction, quaternion_to_matrix, \
    get_spectral_avg, spectrum_to_grid, load_mrc, dt_desymmetrize, idft


class RegionOfInterest:
    def __init__(self, dac, mask_fn, resolution, latent_fraction, device):
        self.vae_container = dac.vae_container
        self.latent_size = self.vae_container.latent_size
        self.pp_latent_size = self.vae_container.decoder.get_postprocess_input_size()
        self.reconst_latent_size = self.latent_size - self.pp_latent_size

        image_size = dac.auxiliaries["image_size"]
        pixel_size = dac.auxiliaries["pixel_size"]
        self.image_max_r = dac.auxiliaries["image_max_r"]

        roi_res_idx = round(image_size * pixel_size / resolution)  # Convert resolution to spectral index
        self.roi_roni_switch = True  # If true, apply ROI, if false apply RONI

        roi_latent_size = round(self.reconst_latent_size * np.clip(latent_fraction, 0., 1.))
        roi_latent_index = np.arange(self.reconst_latent_size)
        self.roi_latent_index, self.roni_latent_index = \
            roi_latent_index[:roi_latent_size], roi_latent_index[roi_latent_size:]

        print(f"{roi_latent_size} latent dimensions assigned to ROI")
        print("Using ROI resolution cutoff at Fourier shell index", roi_res_idx)
        roi_mask_, roi_voxel_size, _ = load_mrc(mask_fn)
        if np.abs(pixel_size - roi_voxel_size) > 1e-1:
            print(f"WARNING: ROI mask voxel size ({round(roi_voxel_size, 2)}) "
                  f"is not equal to data ({round(pixel_size, 2)}).", file=sys.stderr)

        if np.min(roi_mask_) < 0. or 1. < np.max(roi_mask_):
            print(f"WARNING: ROI mask has values outside range [0,1].", file=sys.stderr)

        if np.all(roi_mask_.shape == image_size):
            raise RuntimeError(f"ROI mask size ({roi_mask_.shape}) not equal to data size ({image_size})")

        grid3d_radius = torch.round(torch.sqrt(torch.sum(torch.square(make_grid3d(size=image_size).to(device)), -1)))
        self.roi_res_mask = grid3d_radius < roi_res_idx
        roi_mask_ = torch.Tensor(roi_mask_.copy()).to(device)
        redund = -4.  # Mask redundancy coefficient
        if redund != 0:
            c = np.exp(redund) / (np.exp(redund) - 1)
            self.roi_mask = c * (1 - torch.exp(-redund * roi_mask_))
            self.roni_mask = c * (1 - torch.exp(-redund * (1-roi_mask_)))
        else:
            self.roi_mask = roi_mask_
            self.roni_mask = 1 - roi_mask_

        self.roi_slice = None
        self.roni_slice = None

        dac.auxiliaries["roi_latent_index"] = self.roi_latent_index
        dac.auxiliaries["roni_latent_index"] = self.roni_latent_index

    def get_loss(self, z_selected_):
        z_selected = torch.zeros_like(z_selected_)

        if self.roi_roni_switch:
            z_selected[self.roi_latent_index] = \
                z_selected_[self.roi_latent_index]  # Dedicate to inside mask
        else:
            z_selected[self.roni_latent_index] = \
                z_selected_[self.roni_latent_index]  # Dedicate to outside mask

        z_selected[-self.pp_latent_size:] = z_selected_[-self.pp_latent_size:].detach()  # Include postprocess
        z_selected = z_selected.unsqueeze(0)
        v_ft = self.vae_container.decoder(z_selected, self.image_max_r)
        v_ft = torch.view_as_complex(v_ft)
        v_ft[0, self.roi_res_mask] = v_ft[0, self.roi_res_mask].detach()
        v_ft = dt_desymmetrize(v_ft)[0]
        vol = idft(v_ft, dim=3, real_in=True)
        mask = self.roni_mask if self.roi_roni_switch else self.roi_mask
        masked_mean = torch.sum(vol * mask) / torch.sum(mask)
        roi_loss = np.prod(vol.shape[-2:]) * torch.mean(mask * torch.square(vol - masked_mean))

        if self.roi_roni_switch:
            self.roi_slice = vol[vol.shape[0] // 2].detach().cpu().numpy()
        else:
            self.roni_slice = vol[vol.shape[0] // 2].detach().cpu().numpy()

        self.roi_roni_switch = not self.roi_roni_switch  # Alternate between inside and outside mask

        return roi_loss