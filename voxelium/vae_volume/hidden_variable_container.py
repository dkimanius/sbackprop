#!/usr/bin/env python3

"""
Container for dataset analysis results
"""

import os
import shutil
import sys
import numpy as np
from typing import Dict, List, TypeVar, Tuple, Any

from voxelium.base.torch_utils import optimizer_set_learning_rate
from voxelium.vae_volume.cache import Cache

Tensor = TypeVar('torch.tensor')

import torch
import torch.nn.functional as F

from voxelium.base import euler_to_matrix, ContrastTransferFunction, \
    grid_spectral_average_torch, spectra_to_grid_torch, ModelContainer, PowerModelWrapper

VARIANCE_SPECTRA_OPT_LR = [5e-1, 1e-1]


class HiddenVariableModule(torch.nn.Module):
    def __init__(self, vars: Tensor, norm: float = 1, mean: float = 0):
        super().__init__()
        vars = (vars - mean) / norm
        self.vars = torch.nn.Parameter(vars)
        self.orig = vars.clone().detach()
        self.norm = norm
        self.mean = mean

    def forward(self, index):
        return self.vars[index] * self.norm + self.mean


class HiddenVariableContainer:
    def __init__(
            self,
            vars: Dict[str, torch.nn.Module],
            op_learning_rates: Dict[str, float],
            optics_groups: List[Dict[str, Any]],
            part_og_idx: Tensor,
            image_size: int,
            batch_size: int,
            variance_spectra: Tensor = None,
            latent_space: Tensor = None,
            structure_basis: Tensor = None,
            representation_updated: bool = False
    ):
        self.device = torch.device('cpu')
        self.vars = vars  # Current values
        self.op_learning_rates = op_learning_rates  # Optimizers learning rate

        self.do_optimize = {}
        self.ops = None  # Optimizers

        # Differentiation properties
        for k in self.vars:
            self.vars[k].requires_grad = False
            self.do_optimize[k] = False

        self.latent_space = latent_space
        self.structure_basis = structure_basis

        self.do_ctf = 'ctf_defocus_u' in vars and 'ctf_defocus_v' in vars and 'ctf_angle' in vars

        self.part_og_idx = part_og_idx  # Store a (redundant) copy here

        self.nr_particles = self.vars['pose_alpha'].vars.shape[0]
        self.image_size = image_size
        self.batch_size = batch_size

        self.optics_groups = optics_groups
        self.nr_optics_groups = len(optics_groups)

        # Variance spectrum
        if variance_spectra is not None:
            self.variance_spectra = variance_spectra
        else:
            self.variance_spectra = torch.ones([self.nr_optics_groups, image_size // 2 + 1])

        # Data standard deviation, per optics group
        self.data_stats_established = 'data_amp' in optics_groups[0]

        # If data stats is not established, accumulate over first epoch
        self.acc_power_spectra = \
            torch.zeros([self.nr_optics_groups, image_size // 2 + 1], dtype=torch.float64, requires_grad=False)
        self.acc_ctf2_spectra = \
            torch.zeros([self.nr_optics_groups, image_size // 2 + 1], dtype=torch.float64, requires_grad=False)

        self.acc_data_counts = np.zeros(self.nr_optics_groups, dtype=np.long)
        self.representation_updated = representation_updated

    def init_optimizers(self, op_states: Dict[str, Any] = None):
        self.ops = {}
        for k in self.vars:
            self.ops[k] = torch.optim.Adam(self.vars[k].parameters(), lr=self.op_learning_rates[k])
            if op_states is not None:
                self.ops[k].load_state_dict(op_states[k])

    def set_device(self, device):
        if self.do_ctf:
            for og in self.optics_groups:
                og['ctf'].to(device)

        for k in self.vars:
            self.vars[k] = self.vars[k].to(device)
            if self.do_optimize[k]:
                self.vars[k].requires_grad = True

        self.part_og_idx = self.part_og_idx.to(device)
        if self.latent_space is not None:
            self.latent_space = self.latent_space.to(device)
        if self.structure_basis is not None:
            self.structure_basis = self.structure_basis.to(device)

        self.variance_spectra = self.variance_spectra.to(device)

        self.acc_power_spectra = self.acc_power_spectra.to(device)
        self.acc_ctf2_spectra = self.acc_ctf2_spectra.to(device)

        self.device = device

    def set_latent_size(self, size):
        if self.latent_space is None or self.latent_space.shape[-1] != size:
            self.latent_space = torch.zeros([self.nr_particles, size]).to(self.device)

    def set_structure_basis_size(self, size):
        if self.structure_basis is None or self.structure_basis.shape[-1] != size:
            self.structure_basis = torch.zeros([self.nr_particles, size]).to(self.device)

    def set_latent(self, indices, values):
        self.latent_space[indices] = values
        self.representation_updated = False

    def set_structure_basis(self, indices, values):
        self.structure_basis[indices] = values
        self.representation_updated = False

    def zero_grad(self):
        for k in self.vars:
            self.ops[k].zero_grad()

    def do_align(self):
        self.do_optimize['pose_alpha'] = True
        self.vars['pose_alpha'].requires_grad = True
        self.do_optimize['pose_beta'] = True
        self.vars['pose_beta'].requires_grad = True
        self.do_optimize['pose_gamma'] = True
        self.vars['pose_gamma'].requires_grad = True
        self.do_optimize['shift_x'] = True
        self.vars['shift_x'].requires_grad = True
        self.do_optimize['shift_y'] = True
        self.vars['shift_y'].requires_grad = True

    def do_ctf_optimization(self):
        self.do_optimize['ctf_angle'] = True
        self.vars['ctf_angle'].requires_grad = True
        self.do_optimize['ctf_defocus_u'] = True
        self.vars['ctf_defocus_u'].requires_grad = True
        self.do_optimize['ctf_defocus_v'] = True
        self.vars['ctf_defocus_v'].requires_grad = True

    def accumulate_data_stats(self, data, ctf, idx):
        if self.data_stats_established:
            return

        with torch.no_grad():
            s_idx = Cache.get_spectral_indices(
                (self.image_size, self.image_size),
                max_r=self.image_size // 2,
                device=self.device
            )

            power = torch.mean(torch.square(torch.view_as_real(data)), -1)
            ctf2 = torch.square(ctf)

            power_spectra = grid_spectral_average_torch(power, s_idx)
            ctf2_spectra = grid_spectral_average_torch(ctf2, s_idx)

            part_og_idx = self.part_og_idx[idx]

            for i in range(self.nr_optics_groups):
                mask = part_og_idx == i
                if torch.any(mask):
                    self.acc_data_counts[i] += 1
                    self.acc_power_spectra[i] += torch.mean(power_spectra[mask].detach(), 0)
                    self.acc_ctf2_spectra[i] += torch.mean(ctf2_spectra[mask].detach(), 0)

    def get_data_stats(self, optics_group_idx):
        if self.data_stats_established:
            return self.optics_groups[optics_group_idx]['data_amp'], \
                   self.optics_groups[optics_group_idx]['data_amp_ctf']
        if self.acc_data_counts[optics_group_idx] == 0:
            return torch.ones_like(self.acc_power_spectra[optics_group_idx]) * self.image_size, \
                   torch.ones_like(self.acc_power_spectra[optics_group_idx]) * self.image_size

        with torch.no_grad():
            power = self.acc_power_spectra[optics_group_idx] / self.acc_data_counts[optics_group_idx]
            ctf2 = self.acc_ctf2_spectra[optics_group_idx] / self.acc_data_counts[optics_group_idx]

            amp = torch.sqrt(power.float())
            amp_ctf = amp / (torch.sqrt(ctf2.float()) + 1e-3)

            return amp, amp_ctf

    def finalize_epoch(self):
        for k in self.ops:
            if self.do_optimize[k]:
                self.ops[k].step()

        if not self.data_stats_established:
            for i in range(self.nr_optics_groups):
                self.optics_groups[i]['data_amp'], self.optics_groups[i]['data_amp_ctf'] = self.get_data_stats(i)
            self.data_stats_established = True

    def get_by_index(self, idx: Tensor):
        batch_size = idx.shape[0]
        part_og_idx = self.part_og_idx[idx]

        # Poses #######################################
        pose_alpha = self.vars['pose_alpha'](idx)
        pose_beta = self.vars['pose_beta'](idx)
        pose_gamma = self.vars['pose_gamma'](idx)
        rot_matrices = euler_to_matrix(torch.stack([pose_alpha, pose_beta, pose_gamma], 1))
        # assert torch.all(is_rotation_matrix(rot_matrix))

        # Shifts #######################################
        shifts = torch.stack([self.vars['shift_x'](idx), self.vars['shift_y'](idx)], 1)

        # CTFs #########################################
        if self.do_ctf:
            defocus_a = self.vars['ctf_defocus_u'](idx)
            defocus_b = self.vars['ctf_defocus_v'](idx)
            angle = self.vars['ctf_angle'](idx)
            ctfs = torch.empty([batch_size, self.image_size, self.image_size]).to(self.device)
            for i in range(self.nr_optics_groups):
                mask = part_og_idx == i
                if torch.any(mask):
                    ctfs[mask] = self.optics_groups[i]['ctf'](
                        self.image_size,
                        self.optics_groups[i]['pixel_size'],
                        defocus_a[mask],
                        defocus_b[mask],
                        angle[mask]
                    )
        else:
            ctfs = torch.ones([batch_size, self.image_size, self.image_size]).to(self.device)

        # Sigma weighting and data std ##############################
        s_idx = Cache.get_spectral_indices(
            (self.image_size, self.image_size),
            max_r=self.image_size // 2,
            device=self.device
        )
        amp = torch.empty([batch_size, self.image_size, self.image_size]).to(self.device)
        amp_ctf = torch.empty([batch_size, self.image_size, self.image_size]).to(self.device)
        for i in range(self.nr_optics_groups):
            mask = part_og_idx == i
            if torch.any(mask):
                amp_spectra, amp_ctf_spectra = self.get_data_stats(i)
                # TODO cache this
                amp[mask] = spectra_to_grid_torch(spectra=amp_spectra, indices=s_idx).float()
                amp_ctf[mask] = spectra_to_grid_torch(spectra=amp_ctf_spectra, indices=s_idx).float()

        return {
            "rot_matrices": rot_matrices,
            "shifts": shifts,
            "ctfs": ctfs,
            "amp": amp.detach(),
            "amp_ctf": amp_ctf.detach()
        }

    def get_state_dict(self) -> Dict:
        var_states = {}
        orig = {}
        mean = {}
        norm = {}
        ops = {}
        optics_groups = []

        for k in self.vars:
            var_states[k] = self.vars[k].state_dict()
            orig[k] = self.vars[k].orig
            mean[k] = self.vars[k].mean
            norm[k] = self.vars[k].norm
            ops[k] = self.ops[k].state_dict()

        for og in self.optics_groups:
            og_ = {}
            for k in og:
                og_[k] = og[k]
            if self.do_ctf:
                og_['ctf'] = og['ctf'].get_state_dict()
            optics_groups.append(og_)

        return {
            "type": "HiddenVariableContainer",
            "version": "0.0.1",
            "var_states": var_states,
            "orig": orig,
            "mean": mean,
            "norm": norm,
            "op_states": ops,
            "op_learning_rates": self.op_learning_rates,
            'latent_space': self.latent_space.cpu().detach(),
            'structure_basis': self.structure_basis.cpu().detach(),
            'optics_groups': optics_groups,
            'part_og_idx': self.part_og_idx.cpu().detach(),
            'variance_spectra': self.variance_spectra.cpu(),
            'image_size': self.image_size,
            'nr_particles': self.nr_particles,
            'batch_size': self.batch_size,
            'representation_updated': self.representation_updated
        }

    @staticmethod
    def load_from_state_dict(state_dict, device=None):
        if "type" not in state_dict or state_dict["type"] != "HiddenVariableContainer":
            raise TypeError("Input is not an 'HiddenVariableContainer' instance.")

        if "version" not in state_dict:
            raise RuntimeError("HiddenVariableContainer instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            optics_groups = []
            for og in state_dict['optics_groups']:
                if 'ctf' in og:
                    og['ctf'] = ContrastTransferFunction.load_from_state_dict(og['ctf'])
                optics_groups.append(og)

            vars = {}
            for k in state_dict['var_states']:
                vars[k] = HiddenVariableModule(torch.zeros(state_dict['nr_particles']))
                vars[k].load_state_dict(state_dict['var_states'][k])
                vars[k].orig = state_dict['orig'][k]
                vars[k].norm = state_dict['norm'][k]
                vars[k].mean = state_dict['mean'][k]
            
            hvc = HiddenVariableContainer(
                vars=vars,
                op_learning_rates=state_dict['op_learning_rates'],
                latent_space=state_dict['latent_space'] if 'latent_space' in state_dict else None,
                structure_basis=state_dict['structure_basis'] if 'structure_basis' in state_dict else None,
                optics_groups=optics_groups,
                variance_spectra=state_dict['variance_spectra'],
                part_og_idx=state_dict['part_og_idx'],
                image_size=state_dict['image_size'],
                batch_size=state_dict['batch_size'],
                representation_updated=state_dict['representation_updated']
            )
            if device is not None:
                hvc.set_device(device)
            hvc.init_optimizers(
                op_states=state_dict['op_states']
            )
            return hvc
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
