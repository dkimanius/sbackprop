#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import os
import sys
from typing import List, TypeVar, Dict

import numpy as np
import matplotlib
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

from voxelium.base import dt_desymmetrize, idft, idht, dt_symmetrize, grid_spectral_average_torch
from voxelium.base.torch_utils import make_imshow_fig, make_scatter_fig, make_line_fig, make_series_line_fig
from voxelium.vae_volume.cache import Cache
from voxelium.vae_volume.data_analysis_container import DatasetAnalysisContainer
from voxelium.vae_volume.hidden_variable_container import HiddenVariableContainer

Tensor = TypeVar('torch.tensor')


def tesnor_to_np(tensor):
    return tensor.detach().cpu().numpy()


class TensorboardSummary:
    def __init__(self, logdir, pixel_size, spectral_size, step=0):
        self.summary_fn = os.path.join(logdir, "summary")
        self.summary = SummaryWriter(self.summary_fn)
        self.spectral_size = spectral_size
        self.step = step
        self.pixel_size = pixel_size

    def set_step(self, step: int = None):
        self.step = self.step + 1 if step is None else step

    def write_losses(
            self,
            train_mse,
            kl_weight, kld_loss, data_loss,
            train_loss, roi_weight, roi_loss
    ):
        kld_loss = torch.mean(kld_loss)
        data_loss = torch.mean(data_loss)
        train_loss = torch.mean(train_loss)
        self.summary.add_scalar(f"Loss/KLD weight", kl_weight, self.step)
        self.summary.add_scalar(f"Loss/KLD", kld_loss.detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Loss/Data", data_loss.detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Loss/Train", train_loss.detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Loss/Train MSE", train_mse.detach().cpu().numpy(), self.step)
        if roi_weight > 0:
            self.summary.add_scalar(f"Loss/ROI", roi_loss.detach().cpu().numpy(), self.step)
            self.summary.add_scalar(f"Loss/ROI wight", roi_weight, self.step)

    def write_stats(self, x_ft, y_ft, data_amp, data_ctf_amp):
        if x_ft.shape[-1] != 2:
            x_ft = torch.view_as_real(x_ft)
        if y_ft.shape[-1] != 2:
            y_ft = torch.view_as_real(y_ft)
        self.summary.add_scalar(f"Stats/X mean", torch.mean(x_ft).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/X std", torch.std(x_ft).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/Y mean", torch.mean(y_ft).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/Y std", torch.std(y_ft).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/data amp mean", torch.mean(data_amp).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/data amp std", torch.std(data_amp).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/data ctf amp mean", torch.mean(data_ctf_amp).detach().cpu().numpy(), self.step)
        self.summary.add_scalar(f"Stats/data ctf amp std", torch.std(data_ctf_amp).detach().cpu().numpy(), self.step)

    def write_images(self, x_ft, y_ft, ctf, roi=None):
        x_ft = x_ft.detach()
        y_ft = y_ft.detach()
        ctf = ctf.detach()

        y_ft_ = torch.abs(torch.view_as_complex(y_ft[0])).detach().cpu().numpy()
        c_ = torch.abs(ctf[0].detach()).data.cpu().numpy()
        c_std = np.std(c_)
        y_ft_std = np.std(y_ft_)
        if c_std != 0 and y_ft_std != 0:
            c_ /= c_std
            y_ft_ /= y_ft_std
            y_ft_[:c_.shape[0] // 2] = c_[:c_.shape[0] // 2]
        self.summary.add_figure(f"Data/HT", make_imshow_fig(y_ft_), self.step)

        x_ft_ = dt_desymmetrize(torch.view_as_complex(x_ft[0]), dim=2).detach().cpu().numpy()
        self.summary.add_figure(f"Output/FT", make_imshow_fig(np.abs(x_ft_)), self.step)

        x_ = idft(x_ft_, dim=2, real_in=True)
        y_ = idft(y_ft[0].detach(), dim=2).real.cpu().numpy()
        self.summary.add_figure(f"Output/Image", make_imshow_fig(x_), self.step)
        self.summary.add_figure(f"Data/Image", make_imshow_fig(y_), self.step)

        if roi is not None:
            if roi.roi_slice is not None:
                self.summary.add_figure(f"ROI", make_imshow_fig(roi.roi_slice), self.step)
            if roi.roni_slice is not None:
                self.summary.add_figure(f"RONI", make_imshow_fig(roi.roni_slice), self.step)

    def write_hidden_variable(self, hvc: HiddenVariableContainer):
        vars = hvc.vars

        v = torch.stack([vars['pose_alpha'].vars, vars['pose_beta'].vars, vars['pose_gamma'].vars], 1)
        o = torch.stack([vars['pose_alpha'].orig, vars['pose_beta'].orig, vars['pose_gamma'].orig], 1)
        e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
        self.summary.add_scalar(f"Hidden variables/pose error", e, self.step)

        v = torch.stack([vars['shift_x'].vars, vars['shift_y'].vars], 1)
        o = torch.stack([vars['shift_x'].orig, vars['shift_y'].orig], 1)
        e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
        self.summary.add_scalar(f"Hidden variables/shift error", e, self.step)

        if hvc.do_ctf:
            v = torch.stack([vars['ctf_defocus_u'].vars, vars['ctf_defocus_v'].vars], 1)
            o = torch.stack([vars['ctf_defocus_u'].orig, vars['ctf_defocus_v'].orig], 1)
            e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
            self.summary.add_scalar(f"Hidden variables/ctf defocus error", e, self.step)

            v = vars['ctf_angle'].vars
            o = vars['ctf_angle'].orig
            e = torch.sqrt(F.mse_loss(v, o.to(v.device))).cpu().detach().item()
            self.summary.add_scalar(f"Hidden variables/ctf angle error", e, self.step)

        if hvc.latent_space is not None and torch.any(hvc.latent_space != 0):
            latent_space = hvc.latent_space.detach().cpu().numpy()
            if latent_space.shape[-1] == 2:
                embed = latent_space
            else:
                embed = PCA(n_components=2).fit_transform(
                    latent_space.astype(np.float32)
                )

                # import tsnecuda
                # device_id = hvc.latent_space.device.index
                # embed = tsnecuda.TSNE(n_components=2, device=device_id).fit_transform(
                #     latent_space.astype(np.float32)
                # )
            fig = make_scatter_fig(embed[:, 0], embed[:, 1])
            self.summary.add_figure(
                f"Latent/Train",
                fig,
                self.step
            )
            #fig.savefig(f"{self.summary_fn}/latent_fig_{self.step}.svg")


        # if hvc.structure_basis is not None and torch.any(hvc.structure_basis != 0):
        #     structure_basis = hvc.structure_basis.detach().cpu().numpy()
        #     if structure_basis.shape[-1] == 2:
        #         embed = structure_basis
        #     else:
        #         embed = PCA(n_components=2).fit_transform(
        #             structure_basis.astype(np.float32)
        #         )
        #
        #     self.summary.add_figure(
        #         f"Structure basis/Train",
        #         make_scatter_fig(
        #             embed[:, 0],
        #             embed[:, 1]
        #         ),
        #         self.step
        #     )