#!/usr/bin/env python3

"""
Model container for the VAE
"""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List, TypeVar, Dict, Union

import numpy as np
from voxelium.vae_volume.cache import Cache

from .distributed_processing import DistributedProcessing
from .optim import ExtendedAdam
from .svr_linear import make_compact_grid2d, make_grid3d
from ..base import ModelContainer, RadialBasisFunctions1D, spectra_to_grid_torch, grid_spectral_average_torch
from ..base.torch_utils import optimizer_to, optimizer_set_learning_rate
from voxelium.vae_volume.svr_linear.svr_linear import SparseVolumeReconstructionLinear

Tensor = TypeVar('torch.tensor')


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(torch.nn.Module):
    def __init__(
            self,
            depth,
            embedding_max_count,
            index_embedding_size,
            image_size,
            encoder_max_r,
            latent_size
    ) -> None:
        super(Encoder, self).__init__()

        self.do_index_embedding = index_embedding_size is not None
        self.do_image_embedding = encoder_max_r is not None
        self.embedding_max_count = embedding_max_count
        self.index_embedding_size = index_embedding_size
        self.image_size = image_size
        self.encoder_max_r = encoder_max_r
        self.latent_size = latent_size

        encoder_input_size = 0

        if self.do_index_embedding:
            self.embedding = torch.nn.Embedding(num_embeddings=embedding_max_count, embedding_dim=index_embedding_size)
            encoder_input_size += index_embedding_size

        if self.do_image_embedding:
            mask = Cache.get_encoder_input_mask(image_size, max_r=encoder_max_r)
            input_size = int(torch.sum(mask))
            output_size = 1024 - encoder_input_size
            self.feat_extract = torch.nn.Linear(input_size, output_size)
            encoder_input_size += output_size

        if encoder_input_size == 0:
            raise RuntimeError("Either input image or embedding has to be set for encoder.")

        self.model = ModelContainer({
            'type': 'ResidMLP',
            'input_dim': encoder_input_size,
            'resid_dim': min(encoder_input_size, 2048),
            'resid_count': depth,
            'output_dim': latent_size * 2,
            'activation': 'gelu'
        })

    def forward(self, index, image=None):
        input = None
        if self.do_index_embedding:
            input = self.embedding(index)
        if self.do_image_embedding:
            image_mask = Cache.get_encoder_input_mask(self.image_size, max_r=self.encoder_max_r)
            image_embedding = image[:, image_mask]
            image_embedding = self.feat_extract(image_embedding)
            if self.do_index_embedding:
                input = torch.cat([input, image_embedding], 1)
            else:
                input = image_embedding

        output = self.model(input)

        return output[:, :self.latent_size], \
               1 - torch.nn.functional.elu(output[:, self.latent_size:])  # Clamped at +2


class BasisMap(torch.nn.Module):
    def __init__(self, sb_size, pp_size):
        super().__init__()

        self.sb_a = torch.nn.Parameter(torch.ones((1, sb_size)))

        self.sb_b = torch.nn.Parameter(torch.zeros((1, sb_size)))
        self.pp = ModelContainer({
            'type': 'ResidMLP',
            'input_dim': pp_size,
            'resid_dim': 32,
            'resid_count': 3,
            'output_dim': 3,
            'activation': 'gelu'
        })

    def forward(self, sb, pp):
        return sb * self.sb_a + self.sb_b, self.pp(pp)


class BasisDecoder(torch.nn.Module):
    def __init__(self, sb_input_size, pp_input_size, sb_output_size, depth, width):
        super().__init__()

        if depth <= 1:
            self.sb = torch.nn.Linear(sb_input_size, sb_output_size)
            self.pp = torch.nn.Linear(pp_input_size, 2)
        else:
            self.sb = ModelContainer({
                'type': 'ResidMLP',
                'input_dim': sb_input_size,
                'resid_dim': sb_output_size * width,
                'resid_count': depth,
                'output_dim': sb_output_size,
                'activation': 'gelu'
            })
            self.pp = ModelContainer({
                'type': 'ResidMLP',
                'input_dim': pp_input_size,
                'resid_dim': 3 * width,
                'resid_count': depth,
                'output_dim': 3,
                'activation': 'gelu'
            })

    def forward(self, sb, pp):
        return self.sb(sb), self.pp(pp)


class StructureDecoder(torch.nn.Module):
    def __init__(self, grid3d_size, sb_input_size, input_spectral_weight=None):
        super().__init__()
        self.grid3d_size = grid3d_size
        self.sb_input_size = sb_input_size
        self.max_r = self.grid3d_size // 2
        if input_spectral_weight is None:
            self.input_spectral_weight = None
        else:
            self.input_spectral_weight = torch.nn.Parameter(input_spectral_weight)

        self.projector = SparseVolumeReconstructionLinear(
            size=grid3d_size,
            input_size=sb_input_size,
            input_spectral_weight=input_spectral_weight
        )

        self.spectral_factor = torch.nn.Parameter(torch.ones(self.max_r + 1, requires_grad=False))
        self.spectral_std = torch.nn.Parameter(torch.zeros(self.max_r + 1, requires_grad=False))

        self.caches = {}

    def _load_cache(self, max_r, is_3d):
        hashable = str(max_r) + ("_3d" if is_3d else "_2d")
        if hashable not in self.caches:
            device = self.projector.weight.device
            if is_3d:
                coord, mask = make_grid3d(max_r=max_r)
                radius = torch.sqrt(torch.sum(torch.square(coord), -1))
                spectral_idx = torch.round(radius).long()
                spectral_idx = spectral_idx[mask]
                nc_idx = torch.where(spectral_idx == 0)[0][0]

                self.caches[hashable] = [
                    mask.to(device),
                    spectral_idx.to(device),
                    nc_idx
                ]
            else:
                coord, mask = make_compact_grid2d(max_r=max_r)
                radius = torch.sqrt(torch.sum(torch.square(coord), -1))
                spectral_idx = torch.round(radius).long()
                nc_idx = torch.where(spectral_idx == 0)[0][0]

                self.caches[hashable] = [
                    coord.to(device),
                    mask.to(device),
                    spectral_idx.to(device),
                    nc_idx
                ]

        return self.caches[hashable]

    def get_postprocess_input_size(self):
        return 3

    def get_structure_basis_size(self):
        return self.sb_input_size

    def get_input_size(self):
        return self.get_postprocess_input_size() + self.get_structure_basis_size()

    def get_envelop(self, modulator, spectral_idx):
        return spectra_to_grid_torch(modulator, spectral_idx)

    def get_spectral_weighted_output_2d_mask_size(self, max_r=None):
        if max_r is None:
            max_r = self.max_r
        _, mask, spectral_idx, _ = self._load_cache(max_r, False)
        if self.input_spectral_weight is not None:
            weights = spectra_to_grid_torch(self.input_spectral_weight[:max_r + 1], spectral_idx)
            return weights.sum().detach()
        else:
            return mask.sum().detach()

    def forward(
            self,
            sb_input,
            pp_input,
            max_r=None,
            rot_matrices=None,
            sparse_grad=True,
            data_spectra=None,
            do_postprocess=False
    ):
        is_3d = rot_matrices is None

        if max_r is None:
            max_r = self.max_r

        if is_3d:
            mask, spectral_idx, nc_idx = self._load_cache(max_r, True)
            x_ft_ = self.projector(input=sb_input.contiguous(), max_r=max_r)
            x_ft = x_ft_[:, mask, :]

            if do_postprocess:
                bfac = (torch.nn.functional.elu(pp_input[:, 2] - 5) + 1)
                modulation = torch.exp(-bfac[:, None] * spectral_idx)
                x_ft[:, nc_idx, 0] = pp_input[:, 0]
                x_ft[:, nc_idx, 1] = 0
                x_ft *= pp_input[:, 1][:, None, None] * modulation[..., None]

            mod_spectra = self.spectral_factor.data
            if data_spectra is not None:
                mod_spectra = mod_spectra * data_spectra

            spectral_factor = spectra_to_grid_torch(mod_spectra, spectral_idx)
            x_ft *= spectral_factor[None, ..., None]

            image_size = max_r * 2 + 1
            x_ft_ = torch.zeros([x_ft.shape[0], image_size * image_size * (image_size // 2 + 1), 2]).to(x_ft.device)
            x_ft_[:, mask.flatten(), :] = x_ft
            x_ft = x_ft_.view(-1, image_size, image_size, image_size // 2 + 1, 2)
        else:
            coord, mask, spectral_idx, nc_idx = self._load_cache(max_r, False)

            x_ft = self.projector(
                input=sb_input.contiguous(),
                grid2d_coord=coord,
                rot_matrices=rot_matrices,
                max_r=max_r,
                sparse_grad=sparse_grad
            )

            if do_postprocess:
                bfac = (torch.nn.functional.elu(pp_input[:, 2] - 5) + 1)
                modulation = torch.exp(-bfac[:, None] * spectral_idx)
                x_ft[:, nc_idx, 0] = pp_input[:, 0]
                x_ft[:, nc_idx, 1] = 0
                x_ft *= pp_input[:, 1][:, None, None] * modulation[..., None]

            with torch.no_grad():
                if self.training:
                    b = 0.1
                    power = torch.mean(torch.sum(torch.square(x_ft), -1), 0)
                    s = torch.sqrt(grid_spectral_average_torch(power, spectral_idx)[0])
                    self.spectral_factor.data += (s - self.spectral_std.data) * 0.1
                    self.spectral_factor.data = torch.clip(self.spectral_factor.data, 1-6)
                    self.spectral_std.data = self.spectral_std.data * (1-b) + s * b

                mod_spectra = self.spectral_factor.data
                if data_spectra is not None:
                    mod_spectra = mod_spectra * data_spectra
                spectral_factor = spectra_to_grid_torch(mod_spectra, spectral_idx)

            x_ft *= spectral_factor[None, ..., None]

            image_size = max_r * 2 + 1
            x_ft_ = torch.zeros([x_ft.shape[0], image_size * (image_size // 2 + 1), 2]).to(x_ft.device)
            x_ft_[:, mask, :] = x_ft
            x_ft = x_ft_.view(-1, image_size, image_size // 2 + 1, 2)

        return x_ft

    def get_optim_params(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if name in self.no_decay_params:
                no_decay.append(m)
            else:
                decay.append(m)
        return [{'params': decay, 'weight_decay': 1e-3}, {'params': no_decay}]


class VaeTrainContainer:
    def __init__(
        self,
        image_size: int,
        encoder_depth: int,
        embedding_max_count: int,
        embedding_size: int,
        encoder_max_r: int,
        sb_input_size: int,
        pp_input_size: int,
        sb_output_size: int,
        basis_depth: int,
        basis_width: int,
        structure_decoder_lr: float,
        input_spectral_weight
    ) -> None:
        self.image_size = image_size
        self.grid3d_size = image_size + 1 - image_size % 2
        self.encoder_depth = encoder_depth
        self.embedding_max_count = embedding_max_count
        self.encoder_max_r = encoder_max_r if encoder_max_r is not None and encoder_max_r > 0 else None
        self.embedding_size = embedding_size if embedding_size is not None and embedding_size > 0 else None

        self.structure_decoder_lr = structure_decoder_lr
        self.input_spectral_weight = input_spectral_weight

        self.device = torch.device('cpu')

        if basis_depth == 0:
            pp_input_size = 2

        # Encoder
        self.encoder = Encoder(
            depth=encoder_depth,
            embedding_max_count=embedding_max_count,
            index_embedding_size=self.embedding_size,
            image_size=image_size,
            encoder_max_r=self.encoder_max_r,
            latent_size=sb_input_size + pp_input_size
        )
        self.encoder_ddp = DistributedProcessing.setup_module(self.encoder)

        # Basis Decoder
        if basis_depth == 0:
            self.basis_decoder = BasisMap(sb_input_size, pp_input_size)
            sb_output_size = sb_input_size
        else:
            self.basis_decoder = BasisDecoder(
                sb_input_size,
                pp_input_size,
                sb_output_size,
                basis_depth,
                basis_width
            )
        self.basis_decoder_ddp = DistributedProcessing.setup_module(self.basis_decoder)

        # Structure Decoder
        self.structure_decoder = StructureDecoder(
            grid3d_size=self.grid3d_size,
            sb_input_size=sb_output_size,
            input_spectral_weight=input_spectral_weight
        )
        self.structure_decoder_ddp = DistributedProcessing.setup_module(self.structure_decoder)

        self.encoder_opt = None
        self.basis_decoder_opt = None
        self.structure_decoder_opt = None

        self.sb_input_size = sb_input_size
        self.pp_input_size = pp_input_size

        self.sb_output_size = sb_output_size

        self.basis_depth = basis_depth
        self.basis_width = basis_width

    def init_optimizers(self):
        self.encoder_opt = torch.optim.Adam(self.encoder_ddp.parameters(), lr=1e-4)
        self.basis_decoder_opt = torch.optim.Adam(self.basis_decoder_ddp.parameters(), lr=1e-4, weight_decay=1e-3)
        self.structure_decoder_opt = ExtendedAdam(self.structure_decoder_ddp.parameters(), lr=self.structure_decoder_lr)

    def set_device(self, device):
        self.encoder_ddp = self.encoder_ddp.to(device)
        self.basis_decoder_ddp = self.basis_decoder_ddp.to(device)
        self.structure_decoder_ddp = self.structure_decoder_ddp.to(device)
        self.device = device

    def set_train(self):
        self.encoder_ddp.train()
        self.basis_decoder_ddp.train()
        self.structure_decoder_ddp.train()

    def set_eval(self):
        self.encoder_ddp.eval()
        self.basis_decoder_ddp.eval()
        self.structure_decoder_ddp.eval()

    def zero_grad(self):
        self.encoder_opt.zero_grad()
        self.basis_decoder_opt.zero_grad()
        self.structure_decoder_opt.zero_grad()

    def step(self):
        self.encoder_opt.step()
        self.basis_decoder_opt.step()
        self.structure_decoder_opt.step()

    def split_latent_space(self, latent_space):
        return latent_space[:, :self.sb_input_size], latent_space[:, self.sb_input_size:]

    def get_sb_latent_indices(self):
        return list(np.arange(self.sb_input_size + self.pp_input_size)[:self.sb_input_size])

    def get_pp_latent_indices(self):
        return list(np.arange(self.sb_input_size + self.pp_input_size)[self.sb_input_size:])

    def encoder_image_embedding_required(self):
        return self.encoder.do_image_embedding

    def encoder_index_embedding_required(self):
        return self.encoder.do_index_embedding

    def count_parameters(self):
        encoder_count = self.encoder.model.count_params()
        for p in self.encoder.feat_extract.parameters():
            encoder_count += p.numel()
        decoder_count = self.structure_decoder.projector.weight.numel() + \
                        self.structure_decoder.projector.bias.numel()

        return encoder_count, decoder_count

    def get_state_dict(self) -> Dict:
        return {
            "type": "VaeModelContainer",
            "version": "0.0.1",
            "encoder_depth": self.encoder_depth,
            "encoder_max_r": self.encoder_max_r,
            "image_size": self.image_size,
            "embedding_max_count": self.embedding_max_count,
            "embedding_size": self.embedding_size,
            "encoder": self.encoder.state_dict(),
            "basis_decoder": self.basis_decoder.state_dict(),
            "structure_decoder": self.structure_decoder.state_dict(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "basis_decoder_opt": self.basis_decoder_opt.state_dict(),
            "structure_decoder_opt": self.structure_decoder_opt.state_dict(),
            "sb_input_size": self.sb_input_size,
            "pp_input_size": self.pp_input_size,
            "sb_output_size": self.sb_output_size,
            "basis_depth": self.basis_depth,
            "basis_width": self.basis_width,
            "structure_decoder_lr": self.structure_decoder_lr,
            "input_spectral_weight": self.input_spectral_weight
        }

    @staticmethod
    def load_from_state_dict(state_dict, device=None):
        if "type" not in state_dict or state_dict["type"] != "VaeModelContainer":
            raise TypeError("Input is not an 'VaeModelContainer' instance.")

        if "version" not in state_dict:
            raise RuntimeError("VaeModelContainer instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            vae_container = VaeTrainContainer(
                encoder_depth=state_dict["encoder_depth"],
                encoder_max_r=state_dict["encoder_max_r"],
                image_size=state_dict["image_size"],
                embedding_max_count=state_dict["embedding_max_count"],
                embedding_size=state_dict["embedding_size"],
                sb_input_size=state_dict["sb_input_size"],
                pp_input_size=state_dict["pp_input_size"],
                sb_output_size=state_dict["sb_output_size"],
                basis_depth=state_dict["basis_depth"],
                basis_width=state_dict["basis_width"],
                structure_decoder_lr=state_dict["structure_decoder_lr"],
                input_spectral_weight=state_dict["input_spectral_weight"]
            )
            vae_container.encoder.load_state_dict(state_dict["encoder"])
            vae_container.basis_decoder.load_state_dict(state_dict["basis_decoder"])
            vae_container.structure_decoder.load_state_dict(state_dict["structure_decoder"])

            if device is not None:
                vae_container.set_device(device)

            vae_container.init_optimizers()
            vae_container.encoder_opt.load_state_dict(state_dict["encoder_opt"])
            vae_container.basis_decoder_opt.load_state_dict(state_dict["basis_decoder_opt"])
            vae_container.structure_decoder_opt.load_state_dict(state_dict["structure_decoder_opt"])

            return vae_container
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")
