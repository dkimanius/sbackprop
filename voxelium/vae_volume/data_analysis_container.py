#!/usr/bin/env python3

"""
Container for dataset analysis results
"""

import os
import shutil
import sys
import warnings

import numpy as np
from typing import Dict

import torch
from torch.utils.data import DataLoader

from voxelium import relion
from voxelium.base import get_spectral_indices
from voxelium.vae_volume.cache import Cache
from voxelium.vae_volume.distributed_processing import DistributedProcessing
from voxelium.vae_volume.hidden_variable_container import HiddenVariableContainer, HiddenVariableModule
from voxelium.vae_volume.utils import get_np_dtype

sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))

from voxelium.vae_volume.vae_container import VaeTrainContainer
from voxelium.base.particle_dataset import ParticleDataset


# Default hidden variable learning rates
HV_LEARNING_RATES = {
    'pose_alpha': 1e-3,
    'pose_beta': 1e-3,
    'pose_gamma': 1e-3,
    'shift_x': 1e-1,
    'shift_y': 1e-1,
    'ctf_defocus_u': 1e-2,
    'ctf_defocus_v': 1e-2,
    'ctf_angle': 1e-2
}


class DatasetAnalysisContainer:
    def __init__(
            self,
            vae_container: VaeTrainContainer,
            particle_dataset: ParticleDataset,
            hidden_variable_container: HiddenVariableContainer,
            train_step: int = 0,
            auxiliaries: Dict = None
    ) -> None:
        self.vae_container = vae_container
        self.particle_dataset = particle_dataset
        self.hidden_variable_container = hidden_variable_container
        self.train_step = train_step
        self.auxiliaries = auxiliaries
        self.device = vae_container.device

    @staticmethod
    def initialize_from_args(args, device=None):
        if device is None:
            device = DistributedProcessing.get_device()

        # Check if there is an existing checkpoint file
        found_checkpoint = os.path.isfile(os.path.join(args.log_dir, f"chkpt.pt"))

        # Load dataset
        if found_checkpoint:
            data_analysis_container = DatasetAnalysisContainer.load_from_logdir(args.log_dir, device=device)
            print("Found existing checkpoint file at training step", data_analysis_container.train_step)
            dataset = data_analysis_container.particle_dataset
        else:
            data_analysis_container = None
            dataset = relion.RelionDataset(
                args.input, 
                dtype=get_np_dtype(args.dtype),
                random_subset=str(args.random_subset) if args.random_subset is not None else args.random_subset,
            ).make_particle_dataset()

        optics_groups = dataset.get_optics_group_stats()
        image_size = optics_groups[0]['image_size']
        pixel_size = optics_groups[0]['pixel_size']

        if args.cache is not None:
            dataset.set_cache_root(args.cache)

        for og in optics_groups:
            if og['image_size'] != image_size:
                raise RuntimeError(f"Optics groups must have the same image size.\n"
                              f"But optics group '{og['id']}' has {og['image_size']} and "
                              f"'{optics_groups[0]['id']}' has {image_size}.")
            if og['pixel_size'] != pixel_size:
                warnings.warn(
                    f"Optics groups must have the same pixel size.\n"
                    f"But optics group '{og['id']}' has {og['pixel_size']} A and "
                    f"'{optics_groups[0]['id']}' has {pixel_size} A.",
                    RuntimeWarning
                )

        image_max_r = image_size // 2

        # Preload images
        if args.preload:
            print(f"Preloading images...")
            dataset.preload_images()

        if not found_checkpoint:
            # Image pre-processing
            max_diameter_ang = image_size * pixel_size - args.circular_mask_thickness

            if args.particle_diameter is None:
                diameter_ang = image_size * 0.75 * pixel_size - args.circular_mask_thickness
                print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
            else:
                if args.particle_diameter > max_diameter_ang:
                    print(
                        f"WARNING: Specified particle diameter {round(args.particle_diameter)} angstrom is too large\n"
                        f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
                    )
                    diameter_ang = max_diameter_ang
                else:
                    diameter_ang = args.particle_diameter

            dataset_size = len(dataset.part_rotation)

            ###############################################
            # VAE
            ###############################################
            print("Setting up model...")

            # Convert resolution to spectral index
            if args.encoder_mask_resolution is None or args.encoder_mask_resolution < 0:
                encoder_max_spectral_index = None
            elif args.encoder_mask_resolution == 0:
                encoder_max_spectral_index = image_size // 2 + 1
            else:
                encoder_max_spectral_index = round(image_size * pixel_size / args.encoder_mask_resolution)

            embedding_size = args.encoder_embedding_size if args.encoder_embedding_size is not None else -1

            input_grad_spectral_weight = None
            if args.spectral_weight_grad_res is not None:
                # Convert resolution to spectral index
                res_idx = round(image_size * pixel_size / args.spectral_weight_grad_res)
                input_grad_spectral_weight = torch.arange(0, image_max_r + 5) / float(res_idx)
                input_grad_spectral_weight = torch.pow(10, -input_grad_spectral_weight**2)

            vae_container = VaeTrainContainer(
                encoder_depth=args.encoder_depth,
                image_size=image_size,
                embedding_max_count=dataset.get_nr_noise_groups(),
                embedding_size=embedding_size,
                encoder_max_r=encoder_max_spectral_index,
                sb_input_size=args.sb_latent_size,
                pp_input_size=args.pp_latent_size,
                sb_output_size=args.sb_basis_size,
                basis_depth=args.basis_decoder_depth,
                basis_width=args.basis_decoder_width,
                structure_decoder_lr=args.structure_decoder_lr,
                input_spectral_weight=input_grad_spectral_weight
            )

            vae_container.set_device(device)
            vae_container.init_optimizers()

            ###############################################
            # HIDDEN VARIABLE CONTAINER
            ###############################################
            vars = {}
            optics_groups = []

            # Poses
            euler_angles = torch.Tensor(dataset.part_rotation).float().detach()
            vars['pose_alpha'] = HiddenVariableModule(euler_angles[:, 0], norm=np.pi, mean=0)
            vars['pose_beta'] = HiddenVariableModule(euler_angles[:, 1], norm=np.pi / 2,
                                                     mean=np.pi / 2)  # Tilt [0, 180]
            vars['pose_gamma'] = HiddenVariableModule(euler_angles[:, 2], norm=np.pi, mean=0)

            # Shifts
            shifts = torch.Tensor(dataset.part_translation).float().detach()
            vars['shift_x'] = HiddenVariableModule(shifts[:, 0], norm=1, mean=0)
            vars['shift_y'] = HiddenVariableModule(shifts[:, 1], norm=1, mean=0)

            og_stats = dataset.get_optics_group_stats()
            image_size = og_stats[0]['image_size']
            for i in range(len(og_stats)):
                if image_size != og_stats[i]['image_size']:
                    raise RuntimeError("All optics group image sizes must be the same.")

                optics_groups.append({
                    'id': og_stats[i]['id'],
                    'pixel_size': og_stats[i]['pixel_size']
                })

            # CTFs
            do_ctf = dataset.part_defocus is not None and not np.any(np.isnan(dataset.part_defocus))
            if do_ctf:
                ctf_functions = dataset.get_optics_group_ctfs()
                for i in range(len(ctf_functions)):
                    optics_groups[i]['ctf'] = ctf_functions[i]

                ctf_defocus = torch.Tensor(dataset.part_defocus[:, :2]).detach().float()
                defocus_norm = float(torch.std(ctf_defocus))
                defocus_mean = float(torch.mean(ctf_defocus))
                vars['ctf_defocus_u'] = HiddenVariableModule(ctf_defocus[:, 0], norm=defocus_norm, mean=defocus_mean)
                vars['ctf_defocus_v'] = HiddenVariableModule(ctf_defocus[:, 1], norm=defocus_norm, mean=defocus_mean)

                ctf_angle = torch.Tensor(dataset.part_defocus[:, 2]).detach().float()
                if torch.min(ctf_angle) >= 0:
                    vars['ctf_angle'] = HiddenVariableModule(ctf_angle, norm=360., mean=180.)
                else:
                    vars['ctf_angle'] = HiddenVariableModule(ctf_angle, norm=180., mean=0.)

            par_og_idx = torch.Tensor(dataset.part_og_idx)

            hidden_variable_container = HiddenVariableContainer(
                vars=vars,
                op_learning_rates=HV_LEARNING_RATES,
                optics_groups=optics_groups,
                part_og_idx=par_og_idx,
                image_size=image_size,
                batch_size=args.batch_size
            )
            hidden_variable_container.set_device(device)
            hidden_variable_container.init_optimizers()

            hidden_variable_container.set_latent_size(
                vae_container.sb_input_size + vae_container.pp_input_size)
            hidden_variable_container.set_structure_basis_size(
                vae_container.structure_decoder.get_input_size())

            ###############################################
            # FINALIZE
            ###############################################

            auxiliaries = {
                "image_size": image_size,
                "image_max_r": image_max_r,
                "pixel_size": pixel_size,
                "circular_mask_radius_ang": diameter_ang,
                "circular_mask_radius": diameter_ang / (2 * pixel_size),
                "circular_mask_thickness": args.circular_mask_thickness / pixel_size,
                "input_grad_spectral_weight": input_grad_spectral_weight
            }

            data_analysis_container = DatasetAnalysisContainer(
                vae_container=vae_container,
                particle_dataset=dataset,
                hidden_variable_container=hidden_variable_container,
                auxiliaries=auxiliaries
            )

        return data_analysis_container

    def set_device(self, device):
        self.vae_container.set_device(device)
        self.hidden_variable_container.set_device(device)
        self.device = device

    def get_state_dict(self) -> Dict:
        return {
            "type": "DatasetAnalysisContainer",
            "version": "0.0.1",

            "vae_container": self.vae_container.get_state_dict(),
            "particle_dataset": self.particle_dataset.get_state_dict(),
            "hidden_variable_container": self.hidden_variable_container.get_state_dict(),

            "train_step": self.train_step,

            "auxiliaries": self.auxiliaries
        }

    @staticmethod
    def load_from_state_dict(state_dict, device=None):
        if "type" not in state_dict or state_dict["type"] != "DatasetAnalysisContainer":
            raise TypeError("Input is not an 'DatasetAnalysisContainer' instance.")

        if "version" not in state_dict:
            raise RuntimeError("DatasetAnalysisContainer instance lacks version information.")

        if state_dict["version"] == "0.0.1":
            particle_dataset = ParticleDataset()

            vae_container = VaeTrainContainer.load_from_state_dict(state_dict["vae_container"], device=device)
            particle_dataset.set_state_dict(state_dict["particle_dataset"])
            hidden_variable_container = HiddenVariableContainer.load_from_state_dict(
                state_dict["hidden_variable_container"], device=device
            )

            return DatasetAnalysisContainer(
                vae_container=vae_container,
                particle_dataset=particle_dataset,
                hidden_variable_container=hidden_variable_container,
                train_step=state_dict["train_step"],
                auxiliaries=state_dict["auxiliaries"]
            )
        else:
            raise RuntimeError(f"Version '{state_dict['version']}' not supported.")

    def save_to_checkpoint(self, log_dir, epoch = None):
        state_dict = self.get_state_dict()
        checkpoint_fn = os.path.join(log_dir, f"chkpt{'' if epoch is None else '_' + str(epoch)}.pt")
        checkpoint_backup_fn = os.path.join(log_dir, f"chkpt_backup.pt")
        torch.save(state_dict, checkpoint_backup_fn)
        os.replace(checkpoint_backup_fn, checkpoint_fn)

    @staticmethod
    def load_from_logdir(path, device=None):
        if path[-3:] == ".pt":
            checkpoint_fn = path
            checkpoint_backup_fn = None
        else:
            checkpoint_fn = os.path.join(path, f"chkpt.pt")
            checkpoint_backup_fn = os.path.join(path, f"chkpt_backup.pt")
        try:  # Try loading main file
            state_dict = torch.load(checkpoint_fn, map_location="cpu")
        except Exception as e:  # Except all and try again with backup
            print(f"Failed to load main checkpoint file. will try to load backup file instead.")
            state_dict = torch.load(checkpoint_backup_fn, map_location="cpu")
        return DatasetAnalysisContainer.load_from_state_dict(state_dict, device=device)
