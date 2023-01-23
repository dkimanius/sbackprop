#!/usr/bin/env python
import sys
import time

import umap
import numpy as np
import argparse

import scipy.ndimage
import matplotlib.pylab as plt

import torch

import multiprocessing as mp

from voxelium.base.grid import idft, dt_desymmetrize, load_mrc, save_mrc, get_fsc_torch, spectral_correlation_torch, \
    get_spectral_indices, dft
from voxelium.vae_volume.data_analysis_container import DatasetAnalysisContainer
from voxelium.vae_volume.utils import setup_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help='input checkpoint file', type=str)
    parser.add_argument('ordered_ground_truth_files',
                        help='comma separated paths to ground truth files, in order', type=str)
    parser.add_argument('--gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--dont_cache_embed', action="store_true")
    parser.add_argument('--ignore_cached_embed', action="store_true")
    parser.add_argument('--mask', help='input mask file', type=str, default=None)
    args = parser.parse_args()

    torch.no_grad()
    device, _ = setup_device(args)
    for epoch in range(1, 50):
        dac = DatasetAnalysisContainer.load_from_logdir(args.logdir + f"chkpt_{epoch}.pt")
        latent = dac.hidden_variable_container.latent_space.numpy().astype(np.float32)
        # embed = dac.hidden_variable_container.structure_basis.softmax(dim=-1).numpy().astype(np.float32)
        embed = dac.hidden_variable_container.latent_space.numpy().astype(np.float32)

        if latent.shape[-1] > 2:
            if dac.auxiliaries is None or "embed" not in dac.auxiliaries or args.ignore_cached_embed:
                print("Creating 2D embedding...")

                basis = embed
                # basis = embed[:, :16]

                embed = umap.UMAP(local_connectivity=1, repulsion_strength=2).fit_transform(
                    basis.astype(np.float32)
                )

                # import tsnecuda
                # embed = tsnecuda.TSNE(n_components=2, num_neighbors=256, perplexity=100).fit_transform(
                #     basis.astype(np.float32)
                # )

                # from sklearn.decomposition import PCA
                # embed = PCA(n_components=2).fit_transform(
                #     basis.astype(np.float32)
                # )

                if dac.auxiliaries is None:
                    dac.auxiliaries = {"embed": embed}
                else:
                    dac.auxiliaries["embed"] = embed
                if not args.dont_cache_embed:
                    print("Saving embedding to checkpoint file...")
                    dac.save_to_checkpoint(args.logdir)
            else:
                embed = dac.auxiliaries["embed"]

        # Else, import what the GUI needs

        import vtkmodules.vtkInteractionStyle
        # noinspection PyUnresolvedReferences
        import vtkmodules.vtkRenderingOpenGL2

        N = 3000
        margin = 50
        marker_size = 10

        outlier_mask = np.abs(embed) > 5
        if np.sum(outlier_mask) < len(embed) * 0.3:
            embed[outlier_mask] = np.sign(embed[outlier_mask]) * 5

        x_min = np.min(embed[:, 0])
        x_max = np.max(embed[:, 0])
        y_min = np.min(embed[:, 1])
        y_max = np.max(embed[:, 1])

        x = (embed[:, 0] - x_min) / (x_max - x_min)
        y = (embed[:, 1] - y_min) / (y_max - y_min)

        x = margin + x * (N - 2. * margin)
        y = margin + y * (N - 2. * margin)

        heat_map = np.zeros((N, N))
        heat_map[y.astype(int), x.astype(int)] += 1
        heat_map_smooth = scipy.ndimage.gaussian_filter(heat_map, 3)

        coord = np.zeros((len(x), 2))
        coord[:, 0] = x
        coord[:, 1] = y

        vaec = dac.vae_container
        vaec.set_device(device)
        vaec.set_eval()

        data_spectra, data_ctf_spectra = dac.hidden_variable_container.get_data_stats(0)
        data_ctf_spectra = data_ctf_spectra.to(device)

        nn_time = 0
        ft_time = 0


        def get_volume(z):
            global nn_time, ft_time
            z = torch.Tensor(z).unsqueeze(0).to(device)

            t = time.time()
            sb, pp = vaec.basis_decoder(*vaec.split_latent_space(z))
            v_ft = vaec.structure_decoder(sb, pp, data_spectra=data_ctf_spectra, do_postprocess=True)
            nn_time = nn_time * 0.9 + (time.time() - t) * 0.1

            v_ht = torch.view_as_complex(v_ft)
            v_ht = dt_desymmetrize(v_ht)

            t = time.time()
            vol = idft(v_ht, dim=3, real_in=True)
            ft_time = ft_time * 0.9 + (time.time() - t) * 0.1

            vol /= torch.std(vol)

            print("NN time", nn_time, "FT time", ft_time)

            return vol[0]

        mask = None
        if args.mask is not None:
            mask, _, _ = load_mrc(args.mask)
            mask = torch.from_numpy(mask.copy())
        

        ground_truth_path = args.ordered_ground_truth_files.split(",")
        ground_truth_count = len(ground_truth_path)
        ground_truth_grids = []
        print(f"Number of ground truth files {len(ground_truth_path)}")
        for i in range(ground_truth_count):
            grid, _, _ = load_mrc(ground_truth_path[i].strip())
            ground_truth_grids.append(torch.from_numpy(grid.copy()))

        print("Calculate FSCs")
        count_per_ground_truth = 3
        selected_idx = (np.linspace(0, 0.999, ground_truth_count * count_per_ground_truth) * len(latent)).astype(int)
        fscs = []
        sidx = torch.Tensor(get_spectral_indices(ground_truth_grids[0].shape))
        for i in range(len(selected_idx)):
            j = i // count_per_ground_truth
            idx = selected_idx[i]
            print(i, j, idx)
            vol = get_volume(latent[idx])
            if mask is not None:
                vol *= mask
            gt = dft(ground_truth_grids[j], dim=3)
            save_mrc(vol, 1, [0, 0, 0], f"first_dumps/idx_{i}.mrc")
            vol = dft(vol, dim=3)
            fsc = spectral_correlation_torch(vol, gt, sidx, normalize=True)
            fscs.append(fsc)

        fscs = torch.stack(list(fscs), 0)
        means = torch.mean(fscs, 0).detach().cpu().numpy().round(2)
        stds = torch.std(fscs, 0).detach().cpu().numpy().round(3)
        
        np.save(f"sb_fsc_{epoch}", fscs.numpy())

        


