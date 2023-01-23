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

from voxelium.base.grid import idft, dt_desymmetrize, load_mrc, save_mrc
from voxelium.vae_volume.data_analysis_container import DatasetAnalysisContainer
from voxelium.vae_volume.utils import setup_device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help='input checkpoint file', type=str)
    parser.add_argument('--gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--roi', type=str, default=None, help='Region of interest')
    parser.add_argument('--dont_cache_embed', action="store_true")
    parser.add_argument('--ignore_cached_embed', action="store_true")
    parser.add_argument('--nogui', action="store_true")
    args = parser.parse_args()

    torch.no_grad()
    device, _ = setup_device(args)
    dac = DatasetAnalysisContainer.load_from_logdir(args.logdir)
    latent = dac.hidden_variable_container.latent_space.numpy().astype(np.float32)
    # embed = dac.hidden_variable_container.structure_basis.softmax(dim=-1).numpy().astype(np.float32)
    embed = dac.hidden_variable_container.latent_space.numpy().astype(np.float32)


    selection = np.arange(len(latent))
    # np.random.shuffle(selection)
    # selection = selection[:min(len(latent), 10000)]

    latent = latent[selection]
    embed = embed[selection]
    if latent.shape[-1] > 2:
        if dac.auxiliaries is None or "embed" not in dac.auxiliaries or args.ignore_cached_embed:
            print("Creating 2D embedding...")
            # if "roi_basis_index" in dac.auxiliaries:
            #     basis = embed[:, dac.auxiliaries["roi_basis_index"]]
            # else:
            #     basis = embed

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

    if args.nogui:
        print("No GUI... Exiting!")
        exit(0)

    # Else, import what the GUI needs
    
    import vtkmodules.vtkInteractionStyle
    # noinspection PyUnresolvedReferences
    import vtkmodules.vtkRenderingOpenGL2

    from voxelium.vae_volume.volume_renderer import volumeRendererProcessLoop

    roi = None
    if args.roi is not None:
        roi, _, _ = load_mrc(args.roi)
        roi = torch.Tensor(roi.copy()).to(device)

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

        if roi is not None:
            vol *= roi
        vol /= torch.std(vol)

        print("NN time", nn_time, "FT time", ft_time)

        return vol[0].detach().cpu().numpy()

    # print("dumping first 800...")
    # count = 50
    # selected_idx = np.arange(800)
    # np.random.shuffle(selected_idx)
    # selected_idx = selected_idx[:count]
    # avg = None
    # for i in range(count):
    #     vol = get_volume(latent[selected_idx[i]])
    #     save_mrc(vol, 1, [0, 0, 0], f"first_dumps/idx_{selected_idx[i]}.mrc")
    #     if avg is None:
    #         avg = vol
    #     else:
    #         avg += vol
    # save_mrc(avg / count, 1, [0, 0, 0], f"first_dumps/avg.mrc")
    # exit(0)

    fig_hm, ax_hm = plt.subplots(figsize=(7, 7))  # Heat map

    ax_hm.axis('off')
    plt.tight_layout()

    # MAKE HEAT MAP -------------------------------------------------------------------------------------
    alpha = min(10./np.sqrt(len(x)), 1.)
    # ax_hm.imshow(heat_map_smooth, cmap='RdPu', zorder=0)
    ax_hm.plot(x, y, 'k.', markersize=1, alpha=0.1, zorder=1)
    # ax_hm.scatter(x, y, edgecolors=None, marker='.', c=np.arange(len(x)), cmap="summer", alpha=alpha)

    mx = np.mean(x)
    sx = np.std(x)*3
    my = np.mean(y)
    sy = np.std(y)*3

    ax_hm.set_xlim([mx - sx, mx + sx])
    ax_hm.set_ylim([my - sy, my + sy])

    # VOLUME RENDERER PROCESS ---------------------------------------------------------------------------
    process_loop_queue = mp.Queue()
    volume_render_dispatch_queue = mp.Queue()
    volume_render_process = mp.Process(
        target=volumeRendererProcessLoop,
        args=(
            volume_render_dispatch_queue,
            process_loop_queue
        )
    )
    volume_render_process.start()

    # SELECTION UPDATES --------------------------------------------------------------------------------
    circles = []
    circles_coord = []
    selected_ids = []
    volumes = []
    selected_iso_value = None

    get_volume(np.mean(latent, axis=0))  # Warm up

    def onClickHm(event):
        if event.xdata is None or event.ydata is None:
            return

        xy = np.array([event.xdata, event.ydata])

        state_change = False

        if event.button == 1:
            c = np.sum((coord - xy) ** 2, axis=1)
            idx = np.argmin(c)
            dis2 = c[idx]
            xy = [coord[idx, 0], coord[idx, 1]]

            if dis2 < (N/50)**2 and idx not in selected_ids:
                circle = plt.Circle(xy, marker_size, color='black', alpha=1, zorder=2)
                vol = get_volume(latent[idx])
                circles.append(circle)
                circles_coord.append(xy)
                selected_ids.append(idx)
                volumes.append(vol)
                print("Selected point index", idx)
                ax_hm.add_patch(circle)
                state_change = True

        elif event.button == 3:
            if len(circles) > 0:
                c = np.sum((np.array(circles_coord) - xy) ** 2, axis=1)
                selected_idx = np.argmin(c)
                dis2 = c[selected_idx]

                if dis2 < (N/50)**2:
                    circles[selected_idx].remove()
                    del (circles[selected_idx])
                    del (circles_coord[selected_idx])
                    del (selected_ids[selected_idx])
                    del (volumes[selected_idx])
                    state_change = True

        if state_change:
            volume_render_dispatch_queue.put(volumes)
            fig_hm.canvas.draw()


    def onKeyHm(event):
        sys.stdout.flush()
        if event.key == 'escape':
            for i in range(len(circles)):
                circles[i].remove()

            circles.clear()
            circles_coord.clear()
            selected_ids.clear()
            volumes.clear()

            volume_render_dispatch_queue.put(volumes)
            fig_hm.canvas.draw()
        elif event.key == 'enter':
            print('Saving selected structures to MRC-files')
            for i, v in enumerate(volumes):
                save_mrc(v, 1, [0, 0, 0], "particle_id_" + str(selected_ids[i]) + ".mrc")

    #  --------------------------------------------------------------------------------

    click_connect = fig_hm.canvas.mpl_connect('button_press_event', onClickHm)
    key_connect = fig_hm.canvas.mpl_connect('key_press_event', onKeyHm)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Exiting!")

    volume_render_dispatch_queue.put(None)
    process_loop_queue.put(None)
    volume_render_process.join()
    volume_render_process.terminate()
