#!/usr/bin/env python3

"""
Test module for a training VAE
"""

import time
from datetime import datetime

from torch.utils.data import DataLoader, Sampler

from voxelium.vae_volume.train_arguments import parse_training_args
from voxelium.vae_volume.vae_container import reparameterize

from voxelium.base.grid_rotations import rotate_2d
from voxelium.vae_volume.distributed_processing import DistributedProcessing
from voxelium.vae_volume.region_of_interest import RegionOfInterest
from voxelium.vae_volume.tensorboard_summary import TensorboardSummary
from voxelium.vae_volume.utils import *
from voxelium.base import load_mrc
from voxelium.base.grid import dht, fourier_shift_2d, bilinear_shift_2d, dt_symmetrize, dft, idft, dt_desymmetrize
from voxelium.base.io_logger import IOLogger
from voxelium.vae_volume.data_analysis_container import DatasetAnalysisContainer
from voxelium.vae_volume.svr_linear import make_grid3d

END_LEARNING_RATE = 5e-3
BEGIN_LEARNING_RATE = 5e-3
STOP_LEARNING_RATE_DECAY_STEP = 100000  # steps


def get_encoding(vaec, y, group_idx, hv, circular_mask_thickness, circular_mask_radius, reparam=False):
    image_embedding = bilinear_shift_2d(y, hv["shifts"].detach())
    image_embedding = Cache.apply_circular_mask(
        image_embedding, thickness=circular_mask_thickness, radius=circular_mask_radius)
    image_embedding = dht(image_embedding, dim=2)
    image_embedding = image_embedding / (hv["amp"] + 1e-6)
    image_embedding = image_embedding.detach() * torch.sign(hv["ctfs"].detach())
    image_embedding = rotate_2d(
        image_embedding.unsqueeze(1),
        hv["rot_matrices"][:, :2, :2].detach(),
        inverse=False
    ).squeeze(1)
    image_size = vaec.encoder.image_size
    encoder_max_r = vaec.encoder.encoder_max_r
    image_embedding[:, ~Cache.get_encoder_input_mask(image_size, max_r=encoder_max_r)] = 0
    mu, log_var = vaec.encoder_ddp(index=group_idx, image=image_embedding)

    z = reparameterize(mu=mu, log_var=log_var) if reparam else mu
    sb, pp = vaec.basis_decoder(*vaec.split_latent_space(z))

    return {"mu": mu, "log_var": log_var, "z": z, "sb": sb, "pp": pp}


def run_batch_header(sample, dac, device=torch.device("cpu"), reparam=False):
    hvc = dac.hidden_variable_container
    vaec = dac.vae_container
    circular_mask_radius = dac.auxiliaries['circular_mask_radius']
    circular_mask_thickness = dac.auxiliaries['circular_mask_thickness']
    y = sample['image'].to(device)
    group_idx = sample['noise_group_idx'].to(device)
    y = Cache.apply_square_mask(y, thickness=circular_mask_thickness)
    part_idx = sample['idx'].to(device)

    hv = hvc.get_by_index(part_idx)  # Get hidden variables
    enc = get_encoding(
        vaec, y, group_idx, hv, circular_mask_thickness, circular_mask_radius, reparam)
    hvc.set_latent(sample['idx'], enc["mu"].detach())
    hvc.set_structure_basis(sample['idx'], torch.cat([enc["sb"], enc["pp"]], 1).detach())

    return y, part_idx, hv, enc


def update_representation(dac, data_loader, device=torch.device("cpu")):
    with torch.no_grad():
        for batch_ndx, sample in enumerate(data_loader):
            run_batch_header(sample, dac, device, reparam=False)
    dac.hidden_variable_container.representation_updated = True


def main(rank, args, ddp_args):
    ###############################################
    # SETUP
    ###############################################

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # warnings.simplefilter('error', UserWarning)
    # torch.autograd.set_detect_anomaly(True)

    if args.only_update_representation:
        args.overwrite = False
        print("Will ignore --overwrite, since updating representation.")

    DistributedProcessing.process_setup(rank=rank, args=ddp_args)

    log_dir = args.log_dir
    torch.set_num_threads(args.pytorch_threads)  # TODO manage in distributed process flow

    device = DistributedProcessing.get_device()
    dac = DatasetAnalysisContainer.initialize_from_args(args, device=device)

    # Setup shortcuts
    vaec = dac.vae_container
    hvc = dac.hidden_variable_container
    batch_size = args.batch_size
    dataset = dac.particle_dataset
    dataset_size = len(dataset)
    circular_mask_radius = dac.auxiliaries['circular_mask_radius']
    circular_mask_thickness = dac.auxiliaries['circular_mask_thickness']
    image_size = dac.auxiliaries["image_size"]
    pixel_size = dac.auxiliaries["pixel_size"]
    image_max_r = dac.auxiliaries["image_max_r"]

    vaec.set_train()
    if args.do_align:
        hvc.do_align()
    if args.do_ctf_optimization:
        hvc.do_ctf_optimization()

    class RandomSampler(Sampler[int]):
        def __init__(self, data_source, batch_size) -> None:
            super().__init__(data_source)
            idx = np.arange(len(data_source))
            np.random.shuffle(idx)
            idx1 = idx.copy()
            np.random.shuffle(idx)
            self.data_loader_indices = np.concatenate([idx1, idx])

            self.batch_size = batch_size
            self.dataset_size = len(self.data_loader_indices)
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.current_index + self.batch_size < self.dataset_size:
                idx = self.data_loader_indices[self.current_index:self.current_index + self.batch_size].copy()
                self.current_index += self.batch_size
            else:
                idx = self.data_loader_indices[self.current_index:].copy()
                # np.random.shuffle(self.data_loader_indices)
                self.current_index = 0

            return idx.tolist()

        def __len__(self) -> int:
            return 1e9 #np.ceil(float(self.dataset_size) / float(self.batch_size))

    dataset.setup_ctfs(compute_ctf=False)
    data_loader = DataLoader(
        dataset=dataset,
        # batch_sampler=RandomSampler(dataset, batch_size),
        batch_size=batch_size,
        num_workers=args.dataloader_threads,
        shuffle=True
    )

    ###############################################
    # SETUP OPTIMIZATION STATS
    ###############################################

    print("\nINITIALIZING TRAINING", datetime.now().ctime())

    timing_shown_count = 0

    print("Number of images:", dataset_size)
    print("Number of noise groups:", dataset.get_nr_noise_groups())
    print("Image size", image_size)
    print("Pixel size", round(pixel_size, 2))

    ###############################################
    # REGION OF INTEREST
    ###############################################

    solvent_res_mask = None
    solvent_mask = None
    if args.solvent_mask is not None:
        solvent_mask, roi_voxel_size, _ = load_mrc(args.solvent_mask)
        solvent_mask = torch.Tensor(solvent_mask.copy()).to(device)
        solvent_mask = 1 - torch.clip(solvent_mask, 0, 1)

        coords, _ = make_grid3d(size=image_size)
        grid3d_radius = torch.round(torch.sqrt(torch.sum(torch.square(coords.to(device)), -1)))
        if args.solvent_mask_res is not None and args.solvent_mask_res > 0:
            res_idx = round(image_size * pixel_size / args.solvent_mask_res)  # Convert resolution to spectral index
            solvent_res_mask = grid3d_radius < res_idx

    roi_skip_steps = 0

    do_roi = args.roi_mask is not None
    if do_roi:
        roi = RegionOfInterest(dac, args.roi_mask, args.roi_res, args.roi_latent_fraction, device)
    else:
        dac.auxiliaries["roi_basis_index"] = vaec.get_sb_latent_indices()
        dac.auxiliaries["roni_basis_index"] = []
        roi = None

    ###############################################
    # TENSORBOARD
    ###############################################

    roi_loss = None
    init_step_write_images = [50, 100, 200, 400, 800]

    summary = TensorboardSummary(log_dir, pixel_size, image_max_r + 1)

    ###############################################
    # PROFILING
    ###############################################

    prof = None
    if args.profile_runtime:
        prof = torch.profiler.profile(
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            activities=[torch.profiler.ProfilerActivity.CPU],
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(log_dir, "profiler")),
            # record_shapes=True,
            # profile_memory=True,
            # with_stack=True
        )
        prof.__enter__()

    ###############################################
    # RUN TRAINING
    ###############################################

    last_save_time = time.time()
    epoch = 1

    zero_scalar = torch.tensor(0., requires_grad=False).float()

    g_idx = Cache.get_spectral_indices(
        (image_size, image_size),
        max_r=image_size // 2,
        device=device
    )
    g_idx = dt_symmetrize(g_idx, dim=2)[..., image_size // 2:]
    mask = g_idx < image_max_r

    spectral_weighting_grid = None
    if args.spectral_weight_ll_res is not None:
        # Convert resolution to spectral index
        res_idx = round(image_size * pixel_size / args.spectral_weight_ll_res)
        spectral_weighting_grid = g_idx / float(res_idx)
        spectral_weighting_grid = torch.clip(torch.pow(10, -spectral_weighting_grid**2), 1e-2)

    mask_size = vaec.structure_decoder.get_spectral_weighted_output_2d_mask_size()
    kld_weight = args.latent_regularization * batch_size / (dataset_size * mask_size)

    encoder_param_count, decoder_param_count = vaec.count_parameters()
    print(f"encoder_param_count={encoder_param_count}")
    print(f"decoder_param_count={decoder_param_count}")

    try:
        while True:
            # hvc.zero_grad()
            if args.only_update_representation:
                raise StopIteration("No training! Only update representation requested...")
            dt_epoch = time.time()
            dt = time.time()
            for batch_ndx, sample in enumerate(data_loader):
                epoch = float(dac.train_step * batch_size) / dataset_size
                step = dac.train_step
                if step >= args.max_steps or epoch > args.max_epochs:
                    raise StopIteration("Maximum step/epoch count reached.")

                if step == 0:
                    summary.write_hidden_variable(hvc)  # Write initial values

                tt = time.time()
                roi_weight = 0
                if do_roi:
                    roi_weight = float(step % (roi_skip_steps + 1) == 0) * (roi_skip_steps + 1)

                vaec.zero_grad()
                this_batch_size = sample["idx"].shape[0]

                y, part_idx, hv, enc = run_batch_header(sample, dac, device, reparam=True)

                x_ft = vaec.structure_decoder_ddp(
                    sb_input=enc["sb"],
                    pp_input=enc["pp"],
                    max_r=image_max_r,
                    rot_matrices=hv["rot_matrices"].detach(),
                    sparse_grad=False,
                    do_postprocess=not args.dont_postprocess
                )

                int_shifts = torch.round(hv["shifts"].detach()).long().detach()
                y_ = integer_shift_2d(y, int_shifts)
                y_ = Cache.apply_circular_mask(y_, thickness=circular_mask_thickness, radius=circular_mask_radius)
                y_ft = dft(y_, dim=2)
                hvc.accumulate_data_stats(y_ft.detach(), hv["ctfs"].detach(), part_idx)

                y_ft = y_ft / (hv["amp_ctf"] + 1e-6)
                y_ft = dt_symmetrize(y_ft, dim=2)[:, :, image_size // 2:]
                y_ft = torch.view_as_real(y_ft)

                x_ft_shift = fourier_shift_2d(x_ft, hv["shifts"].detach() - int_shifts)
                ctfs = dt_symmetrize(hv["ctfs"], dim=2)[:, :, image_size // 2:].detach()
                x_ft_shift_ctf = x_ft_shift * ctfs[..., None]
                square_error = torch.sum(torch.square(
                    x_ft_shift_ctf -
                    y_ft.detach()
                ), -1)

                kld_loss = get_kld_loss(enc["mu"], enc["log_var"])

                mse_weight = dt_symmetrize(hv["amp_ctf"], dim=2)[:, :, image_size // 2:]
                mse_weight = torch.square(mse_weight)
                if spectral_weighting_grid is not None:
                    mse_weight *= spectral_weighting_grid
                mse_weight /= torch.mean(mse_weight) + 1e-6
                mse_weight *= torch.square(ctfs.detach())
                data_loss = torch.mean(
                    (mse_weight * square_error)[:, mask],
                    1  # Keep batch dim
                )

                train_loss = (
                         torch.mean(data_loss.nan_to_num(nan=2).clamp(0, 1)) +
                         torch.mean(kld_loss) * kld_weight
                )

                solvent_mask_loss = zero_scalar
                if solvent_mask is not None:
                    selected_idx = np.random.randint(this_batch_size)
                    sb, pp = vaec.basis_decoder(
                        *vaec.split_latent_space(enc["z"][selected_idx].detach().unsqueeze(0)))
                    data_spectra, data_ctf_spectra = dac.hidden_variable_container.get_data_stats(0)
                    v_ft = vaec.structure_decoder(
                        sb.detach(), pp.detach(), max_r=image_max_r, data_spectra=data_ctf_spectra)
                    v_ft = torch.view_as_complex(v_ft)
                    if solvent_res_mask is not None:
                        v_ft[0, solvent_res_mask] = v_ft[0, solvent_res_mask].detach()
                    v_ft = dt_desymmetrize(v_ft)[0]
                    vol = idft(v_ft, dim=3, real_in=True)
                    # masked_mean = torch.sum(vol * solvent_mask) / torch.sum(solvent_mask)
                    solvent_mask_loss = np.prod(vol.shape[-2:]) * vol.square().mul_(solvent_mask).sum().div_(solvent_mask.sum())
                    train_loss += solvent_mask_loss

                gp_loss = zero_scalar
                if args.gradient_penalty:
                    sb_input, _ = vaec.split_latent_space(enc["z"].detach())
                    sb_input_noise = sb_input + torch.randn_like(sb_input)
                    gp_loss = get_gradient_penalty(vaec.basis_decoder.sb, sb_input, sb_input_noise)
                    train_loss += gp_loss

                if step > 0:
                    train_loss.backward()

                vaec.step()

                train_timing = time.time() - tt
                total_timing = time.time() - dt

                summary.set_step(step)

                if prof is not None:
                    prof.step()

                if step % 10 == 0 and step > 0:
                    summary.summary.add_scalar(f"Extra/ Z norm", torch.norm(enc["z"]).detach().cpu().numpy(), step)
                    summary.summary.add_scalar(f"Extra/ MU norm", torch.norm(enc["mu"]).detach().cpu().numpy(), step)
                    summary.summary.add_scalar(f"Extra/ log_var norm", torch.norm(enc["log_var"]).detach().cpu().numpy(), step)
                    summary.summary.add_scalar(f"Extra/ SB norm", torch.norm(enc["sb"]).detach().cpu().numpy(), step)
                    summary.summary.add_scalar(f"Extra/ PP norm", torch.norm(enc["pp"]).detach().cpu().numpy(), step)
                    summary.summary.add_scalar(f"Loss/ Gradient penalty", gp_loss.detach().cpu().numpy(), step)
                    summary.summary.add_scalar(f"Loss/ solvent mask", solvent_mask_loss.detach().cpu().numpy(), step)
                    summary.write_stats(x_ft, y_ft, hv["amp"], hv["amp_ctf"])
                    summary.write_losses(
                        torch.mean(square_error),
                        kld_weight, kld_loss, data_loss,
                        train_loss, roi_weight, roi_loss,
                    )

                if step % args.image_steps == 0 and step > 0 or step in init_step_write_images:
                    summary.write_images(x_ft, y_ft, ctfs, roi)

                if timing_shown_count == 0:
                    print("Training has started...")
                    timing_shown_count += 1
                elif (step % 10001 == 0 or timing_shown_count < 10 and step % 10 == 0) and step > 0:
                    print("Step:", step, "Timing:", round(total_timing, 3), f"({round(train_timing, 3)})")
                    timing_shown_count += 1

                dac.train_step += 1

                dt = time.time()

            hvc.finalize_epoch()
            summary.write_hidden_variable(hvc)
            dac.save_to_checkpoint(log_dir, int(round(epoch)))

            dt_epoch = time.time() - dt_epoch
            print(f"Epoch number {int(round(epoch))} complete ({dt_epoch} s)")

            # exit(0)
    except StopIteration as e:
        print(e)
        print("Updating representation...")
        update_representation(dac, data_loader, device)
    except(KeyboardInterrupt, SystemExit):
        print("Exiting!")

    dac.save_to_checkpoint(log_dir)
    if prof is not None:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    DistributedProcessing.process_cleanup()


if __name__ == "__main__":
    args = parse_training_args()
    log_dir = args.log_dir

    # Remove log directory if overwriting
    if args.overwrite:
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)

    if not os.path.isdir(log_dir):
        print("Creating log-directory:", log_dir)
        os.mkdir(log_dir)

    sys.stdout = IOLogger(os.path.join(log_dir, 'std.out'))

    print(args)
    print(f"Running pytorch version {torch.__version__}")

    DistributedProcessing.global_setup(
        args=args, main_fn=main, verbose=True
    )
