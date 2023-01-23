import argparse


def parse_training_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', help='input job (job directory or optimizer-file)', type=str)
    parser.add_argument('log_dir', type=str, metavar='log_dir', help='path to load a model')
    parser.add_argument('--particle_diameter', help='size of circular mask (ang)', type=int, default=None)
    parser.add_argument('--circular_mask_thickness', help='thickness of mask (ang)', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gpu', dest='gpu', type=str, default=None, help='gpu to use')
    parser.add_argument('--checkpoint_time', help='Minimum time in minutes between checkpoint saves', type=int,
                        default=10)
    parser.add_argument("--image_steps", type=int, default=500, help="Generate a tensorboard image every n steps")
    parser.add_argument('--max_steps', dest='max_steps', type=int, default=int(1e9), help='number of steps to train')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=int(1e9), help='number of epochs to train')
    parser.add_argument('--pytorch_threads', type=int, default=6)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--dataloader_threads', type=int, default=4)
    parser.add_argument(
        '--do_align',
        help='Do optimize pose and translation',
        action="store_true"
    )
    parser.add_argument(
        '--do_ctf_optimization',
        help='Do optimize CTF defocuse and angle',
        action="store_true"
    )
    parser.add_argument(
        '--roi_mask',
        help='An MRC-file containing values between 0 and 1. Ones (1) for region of interest (ROI).',
        type=str, default=None
    )
    parser.add_argument(
        '--roi_latent_fraction',
        help='Fraction of structural latent dimensions assigned to ROI.',
        type=float, default=.8
    )
    parser.add_argument(
        '--roi_res',
        help='Lowest resolution at which ROI is considered.',
        type=float, default=80.
    )
    parser.add_argument(
        '--solvent_mask',
        help='MRC file with ones in the region that is not solvent (region of interest)',
        type=str, default=None
    )
    parser.add_argument(
        '--solvent_mask_res',
        help='Lowest resolution at which solvent mask is considered.',
        type=float, default=None
    )
    parser.add_argument(
        '--spectral_weight_ll_res',
        help='Spectral weighting factor resolution for the log-likelihood',
        type=float, default=None
    )
    parser.add_argument(
        '--spectral_weight_grad_res',
        help='Spectral weighting factor resolution for the gradient',
        type=float, default=None
    )
    parser.add_argument('--profile_runtime', action='store_true')
    parser.add_argument(
        '--latent_regularization',
        help='Latent space global KL divergence regularization parameter',
        type=float, default=1
    )
    parser.add_argument(
        '--encoder_mask_resolution',
        help='The highest resolution frequency component shown to the encoder. Set to zero to use all image.',
        type=float, default=0.
    )
    parser.add_argument(
        '--encoder_embedding_size',
        help='Encoder image group index embedding dimensionality. Set to zero to disable.',
        type=int, default=8
    )

    parser.add_argument(
        '--sb_latent_size',
        help='Structure basis network input size.',
        type=int, default=16
    )
    parser.add_argument(
        '--pp_latent_size',
        help='Postprocessing network input size.',
        type=int, default=2
    )
    parser.add_argument(
        '--sb_basis_size',
        help='Structure basis network output size.',
        type=int, default=16
    )
    parser.add_argument(
        '--pp_basis_size',
        help='Postprocessing network input size.',
        type=int, default=2
    )

    parser.add_argument(
        '--basis_decoder_depth',
        help='Basis decoder network depth (nr layers).',
        type=int, default=0
    )
    parser.add_argument(
        '--basis_decoder_width',
        help='Basis decoder network width.',
        type=int, default=2
    )
    parser.add_argument(
        '--encoder_depth',
        help='Encoder network depth.',
        type=int, default=5
    )
    parser.add_argument(
        '--structure_decoder_lr',
        help='Learning rate of the structure decoder',
        type=float, default=0.0001
    )
    parser.add_argument(
        "--gradient-penalty",
        "--gradient_penalty",
        help="Use gradient penalty for the basis decoder",
        action="store_true"
    )
    parser.add_argument('--do_sigma_weighting', action='store_true')
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float32", 
        help="Data type used for storing images in data set"
    )
    parser.add_argument('--dont_postprocess', action='store_true')
    parser.add_argument('--only_update_representation', action='store_true')
    parser.add_argument(
        "--random-subset",
        "--random_subset", 
        type=int, 
        default=None, 
        help="Which Relion random subset to use. Defaults to all."
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache directory"
    )

    return parser.parse_args()
