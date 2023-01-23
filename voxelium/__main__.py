#!/usr/bin/env python

"""
Voxelium - Cryo-EM data analysis framework
"""


def main():
    import os
    import argparse
    import voxelium
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version=f'Voxelium {voxelium.__version__}')

    import voxelium.vae_volume.train

    modules = {
        "volume_train": voxelium.vae_volume.train,
    }

    subparsers = parser.add_subparsers(title='Choose a module')
    subparsers.required = 'True'

    for key in modules:
        module_parser = subparsers.add_parser(key, description=modules[key].__doc__)
        modules[key].append_args(module_parser)
        module_parser.set_defaults(func=modules[key].main)

    try:
        args = parser.parse_args()
        args.func(args)
    except TypeError:
        parser.print_help()


if __name__ == '__main__':
    main()