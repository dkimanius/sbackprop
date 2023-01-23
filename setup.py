#!/usr/bin/env python

"""
Setup module for Voxelium
"""

import os
import sys
import sysconfig

from setuptools import setup, find_packages
from torch.utils import cpp_extension


def print_debug_msg():
    print("-------------------------------------- ")
    print("------------- DEBUG MODE ------------- ")
    print("-------------------------------------- ")


sys.path.insert(0, f'{os.path.dirname(__file__)}/voxelium')
import voxelium

_DEBUG = False
_DEBUG_LEVEL = 0

project_root = os.path.join(os.path.realpath(os.path.dirname(__file__)),  "voxelium")

include_dirs = [project_root]

cxx_extra_compile_args = []
nvcc_extra_compile_args = []
if _DEBUG:
    print_debug_msg()
    cxx_extra_compile_args += ["-g", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
    nvcc_extra_compile_args += ["-G", "-lineinfo"]
else:
    cxx_extra_compile_args += ["-DNDEBUG", "-O3"]
nvcc_extra_compile_args += cxx_extra_compile_args

ext_modules = [
    cpp_extension.CUDAExtension(
        name='voxelium_svr_linear',
        sources=[
            'voxelium/vae_volume/svr_linear/pybind.cpp',
            'voxelium/vae_volume/svr_linear/trilinear_projection.cpp',
            'voxelium/vae_volume/svr_linear/trilinear_projection_cpu_kernels.cpp',
            'voxelium/vae_volume/svr_linear/trilinear_projection_cuda_kernels.cu',
            'voxelium/vae_volume/svr_linear/volume_extraction.cpp',
            'voxelium/vae_volume/svr_linear/volume_extraction_cpu_kernels.cpp',
            'voxelium/vae_volume/svr_linear/volume_extraction_cuda_kernels.cu',
        ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': cxx_extra_compile_args, 'nvcc': nvcc_extra_compile_args},
    )
]
setup(
    name='voxelium',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "voxelium = voxelium.__main__:main",
        ],
    },
    version=voxelium.__version__
)

if _DEBUG:
    print_debug_msg()
