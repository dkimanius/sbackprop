# Sparse Fourier Backpropagation in Cryo-EM Reconstruction
This repository contains code for the paper: [Sparse Fourier Backpropagation in Cryo-EM Reconstruction](https://proceedings.neurips.cc/paper_files/paper/2022/hash/50729453d56ecf6a8b7be78998776472-Abstract-Conference.html).
Part of Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Main Conference Track.

# Setup a conda environment
You need to setup a Python environment with dependencies. We recommend installing via Miniconda3.

Once you have conda setup, you can install all the Python dependencies into a new environment by running:

```conda env create -f environment.yml```

You can then activate the conda environment by running:

```conda activate sbackprop```

# Compile and Install CUDA code
Once inside the correct environment you can compile and install the CUDA dependencies by running:

```python setup.py install```

# Running Training
You can then run training by running 

```python voxelium/vae_volume/train.py <input STAR-file> <logdir> --gpu 0```

Use ```-h``` for more options.

# Visualizing Results
You can then visualize the results using

```python voxelium/vae_volume/volume_explorer.py <logdir>```

# Citation
```
@article{kimanius2022sparse,
  title={Sparse fourier backpropagation in cryo-em reconstruction},
  author={Kimanius, Dari and Jamali, Kiarash and Scheres, Sjors},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={12395--12408},
  year={2022}
}
```