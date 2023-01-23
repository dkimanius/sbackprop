#!/usr/bin/env python3

"""
Test module for a training VAE
"""
import os
import sys
from typing import List, TypeVar, Any, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


Tensor = TypeVar('torch.tensor')


class DistributedProcessing:
    _doing_ddp = False
    _this_rank = None
    _this_device = torch.device('cpu')
    _this_device_id = None

    @staticmethod
    def global_setup(args, main_fn, verbose=True) -> None:
        rank_gpu_map = []

        if args.gpu is not None:
            queried_gpu_ids = args.gpu.split(",")
            for i in range(len(queried_gpu_ids)):
                gpu_id = int(queried_gpu_ids[i].strip())
                try:
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                except AssertionError:
                    if verbose:
                        print(f'WARNING: GPU with the device id "{gpu_id}" not found.', file=sys.stderr)
                    continue
                if verbose:
                    print(f'Found device "{gpu_name}"')
                rank_gpu_map.append(gpu_id)

            if len(rank_gpu_map) > 0:
                if verbose:
                    print("Running on GPU with device id(s)", *rank_gpu_map)
            else:
                if verbose:
                    print(f'WARNING: no GPUs were found with the specified ids.', file=sys.stderr)

        if len(rank_gpu_map) == 0:
            if verbose:
                print("Running on CPU")

        world_size = max(len(rank_gpu_map), 1)
        ddp_args = {'world_size': world_size, 'rank_gpu_map': rank_gpu_map}

        if world_size > 1:
            mp.spawn(main_fn, args=(args, ddp_args), nprocs=world_size, join=True)
        else:
            main_fn(rank=0, args=args, ddp_args=ddp_args)

    @staticmethod
    def process_setup(rank, args) -> None:

        if len(args['rank_gpu_map']) > 0:
            device_id = args['rank_gpu_map'][rank]
            DistributedProcessing._this_device_id = device_id
            DistributedProcessing._this_device = torch.device('cuda:' + str(device_id))
        else:
            DistributedProcessing._this_device = torch.device('cpu')

        if args['world_size'] > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=args['world_size'])
            DistributedProcessing._doing_ddp = True
            DistributedProcessing._this_rank = rank

            print(f"Process {rank} initialized")

            dist.barrier()

    @staticmethod
    def process_cleanup() -> None:
        if DistributedProcessing._doing_ddp:
            dist.destroy_process_group()

    @staticmethod
    def doing_ddp() -> bool:
        return DistributedProcessing._doing_ddp

    @staticmethod
    def is_rank_zero() -> bool:
        return DistributedProcessing._this_rank == 0

    @staticmethod
    def get_rank() -> int:
        return DistributedProcessing._this_rank

    @staticmethod
    def get_device() -> Any:
        return DistributedProcessing._this_device

    @staticmethod
    def get_device_id() -> int:
        return DistributedProcessing._this_device_id

    @staticmethod
    def setup_module(module: torch.nn.Module) -> Union[torch.nn.Module, DDP]:
        module = module.to(DistributedProcessing.get_device())
        if DistributedProcessing.doing_ddp():
            module = DDP(module, device_ids=[DistributedProcessing.get_device_id()])
        return module
