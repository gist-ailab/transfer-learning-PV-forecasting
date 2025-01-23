import os
import torch
import numpy as np
import wandb
import torch.distributed as dist
import torch.nn as nn


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self._init_distributed_mode(args)  # Initialize distributed training
        self.device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
        model = self._build_model()
        model = model.to(self.device)
        if args.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )
        self.model = model

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _set_wandb(self, setting):
        if self.args.run_name != None:
            run_name = self.args.run_name
        elif self.args.run_name == None:
            run_name = setting
            
        wandb.init(
            project='pv-forecasting',
            entity='pv-forecasting',
            name=f'{run_name}',
            config=self.args,
            id=self.args.wandb_id,
            resume=self.args.wandb_resume,)
        
    def _init_distributed_mode(self, args):
        # Initialize distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            print(f"Distributed training initialized: rank {args.rank}, world_size {args.world_size}, local_rank {args.local_rank}")
        else:
            print('Not using distributed mode')
            args.distributed = False
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
            return

        args.distributed = True

        # 명시적으로 GPU 디바이스 설정
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'

        dist.init_process_group(
            backend=args.dist_backend,
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank,
            )
        dist.barrier(device_ids=[args.local_rank])
        # torch.distributed.barrier(device_ids=[args.local_rank])
        self._setup_for_distributed(args.rank == 0)

    def _setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins
        builtin_print = builtins.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)

        builtins.print = print

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

# %%
