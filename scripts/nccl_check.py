"""One NCCL all-reduce across the job's GPUs (bridges2/train_starvla.sbatch runs it under torchrun before training)."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=timedelta(seconds=90))
x = torch.ones(1024, device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
assert x[0].item() == dist.get_world_size()
if rank == 0:
    print(f"nccl check ok: world {dist.get_world_size()}, P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '0')}", flush=True)
dist.destroy_process_group()
