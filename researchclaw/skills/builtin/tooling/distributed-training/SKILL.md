---
name: distributed-training
description: Multi-GPU and distributed training patterns with PyTorch DDP. Use when scaling training across GPUs.
metadata:
  category: tooling
  trigger-keywords: "distributed,multi-gpu,parallel,ddp,scale"
  applicable-stages: "10,12"
  priority: "7"
  version: "1.0"
  author: researchclaw
  references: "PyTorch DDP Tutorial, pytorch.org; Goyal et al., Accurate Large Minibatch SGD, 2017"
---

## Distributed Training Best Practice
1. Use DistributedDataParallel (DDP) over DataParallel for multi-GPU
2. Initialize process group: dist.init_process_group(backend='nccl')
3. Use DistributedSampler for data sharding
4. Synchronize batch norm: nn.SyncBatchNorm.convert_sync_batchnorm()
5. Only save checkpoint on rank 0
6. Scale learning rate linearly with world size
7. Use gradient accumulation for effectively larger batch sizes
