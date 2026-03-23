---
name: mixed-precision
description: Use FP16/BF16 mixed precision to accelerate training and reduce memory. Use when optimizing GPU performance.
metadata:
  category: tooling
  trigger-keywords: "training,gpu,memory,speed,precision,fp16,bf16"
  applicable-stages: "10,12"
  priority: "5"
  version: "1.0"
  author: researchclaw
  references: "Micikevicius et al., Mixed Precision Training, ICLR 2018"
  code-template: |
    scaler = torch.cuda.amp.GradScaler()
    for batch in dataloader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(batch)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
---

## Mixed Precision Training Best Practice
Use torch.cuda.amp for automatic mixed precision:
- Wrap forward pass in torch.cuda.amp.autocast()
- Use GradScaler for loss scaling
- BF16 preferred over FP16 on Ampere+ GPUs (RTX 3xxx, A100, RTX 4xxx)
- Watch for NaN gradients — reduce learning rate if needed
- Do NOT use amp with custom CUDA kernels unless tested
