import torch


def get_lr_scheduler(
        optimiser,
        scheduler_type,
        warmup_steps=None,
        total_steps=None
):
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=total_steps - warmup_steps
        )
    elif scheduler_type == "linear_with_warmup":
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) /
                float(max(1, total_steps - warmup_steps))
            )
        return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    else:
        return None
