import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""

        # 确保配置对象包含所有必要参数
    total_epochs = args.training.n_epochs  # 总训练轮次
    warmup_epochs = args.optim.warmup_epochs
    base_lr = args.optim.lr
    min_lr = args.optim.min_lr

    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs 
    else:
        if args.optim.lr_schedule == "constant":
            lr = base_lr
        elif args.optim.lr_schedule == "cosine":
        # 安全计算分母
            decay_period = total_epochs - warmup_epochs
            if decay_period <= 0:  # 保护除零错误
                decay_ratio = 0
            else:
                decay_ratio = (epoch - warmup_epochs) / decay_period
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))


            # lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            #     (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / args.decay_epochs))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
