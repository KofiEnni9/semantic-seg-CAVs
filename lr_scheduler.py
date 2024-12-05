from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts

def PolyLRScheduler(optimizer, num_epochs, warmup_epochs, warmup_factor, min_lr, train_loader):
    warmup_iters = len(train_loader) * warmup_epochs
    main_iters = len(train_loader) * (num_epochs - warmup_epochs)

    # Warm-up scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=warmup_factor, 
        end_factor=1.0, 
        total_iters=warmup_iters
    )

    # Main scheduler (Cosine Annealing)
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_max=main_iters,
        eta_min=min_lr
    )

    # Combine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_iters]
    )

    return scheduler

