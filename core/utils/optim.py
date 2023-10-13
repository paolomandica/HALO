from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, PolynomialLR, ConstantLR
from geoopt.optim import RiemannianSGD, RiemannianAdam

optimizers = {
    "hyper": {"sgd": RiemannianSGD, "adam": AdamW},  # RiemannianAdam,
    "euclidean": {
        "sgd": SGD,
        "adam": AdamW,
    },
}


def build_optimizer(model, hyper, optim_type, lr, weight_decay, momentum):
    hyper_key = "hyper" if hyper else "euclidean"

    assert optim_type in optimizers[hyper_key]
    optim = optimizers[hyper_key][optim_type]

    if optim_type == "sgd":
        optimizer = optim(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optim_type == "adam":
        optimizer = optim(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    return optimizer


def build_scheduler(optimizer, scheduler_type, num_iters, warmup_iters, lr_power):
    # use polynomial scheduler
    if scheduler_type == "poly":
        scheduler = PolynomialLR(optimizer, total_iters=num_iters, power=lr_power)
    # keep the learning rate constant
    elif scheduler_type == "constant":
        scheduler = None

    if warmup_iters > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_iters
        )
        if scheduler is None:
            scheduler = warmup_scheduler
        else:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_iters],
            )
    return scheduler
