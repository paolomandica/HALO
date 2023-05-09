from pathlib import Path
import setproctitle
import warnings
import torch
import os
import shutil
from core.utils.misc import mkdir, parse_args
from core.configs import cfg
from core.train_learners import SourceFreeLearner, SourceLearner, SourceTargetLearner, FullySupervisedLearner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl

warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True
# torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

protocol_types = {
    'source': SourceLearner,
    'source_free': SourceFreeLearner,
    'source_target': SourceTargetLearner,
    'fully_sup': FullySupervisedLearner
}


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str,  every: int):
        super().__init__()
        self.dirpath = dirpath
        self.every = every
        self.filename = "model_{global_step}"

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if (pl_module.global_step + 1) % self.every == 0:
            assert self.dirpath is not None
            # current = Path(self.dirpath) / f"model_step_{pl_module.global_step}.ckpt"
            self.filename = f"model_{pl_module.global_step}"
            filepath = Path(self.dirpath) + self.filename + '.ckpt'
            trainer.save_checkpoint(filepath)

class ActiveRoundCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str):
        super().__init__(save_top_k=-1)
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if (pl_module.global_step + 1) in pl_module.active_iters:
            assert self.dirpath is not None
            # current = Path(self.dirpath) / f"model_step_{pl_module.global_step}_round_{pl_module.active_round}.ckpt"
            self.filename = f"model_{pl_module.global_step}_round_{pl_module.active_round}"
            filepath = Path(self.dirpath) + self.filename + '.ckpt'
            trainer.save_checkpoint(filepath)


def main():
    args = parse_args()
    print(args, end='\n\n')

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    setproctitle.setproctitle(f'{args.proctitle}')

    # init wandb logger
    wandb_logger = None
    if cfg.WANDB.ENABLE and not cfg.DEBUG:
        wandb_logger = WandbLogger(project=cfg.WANDB.PROJECT, name=cfg.WANDB.NAME,
                                   entity=cfg.WANDB.ENTITY, group=cfg.WANDB.GROUP,
                                   config=cfg, save_dir='.')

    pl.seed_everything(cfg.SEED, workers=True)

    # init learner
    if cfg.PROTOCOL in protocol_types:
        print(f'\n\n>>>>>>>>>>>>>> PROTOCOL: {cfg.PROTOCOL} <<<<<<<<<<<<<<\n\n')
        learner = protocol_types[cfg.PROTOCOL](cfg)
    else:
        raise NotImplementedError(f'Protocol {cfg.PROTOCOL} is not implemented.')

    checkcall_1 = ModelCheckpoint(
        save_top_k=1,
        monitor="mIoU",
        mode="max",
        dirpath=cfg.OUTPUT_DIR,
        filename="model_{global_step}_{mIoU:.2f}",
    )

    callbacks = [checkcall_1]
    
    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.SOLVER.GPUS,
        max_epochs=1,
        max_steps=-1,
        log_every_n_steps=50,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        strategy="ddp",
        num_nodes=1,
        logger=wandb_logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        val_check_interval=500,
        precision=32,
        detect_anomaly=True)

    # start training

    if cfg.checkpoint:
        print(f"Resuming from checkpoint: {cfg.checkpoint}")
        trainer.fit(learner, ckpt_path=cfg.checkpoint)
    else:
        trainer.fit(learner)

if __name__ == '__main__':
    main()

    # remove gtIndicator subdirectory
    path = os.path.join(cfg.OUTPUT_DIR, 'gtIndicator')
    if os.path.exists(path):
        try:
            print("Removing gtIndicator directory...")
            shutil.rmtree(path)
        except:
            print("Failed to remove gtIndicator directory.")
