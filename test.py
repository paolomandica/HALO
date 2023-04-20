import setproctitle
import warnings
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.utils.misc import mkdir, parse_args
from core.configs import cfg
from core.train_learners import Test
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl

warnings.filterwarnings('ignore')


def main():
    args = parse_args()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    setproctitle.setproctitle(f'{args.proctitle}')

    pl.seed_everything(cfg.SEED, workers=True)

    learner = Test(cfg)

    # init trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.SOLVER.GPUS,
        precision=32,
        detect_anomaly=True)

    # start training
    trainer.test(learner)


if __name__ == '__main__':
    main()
