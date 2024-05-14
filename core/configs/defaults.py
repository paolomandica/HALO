import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "deeplabv3plus_resnet101"
_C.MODEL.NUM_CLASSES = 19
_C.MODEL.WEIGHTS = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
_C.MODEL.FREEZE_BN = True
_C.MODEL.HYPER = True
_C.MODEL.CURVATURE = 1.0
_C.MODEL.REDUCED_CHANNELS = 64
_C.MODEL.HFR = True

_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.GROUP = "deeplabv2_r101_pretrain"
_C.WANDB.PROJECT = "active_domain_adapt"
_C.WANDB.ENTITY = "pinlab-sapienza"

_C.INPUT = CN()
_C.INPUT.SOURCE_INPUT_SIZE_TRAIN = (1280, 720)
_C.INPUT.TARGET_INPUT_SIZE_TRAIN = (1280, 640)
_C.INPUT.INPUT_SIZE_TEST = (1280, 640)
_C.INPUT.INPUT_SCALES_TRAIN = (1.0, 1.0)
_C.INPUT.IGNORE_LABEL = 255
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False

_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.SOURCE_TRAIN = ""
_C.DATASETS.TARGET_TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.GPUS = [0, 1, 2, 3]
_C.SOLVER.NUM_ITER = 60000

# optimizer and learning rate
_C.SOLVER.LR_METHOD = "poly"
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.LR_POWER = 0.5
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WARMUP_ITERS = 600

# 4 images per batch, two for source and two for target
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.BATCH_SIZE_VAL = 1

# hyper-parameters
_C.SOLVER.CONSISTENT_LOSS = 0.0
_C.SOLVER.NEGATIVE_LOSS = 1.0
_C.SOLVER.NEGATIVE_THRESHOLD = 0.05

# local consistent loss
_C.SOLVER.LCR_TYPE = "l1"


_C.ACTIVE = CN()
# active strategy
_C.ACTIVE.UNCERTAINTY = "entropy"
_C.ACTIVE.PURITY = "hyper"
_C.ACTIVE.SELECT_ITER = [0, 15000, 30000, 40000, 50000]
# total selection ratio per image
_C.ACTIVE.BUDGET = 0.05
# hyper-parameters
_C.ACTIVE.RADIUS_K = 1
_C.ACTIVE.NORMALIZE = True
_C.ACTIVE.MASK_RADIUS_K = 5
_C.ACTIVE.K = 100
# visualization of active selection
_C.ACTIVE.VIZ_MASK = False


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.VIZ_SCORE = False
_C.TEST.VIZ_WRONG = False
_C.TEST.SAVE_EMBED = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.NAME = "debug"
_C.OUTPUT_DIR = ""
_C.resume = ""
_C.SEED = -1
_C.DEBUG = False
_C.PROTOCOL = "source_target"
