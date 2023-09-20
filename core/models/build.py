from .feature_extractor import resnet_feature_extractor
from .classifier import (
    ASPP_Classifier_V2,
    DepthwiseSeparableASPP,
    ASPP_Classifier_V2_Hyper,
    DepthwiseSeparableASPP_Hyper,
)
from core.models.segformer import SegformerDecodeHead
from .layers import FrozenBatchNorm2d
import torch.nn as nn
from transformers import SegformerModel


def build_feature_extractor(cfg):
    model_name, backbone_name = cfg.MODEL.NAME.split("_")
    if model_name == "segformer":
        pretrain_name = "nvidia/mit-" + backbone_name
        backbone = SegformerModel.from_pretrained(pretrain_name)
    elif backbone_name.startswith("resnet"):
        if cfg.MODEL.WEIGHTS != "none":
            pretrained_backbone_flag = True
            pretrained_weights = cfg.MODEL.WEIGHTS
        else:
            pretrained_backbone_flag = False
            pretrained_weights = None

        backbone = resnet_feature_extractor(
            backbone_name,
            pretrained_weights=pretrained_weights,
            aux=False,
            pretrained_backbone=pretrained_backbone_flag,
            freeze_bn=cfg.MODEL.FREEZE_BN,
        )
    else:
        raise NotImplementedError("Unsupported backbone: {}.".format(backbone_name))
    return backbone


def build_classifier(cfg, encoder_config=None):
    model_name, backbone_name = cfg.MODEL.NAME.split("_")
    bn_layer = nn.BatchNorm2d
    hyper = cfg.MODEL.HYPER
    if cfg.MODEL.FREEZE_BN:
        bn_layer = FrozenBatchNorm2d

    if model_name == "deeplabv2" and not hyper:
        classifier = ASPP_Classifier_V2(
            2048, [6, 12, 18, 24], [6, 12, 18, 24], cfg.MODEL.NUM_CLASSES
        )
    elif model_name == "deeplabv2" and hyper:
        classifier = ASPP_Classifier_V2_Hyper(
            2048,
            [6, 12, 18, 24],
            [6, 12, 18, 24],
            cfg.MODEL.NUM_CLASSES,
            reduced_channels=cfg.MODEL.REDUCED_CHANNELS,
        )
    elif model_name == "deeplabv3plus" and not hyper:
        classifier = DepthwiseSeparableASPP(
            inplanes=2048,
            dilation_series=[1, 6, 12, 18],
            padding_series=[1, 6, 12, 18],
            num_classes=cfg.MODEL.NUM_CLASSES,
            norm_layer=bn_layer,
            hfr=cfg.MODEL.HFR,
            reduced_channels=cfg.MODEL.REDUCED_CHANNELS,
        )
    elif model_name == "deeplabv3plus" and hyper:
        classifier = DepthwiseSeparableASPP_Hyper(
            inplanes=2048,
            dilation_series=[1, 6, 12, 18],
            padding_series=[1, 6, 12, 18],
            num_classes=cfg.MODEL.NUM_CLASSES,
            norm_layer=bn_layer,
            reduced_channels=cfg.MODEL.REDUCED_CHANNELS,
            hfr=cfg.MODEL.HFR,
        )
    elif model_name == "segformer":
        # output_channels = {
        #     "b0": 256,
        #     "b1": 512,
        #     "b2": 512,
        #     "b3": 512,
        #     "b4": 512,
        #     "b5": 512,
        # }
        assert encoder_config is not None

        # classifier = SegFormerSegmentationHead(
        #     channels=output_channels[backbone_name],
        #     num_classes=cfg.MODEL.NUM_CLASSES,
        #     hyper=cfg.MODEL.HYPER,
        #     hfr=cfg.MODEL.HFR,
        # )

        classifier = SegformerDecodeHead(
            config=encoder_config,
            num_classes=cfg.MODEL.NUM_CLASSES,
            hyper=cfg.MODEL.HYPER,
            hfr=cfg.MODEL.HFR,
        )

    else:
        raise NotImplementedError("Unsupported classifier: {}.".format(model_name))
    return classifier
