import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.configs import cfg
from ..utils.hyperbolic import HyperMapper


def init_conv_layer(conv, size, in_channels=None):
    weight = torch.ones((size, size), dtype=torch.float32)
    weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
    if in_channels is not None:
        weight = weight.repeat([in_channels, 1, 1, 1])
    weight = nn.Parameter(weight)
    conv.weight = weight
    conv.requires_grad_(False)


def normalize_map(x):
    return (x - x.min().item()) / (x.max().item() - x.min().item())


class FloatingRegionScore(nn.Module):

    def __init__(self, in_channels=19, padding_mode='zeros', size=33, purity_type=None, K=100):
        """
        purity_conv: size*size
        entropy_conv: size*size
        """
        super(FloatingRegionScore, self).__init__()
        self.in_channels = in_channels
        assert size % 2 == 1, "error size"

        if purity_type is None:
            purity_type = cfg.ACTIVE.PURITY

        # init entropy_conv
        self.entropy_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size,
                                      stride=1, padding=int(size / 2), bias=False,
                                      padding_mode=padding_mode)
        init_conv_layer(self.entropy_conv, size)

        # init purity_conv
        if purity_type == 'hyper' or purity_type == 'ripu_quantized':
            self.K, size, in_channels = K, 3, K

        self.purity_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                                     stride=1, padding=int(size / 2), bias=False,
                                     padding_mode=padding_mode, groups=in_channels)
        init_conv_layer(self.purity_conv, size, in_channels=in_channels)

        self.mapper = HyperMapper(c=cfg.MODEL.CURVATURE)

    def compute_region_uncertainty(self, unc_type, logit, p, normalize=False, ground_truth=None):
        if unc_type == 'entropy':
            region_uncertainty = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(
                dim=0).unsqueeze(dim=0) / math.log(19)  # [1, 1, h, w]
        elif unc_type == 'oracle_acc':
            gt_clone = ground_truth.clone() #.detach().cpu()
            gt_clone[ground_truth == 255] = p.argmax(dim=0)[ground_truth == 255]
            # Select the 1 - p for the correct class in gtclone
            region_uncertainty = 1 - torch.gather(p, dim=0, index=gt_clone.unsqueeze(dim=0)).unsqueeze(dim=0)
        else:
            region_uncertainty = torch.zeros(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()

        if normalize:
            region_uncertainty = (region_uncertainty - region_uncertainty.min().item()) / (
                region_uncertainty.max().item() - region_uncertainty.min().item())

        if unc_type != 'none':
            region_uncertainty = self.entropy_conv(region_uncertainty.float())

        return region_uncertainty

    def quantize_uncert_map(self, decoder_out):
        EPS = 1e-5
        # decoder_out_norm = decoder_out.squeeze(0).norm(dim=0)
        decoder_out_norm = self.mapper.poincare_distance_origin(decoder_out, dim=1).squeeze(0)
        decoder_out_norm = (decoder_out_norm - decoder_out_norm.min().item()) / \
            (decoder_out_norm.max().item() - decoder_out_norm.min().item())
        decoder_out_norm = 1 - decoder_out_norm
        norm_min, norm_max = decoder_out_norm.min(), decoder_out_norm.max()
        decoder_out_norm = (decoder_out_norm - norm_min) / \
            (norm_max - norm_min)
        predict = (decoder_out_norm * self.K) - 0.5  # [h, w]
        # predict = (decoder_out.squeeze(0).norm(dim=0) * self.K) - 0.5  # [h, w]
        predict = torch.clamp(predict, min=-0.5+EPS, max=self.K-0.5-EPS)
        predict = torch.round(predict).long()
        return predict

    def quantize_prob_map(self, predict):
        EPS = 1e-5
        predict = torch.max(predict, dim=0)
        predict = predict[0]
        predict = predict * self.K - 0.5
        predict = torch.clamp(predict, min=-0.5+EPS, max=self.K-0.5-EPS)
        predict = torch.round(predict).long()
        return predict

    def compute_region_impurity(self, predict, K, normalize=False):
        one_hot = F.one_hot(predict, num_classes=K).float()
        one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)  # [1, 19, h, w]
        summary = self.purity_conv(one_hot)  # [1, 19, h, w]
        count = torch.sum(summary, dim=1, keepdim=True)  # [1, 1, h, w]
        dist = summary / count
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6),
                                    dim=1, keepdim=True) / math.log(K)
        if normalize:
            region_impurity = normalize_map(region_impurity)
        return region_impurity, count

    def forward(self, logit: torch.Tensor, decoder_out: torch.Tensor = None, unc_type: str = None,
                pur_type: str = None, normalize: bool = False, ground_truth = None):
        """
        Compute regions score, impurity and uncertainty.

        Args:
            logit: an n-dimensional Tensor
            decoder_out: an n-dimensional Tensor
            unc_type: type of uncertainty to compute (entropy, hyperbolic, certainty, none)
            pur_type: type of impurity to compute (ripu, none)
            normalize: normalize the impurity and uncertainty maps

        Return:
            score, purity, entropy
        """
        logit = logit.squeeze(dim=0)  # [19, h ,w]
        p = torch.softmax(logit, dim=0)  # [19, h, w]

        assert unc_type in ['entropy', 'none', 'oracle_acc'], "error: unc_type '{}' not implemented".format(unc_type)
        if self.entropy_conv.weight.device != logit.device:
            self.entropy_conv = self.entropy_conv.to(logit.device)
            self.purity_conv = self.purity_conv.to(logit.device)

        region_uncertainty = self.compute_region_uncertainty(unc_type, logit, p, ground_truth=ground_truth)

        if pur_type == 'ripu':
            predict = torch.argmax(p, dim=0)   # [h, w]
            region_impurity, count = self.compute_region_impurity(
                predict, self.in_channels, normalize)
        elif pur_type == 'ripu_quantized':
            predict = self.quantize_prob_map(p)
            region_impurity, count = self.compute_region_impurity(
                predict, self.K, normalize)
        elif pur_type == 'hyper':
            predict = self.quantize_uncert_map(decoder_out)
            region_impurity, count = self.compute_region_impurity(
                predict, self.K, normalize)
            radius = self.mapper.poincare_distance_origin(decoder_out, dim=1).unsqueeze(0)
            region_impurity = normalize_map(region_impurity)
            radius = normalize_map(radius)
            region_impurity = region_impurity * radius
        elif pur_type == 'none':
            region_impurity = torch.zeros(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()
            count = torch.ones(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()
        elif pur_type == 'radius':
            region_impurity = self.mapper.poincare_distance_origin(decoder_out, dim=1).unsqueeze(0)
            count = torch.ones(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()
        else:
            raise NotImplementedError("Error: purity type '{}' not implemented".format(pur_type))

        prediction_uncertainty = region_uncertainty / count  # [1, 1, h, w]

        if normalize:
            prediction_uncertainty = normalize_map(prediction_uncertainty)
            region_impurity = normalize_map(region_impurity)

        score = region_impurity * prediction_uncertainty

        # squeeze the batch dimension
        score = score.squeeze(dim=0).squeeze(dim=0)
        region_impurity = region_impurity.squeeze(dim=0).squeeze(dim=0)
        prediction_uncertainty = prediction_uncertainty.squeeze(
            dim=0).squeeze(dim=0)

        return score, region_impurity, prediction_uncertainty
