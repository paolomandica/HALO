import math
from typing import Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.configs import cfg


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
        else:
            purity_type = purity_type

        if purity_type == 'hyper':
            self.K = K
            self.purity_conv = nn.Conv2d(in_channels=self.K, out_channels=self.K, kernel_size=3,
                                         stride=1, padding=int(3 / 2), bias=False,
                                         padding_mode='zeros', groups=self.K)
            weight = torch.ones((3, 3), dtype=torch.float32)
            weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
            weight = weight.repeat([self.K, 1, 1, 1])
            weight = nn.Parameter(weight)
            self.purity_conv.weight = weight
            self.purity_conv.requires_grad_(False)
        else:
            self.purity_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=size,
                                         stride=1, padding=int(size / 2), bias=False,
                                         padding_mode=padding_mode, groups=in_channels)
            weight = torch.ones((size, size), dtype=torch.float32)
            weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
            weight = weight.repeat([in_channels, 1, 1, 1])
            weight = nn.Parameter(weight)
            self.purity_conv.weight = weight
            self.purity_conv.requires_grad_(False)

        self.entropy_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=size,
                                      stride=1, padding=int(size / 2), bias=False,
                                      padding_mode=padding_mode)
        weight = torch.ones((size, size), dtype=torch.float32)
        weight = weight.unsqueeze(dim=0).unsqueeze(dim=0)
        weight = nn.Parameter(weight)
        self.entropy_conv.weight = weight
        self.entropy_conv.requires_grad_(False)

    def compute_region_uncertainty(self, unc_type, logit, p, decoder_out, normalize=False):
        if unc_type == 'entropy':
            region_uncertainty = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(
                dim=0).unsqueeze(dim=0) / math.log(19)  # [1, 1, h, w]
        elif unc_type == 'ent_cert':
            region_uncertainty = 1 - torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(
                dim=0).unsqueeze(dim=0) / math.log(19)  # [1, 1, h, w]
        elif unc_type == 'hyperbolic':
            region_uncertainty = 1 - \
                decoder_out.norm(dim=1, p=2).unsqueeze(dim=1)
        elif unc_type == 'certainty':
            region_uncertainty = decoder_out.norm(dim=1, p=2).unsqueeze(dim=1)
        elif unc_type == 'none':
            region_uncertainty = torch.zeros(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()
        else:
            raise NotImplementedError(
                "unc_type '{}' not implemented".format(unc_type))
        if normalize:
            region_uncertainty = (region_uncertainty - region_uncertainty.min().item()) / (
                region_uncertainty.max().item() - region_uncertainty.min().item())
        return region_uncertainty

    def quantize_uncert_map(self, decoder_out, type='kmeans', cluster_centers=None):
        assert type in ['uniform',
                        'kmeans'], "type '{}' not implemented".format(type)

        if type == 'uniform':
            EPS = 1e-5
            decoder_out_norm = decoder_out.squeeze(0).norm(dim=0)
            norm_min, norm_max = decoder_out_norm.min(), decoder_out_norm.max()
            decoder_out_norm = (decoder_out_norm - norm_min) / (norm_max - norm_min)
            predict = (decoder_out_norm * self.K) - 0.5  # [h, w]

            # predict = (decoder_out.squeeze(0).norm(dim=0) * self.K) - 0.5  # [h, w]

            predict = torch.clamp(predict, min=-0.5+EPS, max=self.K-0.5-EPS)
            predict = torch.round(predict).long()
            return predict
        elif type == 'kmeans':
            if cluster_centers is None:
                kmeans_dict = json.load(open('kmeans/kmeans_dict.json', 'r'))
                checkpoint = kmeans_dict.values()[-1]
                cluster_centers = torch.Tensor(kmeans_dict[checkpoint])

            predict = decoder_out.squeeze(0).norm(dim=0)
            predict_flatten = predict.reshape(-1, 1)
            cluster_ids_sorted = torch.cdist(predict_flatten.reshape(
                1, -1, 1).float(), cluster_centers.reshape(1, -1, 1).float().to(predict_flatten.device))
            _, indices = cluster_ids_sorted.squeeze(0).min(dim=-1)
            return indices.reshape(decoder_out.shape[-2:]).long()

    def compute_region_impurity(self, predict, K, normalize=False):
        one_hot = F.one_hot(predict, num_classes=K).float()
        one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)  # [1, 19, h, w]
        summary = self.purity_conv(one_hot)  # [1, 19, h, w]
        count = torch.sum(summary, dim=1, keepdim=True)  # [1, 1, h, w]

        dist = summary / count
        region_impurity = torch.sum(-dist * torch.log(dist + 1e-6),
                                    dim=1, keepdim=True) / math.log(K)

        if normalize:
            region_impurity = (region_impurity - region_impurity.min().item()) / \
                (region_impurity.max().item() - region_impurity.min().item())

        return region_impurity, count

    def forward(self, logit: torch.Tensor, decoder_out: torch.Tensor = None, unc_type: str = None, pur_type: str = None, normalize: bool = False, alpha: float = None, quant_type=None, cluster_centers=None):
        """
        Compute regions score, impurity and uncertainty.

        Args:
            logit: an n-dimensional Tensor
            decoder_out: an n-dimensional Tensor
            unc_type: type of uncertainty to compute (entropy, hyperbolic, certainty, none)
            pur_type: type of impurity to compute (ripu, none)
            normalize: normalize the impurity and uncertainty
            alpha: weight of hyperbolic uncertainty mixed with entropy uncertainty

        Return:
            score, purity, entropy
        """
        logit = logit.squeeze(dim=0)  # [19, h ,w]
        p = torch.softmax(logit, dim=0)  # [19, h, w]

        if quant_type == None:
            quant_type = cfg.ACTIVE.QUANT

        assert unc_type in ['entropy', 'hyperbolic', 'certainty', 'ent_cert',
                            'none'], "error: unc_type '{}' not implemented".format(unc_type)
        if self.entropy_conv.weight.device != logit.device:
            self.entropy_conv = self.entropy_conv.to(logit.device)
            self.purity_conv = self.purity_conv.to(logit.device)

        if alpha is not None:
            region_uncertainty_entropy = self.compute_region_uncertainty(
                'entropy', logit, p, decoder_out)
            region_uncertainty_hyper = self.compute_region_uncertainty(
                'hyperbolic', logit, p, decoder_out)
            region_uncertainty = (
                1-alpha) * region_uncertainty_entropy + alpha * region_uncertainty_hyper
        else:
            region_uncertainty = self.compute_region_uncertainty(
                unc_type, logit, p, decoder_out)

        if unc_type == 'none':
            region_sum_uncert = region_uncertainty
        else:
            region_sum_uncert = self.entropy_conv(
                region_uncertainty.float())  # [1, 1, h, w]

        # one_hot = F.one_hot(predict, num_classes=self.in_channels).float()
        # one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)  # [1, 19, h, w]
        # summary = self.purity_conv(one_hot)  # [1, 19, h, w]
        # count = torch.sum(summary, dim=1, keepdim=True)  # [1, 1, h, w]

        assert pur_type in [
            'ripu', 'hyper', 'none'], "error: pur_type '{}' not implemented".format(pur_type)
        if pur_type == 'ripu':
            predict = torch.argmax(p, dim=0)   # [h, w]
            region_impurity, count = self.compute_region_impurity(
                predict, self.in_channels, normalize)
            # dist = summary / count
            # region_impurity = torch.sum(-dist * torch.log(dist + 1e-6), dim=1, keepdim=True) / math.log(19)
        elif pur_type == 'hyper':
            # EPS = 1e-5
            # predict = (decoder_out.squeeze(0).norm(dim=0) * self.K) - 0.5  # [h, w]
            # predict = torch.clamp(predict, min=-0.5+EPS, max=self.K-0.5-EPS)
            # predict = torch.round(predict).long()
            predict = self.quantize_uncert_map(
                decoder_out, type=quant_type, cluster_centers=cluster_centers)

            region_impurity, count = self.compute_region_impurity(
                predict, self.K, normalize)
            
        elif pur_type == 'none':
            region_impurity = torch.zeros(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()
            count = torch.ones(
                (1, 1, logit.shape[1], logit.shape[2]), dtype=torch.float32).cuda()

        prediction_uncertainty = region_sum_uncert / count  # [1, 1, h, w]

        if normalize:
            prediction_uncertainty = (prediction_uncertainty - prediction_uncertainty.min().item()) / (
                prediction_uncertainty.max().item() - prediction_uncertainty.min().item())
            region_impurity = (region_impurity - region_impurity.min().item()) / \
                (region_impurity.max().item() - region_impurity.min().item())

        score = region_impurity * prediction_uncertainty
        return score.squeeze(dim=0).squeeze(dim=0), region_impurity.squeeze(dim=0).squeeze(
            dim=0), prediction_uncertainty.squeeze(dim=0).squeeze(dim=0)
