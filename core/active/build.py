import math
import torch
import random

import numpy as np
import torch.nn.functional as F

from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore
from .spatial_purity import SpatialPurity


import os
from core.configs import cfg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from core.utils.misc import get_color_pallete

CITYSCAPES_MEAN = torch.Tensor([123.675, 116.28, 103.53]).reshape(1, 1, 3).numpy()
CITYSCAPES_STD = torch.Tensor([58.395, 57.12, 57.375]).reshape(1, 1, 3).numpy()

np.random.seed(cfg.SEED+1)
VIZ_LIST = list(np.random.randint(0, 500, 20))

def PixelSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            if not cfg.MODEL.HYPER:
                tgt_out = classifier(tgt_feat, size=tgt_size)
            else:
                tgt_out, decoder_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)

                if cfg.ACTIVE.UNCERTAINTY == 'entropy':
                    uncertainty = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                elif cfg.ACTIVE.UNCERTAINTY == 'hyperbolic':
                    uncertainty = decoder_out[i:i + 1, :, :, :]
                    uncertainty = F.interpolate(uncertainty, size=size, mode='bilinear', align_corners=True)
                    uncertainty = uncertainty.squeeze(0).norm(dim=0, p=2)
                else:
                    uncertainty = torch.ones_like(p[0, :, :])

                if cfg.ACTIVE.PURITY == 'ripu':
                    pseudo_label = torch.argmax(p, dim=0)
                    one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                    one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                    purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                else:
                    purity = torch.ones_like(p[0, :, :])

                score = uncertainty * purity

                score[active] = -float('inf')

                for pixel in range(active_pixels):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()


def RegionSelection(cfg, feature_extractor, classifier, tgt_epoch_loader, round_number):

    feature_extractor.eval()
    classifier.eval()

    floating_region_score = FloatingRegionScore(
        in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1)
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)
    uncertainty_type = cfg.ACTIVE.UNCERTAINTY
    purity_type = cfg.ACTIVE.PURITY
    alpha = cfg.ACTIVE.ALPHA

    with torch.no_grad():
        idx = 0
        for tgt_data in tqdm(tgt_epoch_loader):
            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            if not cfg.MODEL.HYPER:
                tgt_out = classifier(tgt_feat, size=tgt_size)
                decoder_out = None
            else:
                tgt_out, decoder_out = classifier(tgt_feat, size=tgt_size)

            # just a single iteration, because len(origin_mask)=1
            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                if cfg.ACTIVE.RATIO == 1:
                    active_mask = ground_truth
                    active = torch.ones_like(active_mask, dtype=torch.bool)
                    selected = torch.ones_like(active_mask, dtype=torch.bool)
                elif uncertainty_type == 'certuncert':
                    output = tgt_out[i:i + 1, :, :, :]
                    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

                    decoder_out = decoder_out[i:i + 1, :, :, :]
                    decoder_out = F.interpolate(decoder_out, size=size, mode='bilinear', align_corners=True)

                    score_unc, purity, uncertainty = floating_region_score(
                        output, decoder_out=decoder_out, normalize=cfg.ACTIVE.NORMALIZE, unc_type='entropy', pur_type='ripu', alpha=alpha)
                    
                    score_cert, purity, uncertainty = floating_region_score(
                        output, decoder_out=decoder_out, normalize=cfg.ACTIVE.NORMALIZE, unc_type='certainty', pur_type='ripu', alpha=alpha)

                    score_unc[active] = -float('inf')

                    weight_uncert = cfg.ACTIVE.WEIGHT_UNCERT[round_number-1]
                    active_regions = math.ceil(weight_uncert * ((num_pixel_cur * active_ratio) / per_region_pixels))

                    for pixel in range(active_regions):
                        values, indices_h = torch.max(score_unc, dim=0)
                        _, indices_w = torch.max(values, dim=0)
                        w = indices_w.item()
                        h = indices_h[w].item()

                        active_start_w = w - active_radius if w - active_radius >= 0 else 0
                        active_start_h = h - active_radius if h - active_radius >= 0 else 0
                        active_end_w = w + active_radius + 1
                        active_end_h = h + active_radius + 1

                        mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                        mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                        mask_end_w = w + mask_radius + 1
                        mask_end_h = h + mask_radius + 1

                        # mask out
                        score_unc[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                        active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                        selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                        # active sampling
                        active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                            ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]
                        
                    score_cert[active] = -float('inf')
                    active_regions = math.ceil((1-weight_uncert) * ((num_pixel_cur * active_ratio) / per_region_pixels))

                    for pixel in range(active_regions):
                        values, indices_h = torch.max(score_cert, dim=0)
                        _, indices_w = torch.max(values, dim=0)
                        w = indices_w.item()
                        h = indices_h[w].item()

                        active_start_w = w - active_radius if w - active_radius >= 0 else 0
                        active_start_h = h - active_radius if h - active_radius >= 0 else 0
                        active_end_w = w + active_radius + 1
                        active_end_h = h + active_radius + 1

                        mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                        mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                        mask_end_w = w + mask_radius + 1
                        mask_end_h = h + mask_radius + 1

                        # mask out
                        score_cert[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                        active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                        selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                        # active sampling
                        active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                            ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]
                    

                else:
                    output = tgt_out[i:i + 1, :, :, :]
                    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)

                    if uncertainty_type in ['certainty', 'hyperbolic'] or (uncertainty_type == 'none' and cfg.MODEL.HYPER) or (cfg.ACTIVE.ALPHA and cfg.MODEL.HYPER):
                        decoder_out = decoder_out[i:i + 1, :, :, :]
                        decoder_out = F.interpolate(decoder_out, size=size, mode='bilinear', align_corners=True)

                    score, purity, uncertainty = floating_region_score(
                        output, decoder_out=decoder_out, normalize=cfg.ACTIVE.NORMALIZE, unc_type=uncertainty_type, pur_type=purity_type, alpha=alpha)

                    score[active] = -float('inf')

                    active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)

                    for pixel in range(active_regions):
                        values, indices_h = torch.max(score, dim=0)
                        _, indices_w = torch.max(values, dim=0)
                        w = indices_w.item()
                        h = indices_h[w].item()

                        active_start_w = w - active_radius if w - active_radius >= 0 else 0
                        active_start_h = h - active_radius if h - active_radius >= 0 else 0
                        active_end_w = w + active_radius + 1
                        active_end_h = h + active_radius + 1

                        mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                        mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                        mask_end_w = w + mask_radius + 1
                        mask_end_h = h + mask_radius + 1

                        # mask out
                        score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                        active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                        selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                        # active sampling
                        active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                            ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                active_mask_np = np.array(active_mask.cpu().numpy(), dtype=np.uint8)
                active_mask_IMG = Image.fromarray(active_mask_np)
                active_mask_IMG.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

            if cfg.ACTIVE.VIZ_MASK and idx in VIZ_LIST:
                img_np = F.interpolate(tgt_input, size=size, mode='bilinear',
                                       align_corners=True).cpu().numpy()[0].transpose(1, 2, 0)
                img_np = (img_np * CITYSCAPES_STD + CITYSCAPES_MEAN).astype(np.uint8)
                # active_mask_np = F.interpolate(torch.Tensor(active_mask_np).unsqueeze(0).unsqueeze(0), size=torch.Size(img_np.shape[:2]), mode='bilinear', align_corners=True)
                # active_mask_np = active_mask_np.cpu().numpy().squeeze(0).squeeze(0)
                score_np = score.cpu().numpy()
                name = tgt_data['name'][0]
                visualization_plots(img_np, score_np, active_mask_np, round_number, name)
            idx += 1

    feature_extractor.train()
    classifier.train()


def visualization_plots(img_np, score_np, active_mask_np, round_number, name, cmap1='gray', cmap2='viridis', alpha=0.7):

    fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 10))

    # plot original image
    # axes[0].set_title('Original Image')
    axes[0].imshow(img_np)
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)

    if cfg.ACTIVE.UNCERTAINTY == 'entropy':
        title = 'Entropy + '
    elif cfg.ACTIVE.UNCERTAINTY == 'hyperbolic':
        title = 'Hyperbolic Uncertainty + '
    elif cfg.ACTIVE.UNCERTAINTY == 'certainty':
        title = 'Hyperbolic Certainty + '
    else:
        title = ''

    if cfg.ACTIVE.PURITY == 'ripu':
        title += 'Impurity'

    axes[1].set_title('Total Score: '+title)
    axes[1].imshow(img_np, cmap=cmap1)
    im_score = axes[1].imshow(score_np,  cmap=cmap2, alpha=alpha)
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    # plot original image
    axes[2].set_title('Selected Pixel - Active Round: '+str(round_number))
    axes[2].imshow(img_np, cmap=cmap1)
    axes[2].imshow(active_mask_np, cmap='autumn', alpha=alpha)
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)

    # make directory if it doesn't exist
    if not os.path.exists(cfg.OUTPUT_DIR + '/viz'):
        os.makedirs(cfg.OUTPUT_DIR + '/viz')
    name = name.rsplit('/', 1)[-1].rsplit('_', 1)[0]
    file_name = cfg.OUTPUT_DIR + '/viz/' + name + '_round'+str(round_number)+'.png'

    plt.suptitle(name)
    plt.savefig(file_name)
    plt.close()
