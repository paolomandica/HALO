import math
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from core.active.floating_region import FloatingRegionScore

import os
from core.configs import cfg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from core.utils.misc import get_color_pallete

CITYSCAPES_MEAN = torch.Tensor([123.675, 116.28, 103.53]).reshape(1, 1, 3).numpy()
CITYSCAPES_STD = torch.Tensor([58.395, 57.12, 57.375]).reshape(1, 1, 3).numpy()

np.random.seed(cfg.SEED+1)
VIZ_LIST = list(np.random.randint(0, 500, 20))


def visualize_wrong(image, output, decoder_out, gt_segm_map, name, cfg, cmap1='gray', cmap2='viridis', alpha=0.7):

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()

    if cfg.MODEL.HYPER:
        score, impurity, entropy = floating_region_score(output, decoder_out=decoder_out, unc_type='entropy', pur_type='ripu', normalize=True)
        _, _, hypunc = floating_region_score(output, decoder_out=decoder_out, unc_type='hyperbolic', pur_type='ripu', normalize=True)
        _, _, hypcert = floating_region_score(output, decoder_out=decoder_out, unc_type='certainty', pur_type='ripu', normalize=True)
    else:
        score, impurity, entropy = floating_region_score(output, unc_type='entropy', pur_type='ripu', normalize=True)

    img_np = F.interpolate(image.unsqueeze(0), size=decoder_out.shape[-2:], mode='nearest')[0]   # 640, 1280 -->  160, 320
    img_np = img_np.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * CITYSCAPES_STD + CITYSCAPES_MEAN).astype(np.uint8)

    score_np = score.cpu().numpy()
    entropy_np = entropy.cpu().numpy()
    hypunc_np = hypunc.cpu().numpy()
    hypcert_np = hypcert.cpu().numpy()
    impurity_np = impurity.cpu().numpy()


    fig, axes = plt.subplots(4, 3, figsize=(10, 10)) # constrained_layout = True

    axes[0,0].set_title('Hyper Uncertainty')
    axes[0,0].imshow(img_np, cmap=cmap1)
    im_score = axes[0,0].imshow(hypunc_np,  cmap=cmap2, alpha=alpha)
    axes[0,0].xaxis.set_visible(False)
    axes[0,0].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("left", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='left')


    axes[1,0].set_title('Entropy')
    axes[1,0].imshow(img_np, cmap=cmap1)
    im_score = axes[1,0].imshow(entropy_np,  cmap=cmap2, alpha=alpha)
    axes[1,0].xaxis.set_visible(False)
    axes[1,0].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes("left", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='left')

    axes[2,0].set_title('Impurity')
    axes[2,0].imshow(img_np, cmap=cmap1)
    im_score = axes[2,0].imshow(impurity_np,  cmap=cmap2, alpha=alpha)
    axes[2,0].xaxis.set_visible(False)
    axes[2,0].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes("left", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='left')

    axes[3,0].set_title('Hyper Certainty')
    axes[3,0].imshow(img_np, cmap=cmap1)
    im_score = axes[3,0].imshow(hypcert_np,  cmap=cmap2, alpha=alpha)
    axes[3,0].xaxis.set_visible(False)
    axes[3,0].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[3,0])
    cax = divider.append_axes("left", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='left')



    output_large = F.interpolate(output, size=image.shape[-2:], mode='bilinear', align_corners=True)
    pred_segm_map = output_large.argmax(dim=1).squeeze(dim=0).cpu().numpy()
    # pred_mask = get_color_pallete(pred_segm_map, "city")
    # pred_mask = pred_mask.convert('RGB')
    gt_segm_map = gt_segm_map.squeeze(dim=0).cpu().numpy()
    # gt_mask = get_color_pallete(gt_segm_map, "city")
    # gt_mask = gt_mask.convert('RGB')

    # # plot gt semantic segmentation map
    # axes[1,0].set_title('GT Segmentation Map')
    # # axes[1,0].imshow(img_np, cmap=cmap1)
    # im_gt = axes[1,0].imshow(gt_mask) #, cmap=cmap2, alpha=alpha)
    # axes[1,0].xaxis.set_visible(False)
    # axes[1,0].yaxis.set_visible(False)

    # # plot predicted semantic segmentation map
    # axes[1,1].set_title('Predicted Segmentation Map')
    # # axes[1,1].imshow(img_np, cmap=cmap1)
    # im_pred = axes[1,1].imshow(pred_mask) #, cmap=cmap2, alpha=alpha)
    # axes[1,1].xaxis.set_visible(False)
    # axes[1,1].yaxis.set_visible(False)


    mask1 = pred_segm_map != gt_segm_map
    mask2 = gt_segm_map != 255

    wrong_pred_mask = (mask1 * mask2)*1.0
    wrong_pred_mask = F.interpolate(torch.Tensor(wrong_pred_mask).unsqueeze(0).unsqueeze(0), size=decoder_out.shape[-2:], mode='nearest')[0,0]   # 640, 1280 -->  160, 320
    wrong_pred_mask = wrong_pred_mask.cpu().numpy()

    axes[0,1].set_title('Hyper Uncertainty of Wrong Predictions')
    axes[0,1].imshow(img_np, cmap=cmap1)
    im_score = axes[0,1].imshow(hypunc_np*wrong_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[0,1].xaxis.set_visible(False)
    axes[0,1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    axes[1,1].set_title('Entropy of Wrong Predictions')
    axes[1,1].imshow(img_np, cmap=cmap1)
    im_score = axes[1,1].imshow(entropy_np*wrong_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[1,1].xaxis.set_visible(False)
    axes[1,1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    axes[2,1].set_title('Impurity of Wrong Predictions')
    axes[2,1].imshow(img_np, cmap=cmap1)
    im_score = axes[2,1].imshow(impurity_np*wrong_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[2,1].xaxis.set_visible(False)
    axes[2,1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    axes[3,1].set_title('Hyper Certainty of Wrong Predictions')
    axes[3,1].imshow(img_np, cmap=cmap1)
    im_score = axes[3,1].imshow(hypcert_np*wrong_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[3,1].xaxis.set_visible(False)
    axes[3,1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[3,1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')
    




    mask1 = pred_segm_map == gt_segm_map
    mask2 = gt_segm_map != 255

    correct_pred_mask = (mask1 * mask2)*1.0
    correct_pred_mask = F.interpolate(torch.Tensor(correct_pred_mask).unsqueeze(0).unsqueeze(0), size=decoder_out.shape[-2:], mode='nearest')[0,0]   # 640, 1280 -->  160, 320
    correct_pred_mask = correct_pred_mask.cpu().numpy()

    axes[0,2].set_title('Hyper Uncertainty of Correct Predictions')
    axes[0,2].imshow(img_np, cmap=cmap1)
    im_score = axes[0,2].imshow(hypunc_np*correct_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[0,2].xaxis.set_visible(False)
    axes[0,2].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    axes[1,2].set_title('Entropy of Correct Predictions')
    axes[1,2].imshow(img_np, cmap=cmap1)
    im_score = axes[1,2].imshow(entropy_np*correct_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[1,2].xaxis.set_visible(False)
    axes[1,2].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[1,2])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    axes[2,2].set_title('Impurity of Correct Predictions')
    axes[2,2].imshow(img_np, cmap=cmap1)
    im_score = axes[2,2].imshow(impurity_np*correct_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[2,2].xaxis.set_visible(False)
    axes[2,2].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[2,2])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')

    axes[3,2].set_title('Hyper Certainty of Correct Predictions')
    axes[3,2].imshow(img_np, cmap=cmap1)
    im_score = axes[3,2].imshow(hypcert_np*correct_pred_mask,  cmap=cmap2, alpha=alpha)
    axes[3,2].xaxis.set_visible(False)
    axes[3,2].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[3,2])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='right')


    # make directory if it doesn't exist
    # if not os.path.exists(cfg.OUTPUT_DIR + '/viz'):
    #     os.makedirs(cfg.OUTPUT_DIR + '/viz')
    # name = name.rsplit('/', 1)[-1].rsplit('_', 1)[0]
    # file_name = cfg.OUTPUT_DIR + '/viz/' + name + '_wrong.png'

    # plt.suptitle(name)
    plt.savefig(name)
    plt.close()