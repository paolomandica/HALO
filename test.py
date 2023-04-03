import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import re

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn.functional as F
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger
from core.active.floating_region import FloatingRegionScore

import setproctitle
import warnings
warnings.filterwarnings('ignore')

# MEAN : 0.28689554, 0.32513303, 0.28389177 - 123.675, 116.28 , 103.53
# STD  : 0.18696375, 0.19017339, 0.18720214 - 58.395, 57.12 , 57.375
CITYSCAPES_MEAN = torch.Tensor([123.675, 116.28, 103.53]).reshape(1,1,3).numpy()
CITYSCAPES_STD = torch.Tensor([58.395, 57.12, 57.375]).reshape(1,1,3).numpy()

np.random.seed(cfg.SEED+1)
VIZ_LIST = list(np.random.randint(0, 500, 20))



def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def inference(feature_extractor, classifier, image, label, flip=False, vis_score=True, vis_mask=False, name=None, cfg=None, idx=None):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        if not cfg.MODEL.HYPER:
            output = classifier(feature_extractor(image))
        else:
            output, decoder_out = classifier(feature_extractor(image))

    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]  # 19, h, w

    if cfg.MODEL.HYPER:
        decoder_out = F.interpolate(decoder_out, size=size, mode='bilinear', align_corners=True)
        if flip:
            decoder_out = (decoder_out[0] + decoder_out[1].flip(2)) / 2
        else:      
            decoder_out = decoder_out[0]    # ch, h, w
        

    if vis_score and idx in VIZ_LIST:
        image = F.interpolate(image, size=size, mode='bilinear', align_corners=True)
        visualize_score(image[0], output.unsqueeze(dim=0), decoder_out.unsqueeze(dim=0), label, name, cfg)

    return output.unsqueeze(dim=0)


def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


def visualize_score(image, output, decoder_out, gt_segm_map, name, cfg, cmap1='gray', cmap2='viridis', alpha=0.7):

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1, cfg=cfg).cuda()
    # per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    # active_radius = cfg.ACTIVE.RADIUS_K
    # mask_radius = cfg.ACTIVE.RADIUS_K * 2
    # active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)

    if cfg.ACTIVE.UNCERTAINTY == 'entropy':
        score, purity, uncertainty = floating_region_score(output)
    elif cfg.ACTIVE.UNCERTAINTY == 'hyperbolic' or cfg.ACTIVE.UNCERTAINTY == 'certainty':
        score, purity, uncertainty = floating_region_score(output, decoder_out=decoder_out)
    elif cfg.ACTIVE.UNCERTAINTY == 'none' and cfg.ACTIVE.PURITY == 'ripu':
        if cfg.MODEL.HYPER:
            score, purity, uncertainty = floating_region_score(output, decoder_out=decoder_out)
        else:
            score, purity, uncertainty = floating_region_score(output)

    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * CITYSCAPES_STD + CITYSCAPES_MEAN).astype(np.uint8)
    # img_np = image.cpu().numpy().astype(np.uint8)  #  Is it denormalized?

    score_np = score.cpu().numpy()
    uncertainty_np = uncertainty.cpu().numpy()
    purity_np = purity.cpu().numpy()


    fig, axes = plt.subplots(2, 3, constrained_layout = True, figsize=(10, 10))

    axes[0,0].set_title('Total Score')
    axes[0,0].imshow(img_np, cmap=cmap1)
    im_score = axes[0,0].imshow(score_np,  cmap=cmap2, alpha=alpha)
    axes[0,0].xaxis.set_visible(False)
    axes[0,0].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("bottom", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location='bottom')

    if cfg.ACTIVE.UNCERTAINTY == 'entropy':
        title = 'Entropy'
    elif cfg.ACTIVE.UNCERTAINTY == 'hyperbolic':
        title = 'Hyperbolic Uncertainty'
    elif cfg.ACTIVE.UNCERTAINTY == 'certainty':
        title = 'Hyperbolic Certainty'
    axes[0,1].set_title(title)
    axes[0,1].imshow(img_np, cmap=cmap1)
    im_uncertainty = axes[0,1].imshow(uncertainty_np,  cmap=cmap2, alpha=alpha)
    axes[0,1].xaxis.set_visible(False)
    axes[0,1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes("bottom", size="20%", pad=0.05)
    plt.colorbar(im_uncertainty, cax=cax, location='bottom')

    axes[0,2].set_title('Impurity')
    axes[0,2].imshow(img_np, cmap=cmap1)
    im_purity = axes[0,2].imshow(purity_np,  cmap=cmap2, alpha=alpha)
    axes[0,2].xaxis.set_visible(False)
    axes[0,2].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes("bottom", size="20%", pad=0.05)
    plt.colorbar(im_purity, cax=cax, location='bottom')

    # make directory if it doesn't exist
    if not os.path.exists(cfg.OUTPUT_DIR + '/viz'):
        os.makedirs(cfg.OUTPUT_DIR + '/viz')
    name = name.rsplit('/', 1)[-1].rsplit('_', 1)[0]
    file_name = cfg.OUTPUT_DIR + '/viz/' + name + '.png'






    pred_segm_map = output.argmax(dim=1).squeeze(dim=0).cpu().numpy()
    pred_mask = get_color_pallete(pred_segm_map, "city")
    pred_mask = pred_mask.convert('RGB')
    gt_segm_map = gt_segm_map.squeeze(dim=0).cpu().numpy()
    gt_mask = get_color_pallete(gt_segm_map, "city")
    gt_mask = gt_mask.convert('RGB')



    # plot gt semantic segmentation map
    axes[1,0].set_title('GT Segmentation Map')
    # axes[1,0].imshow(img_np, cmap=cmap1)
    im_gt = axes[1,0].imshow(gt_mask) #, cmap=cmap2, alpha=alpha)
    axes[1,0].xaxis.set_visible(False)
    axes[1,0].yaxis.set_visible(False)

    # plot predicted semantic segmentation map
    axes[1,1].set_title('Predicted Segmentation Map')
    # axes[1,1].imshow(img_np, cmap=cmap1)
    im_pred = axes[1,1].imshow(pred_mask) #, cmap=cmap2, alpha=alpha)
    axes[1,1].xaxis.set_visible(False)
    axes[1,1].yaxis.set_visible(False)

    # plot original image
    axes[1,2].set_title('Original Image')
    axes[1,2].imshow(img_np)
    axes[1,2].xaxis.set_visible(False)
    axes[1,2].yaxis.set_visible(False)
    

    # plot difference between gt and predicted semantic segmentation map
    # axes[1,2].set_title('Difference Map')
    # axes[1,2].imshow(img_np, cmap=cmap1)
    # diff_map = gt_segm_map != pred_segm_map
    # im_diff = axes[1,2].imshow(diff_map, cmap=cmap2, alpha=alpha)
    # axes[1,2].xaxis.set_visible(False)
    # axes[1,2].yaxis.set_visible(False)

    plt.suptitle(name)
    plt.savefig(file_name)
    plt.close()

    # mask = get_color_pallete(pred_segm_map, "city")
    # if mask.mode == 'P':
    #     mask = mask.convert('RGB')




def test(cfg):
    logger = logging.getLogger("AL-RIPU.tester")
    logger.info("Start testing")
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    classifier.to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))['state_dict']
        # breakpoint()
        # feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor_weights = {k: v for k, v in checkpoint.items() if k.startswith('feature_extractor')}
        feature_extractor_weights = OrderedDict([[k.split('feature_extractor.')[-1], v.cpu()]
                                                for k, v in feature_extractor_weights.items()])
        feature_extractor.load_state_dict(feature_extractor_weights)
        # classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier_weights = {k: v for k, v in checkpoint.items() if k.startswith('classifier')}
        classifier_weights = OrderedDict([[k.split('classifier.')[-1], v.cpu()] for k, v in classifier_weights.items()])
        classifier.load_state_dict(classifier_weights)

    feature_extractor.eval()
    classifier.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    output_folder = '.'
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)

    assert cfg.TEST.BATCH_SIZE == 1, "Test batch size should be 1!"
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=None)
    idx = 0
    for batch in tqdm(test_loader):
        x, y, name = batch['img'], batch['label'], batch['name']
        name = name[0]
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True).long()

        flip = True
        pred = inference(feature_extractor, classifier, x, y, flip=flip, 
                         vis_score=cfg.TEST.VIZ_SCORE, vis_mask=cfg.TEST.VIZ_MASK, name=name, cfg=cfg, idx=idx)

        output = pred.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        # save the result
        pred = pred.cpu().numpy().squeeze().argmax(0)
        if cfg.MODEL.NUM_CLASSES == 16:
            pred = transform_color(pred)
        mask = get_color_pallete(pred, "city")
        mask_filename = name if len(name.split("/")) < 2 else name.split("/")[1]
        if mask.mode == 'P':
            mask = mask.convert('RGB')
        mask.save(os.path.join(output_folder, mask_filename))

        idx += 1

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    aAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('mIoU: {:.4f}'.format(mIoU))
    logger.info('mAcc: {:.4f}'.format(mAcc))
    logger.info('aAcc: {:.4f}'.format(aAcc))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info(
            '{} {} iou/accuracy: {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))


def main():
    parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--proctitle",
                        type=str,
                        default="AL-RIPU",
                        help="allow a process to change its title", )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setproctitle.setproctitle(f'{args.proctitle}')
    save_dir = ""
    logger = setup_logger("AL-RIPU", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg)


if __name__ == "__main__":
    main()
