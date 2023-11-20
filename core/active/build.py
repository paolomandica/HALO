import math
import torch
import copy

import numpy as np
import torch.nn.functional as F

# from kmeans_pytorch import kmeans

from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore

import os
from core.configs import cfg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from core.utils.misc import get_color_pallete

CITYSCAPES_MEAN = torch.Tensor([123.675, 116.28, 103.53]).reshape(1, 1, 3).numpy()
CITYSCAPES_STD = torch.Tensor([58.395, 57.12, 57.375]).reshape(1, 1, 3).numpy()

np.random.seed(cfg.SEED + 1)
VIZ_LIST = list(np.random.randint(0, 500, 20))


def select_pixels_to_label(
    score,
    active_regions,
    active_radius,
    mask_radius,
    active,
    selected,
    active_mask,
    ground_truth,
):
    for pixel in range(active_regions):
        values, indices_h = torch.max(score, dim=0)
        max_value, indices_w = torch.max(values, dim=0)
        if max_value == -float("inf"):
            break
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
        score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float("inf")
        active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
        selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
        # active sampling
        active_mask[
            active_start_h:active_end_h, active_start_w:active_end_w
        ] = ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

    return score, active, selected, active_mask


def to_np_array(tensor):
    return np.array(tensor.cpu().numpy(), dtype=np.uint8)


def RegionSelection(cfg, feature_extractor, classifier, tgt_epoch_loader, round_number):
    feature_extractor.eval()
    classifier.eval()

    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.MASK_RADIUS_K
    active_budget = cfg.ACTIVE.BUDGET / len(cfg.ACTIVE.SELECT_ITER)
    uncertainty_type = cfg.ACTIVE.UNCERTAINTY
    purity_type = cfg.ACTIVE.PURITY
    K = cfg.ACTIVE.K
    is_segformer = cfg.MODEL.NAME.startswith("segformer")
    if is_segformer:
        segformer_config = feature_extractor.config

    floating_region_score = FloatingRegionScore(
        in_channels=cfg.MODEL.NUM_CLASSES,
        size=2 * active_radius + 1,
        purity_type=purity_type,
        K=K,
    )

    with torch.no_grad():
        idx = 0
        for tgt_data in tqdm(tgt_epoch_loader):
            tgt_input, path2mask = tgt_data["img"], tgt_data["path_to_mask"]
            origin_mask, origin_label = (
                tgt_data["origin_mask"],
                tgt_data["origin_label"],
            )
            origin_size = tgt_data["size"]
            active_indicator = tgt_data["active"]
            selected_indicator = tgt_data["selected"]
            path2indicator = tgt_data["path_to_indicator"]

            tgt_input = tgt_input.cuda(non_blocking=True)

            if idx == 0:
                feature_extractor.to(tgt_input.device)
                classifier.to(tgt_input.device)

            tgt_size = tgt_input.shape[-2:]
            if is_segformer:
                return_dict = segformer_config.return_dict
                outputs = feature_extractor(
                    tgt_input, output_hidden_states=True, return_dict=return_dict
                )
                tgt_feat = outputs.hidden_states if return_dict else outputs[1]
            else:
                tgt_feat = feature_extractor(tgt_input)
            tgt_out, decoder_out = classifier(tgt_feat, size=tgt_size)

            # just a single iteration, because len(origin_mask)=1
            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i : i + 1, :, :, :]
                output = F.interpolate(
                    output, size=size, mode="bilinear", align_corners=True
                )

                if (
                    uncertainty_type in ["certainty", "hyperbolic"]
                    or (purity_type in ["hyper", "radius", "euc_norm"])
                    or (uncertainty_type == "none" and cfg.MODEL.HYPER)
                ):
                    decoder_out = decoder_out[i : i + 1, :, :, :]
                    decoder_out = F.interpolate(
                        decoder_out, size=size, mode="bilinear", align_corners=True
                    )

                score, _, _ = floating_region_score(
                    output,
                    decoder_out=decoder_out,
                    normalize=cfg.ACTIVE.NORMALIZE,
                    unc_type=uncertainty_type,
                    pur_type=purity_type,
                    ground_truth=ground_truth,
                )
                score_clone = score.clone()
                score[active] = -float("inf")

                active_regions = math.ceil(
                    num_pixel_cur * active_budget / per_region_pixels
                )
                score, active, selected, active_mask = select_pixels_to_label(
                    score,
                    active_regions,
                    active_radius,
                    mask_radius,
                    active,
                    selected,
                    active_mask,
                    ground_truth,
                )

                active_mask_np = to_np_array(active_mask)
                active_mask_IMG = Image.fromarray(active_mask_np)
                active_mask_IMG.save(path2mask[i])
                indicator = {"active": active, "selected": selected}
                torch.save(indicator, path2indicator[i])

            if cfg.ACTIVE.VIZ_MASK and idx in VIZ_LIST:
                img_np = (
                    F.interpolate(
                        tgt_input, size=size, mode="bilinear", align_corners=True
                    )
                    .cpu()
                    .numpy()[0]
                    .transpose(1, 2, 0)
                )
                img_np = (img_np * CITYSCAPES_STD + CITYSCAPES_MEAN).astype(np.uint8)
                name = tgt_data["name"][0]
                score_np = score_clone.cpu().numpy()
                visualization_plots(
                    img_np, score_np, active_mask_np, round_number, name
                )
            idx += 1

    feature_extractor.train()
    classifier.train()


def visualization_plots(
    img_np,
    score_np,
    active_mask_np,
    round_number,
    name,
    cmap1="gray",
    cmap2="viridis",
    alpha=0.7,
    title=None,
):
    fig, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(10, 10))

    # plot original image
    # axes[0].set_title('Original Image')
    axes[0].imshow(img_np)
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)

    if title is None:
        if cfg.ACTIVE.UNCERTAINTY == "entropy":
            title = "Entropy + "
        elif cfg.ACTIVE.UNCERTAINTY == "hyperbolic":
            title = "Hyperbolic Uncertainty + "
        elif cfg.ACTIVE.UNCERTAINTY == "certainty":
            title = "Hyperbolic Certainty + "
        else:
            title = ""

        if cfg.ACTIVE.PURITY == "ripu":
            title += "Impurity"

    axes[1].set_title("Total Score: " + title)
    axes[1].imshow(img_np, cmap=cmap1)
    im_score = axes[1].imshow(score_np, cmap=cmap2, alpha=alpha)
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im_score, cax=cax, location="right")

    # plot original image
    axes[2].set_title("Selected Pixel - Active Round: " + str(round_number))
    axes[2].imshow(img_np, cmap=cmap1)
    axes[2].imshow(active_mask_np, cmap="autumn", alpha=alpha)
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)

    # make directory if it doesn't exist
    if not os.path.exists(cfg.SAVE_DIR + "/viz"):
        os.makedirs(cfg.SAVE_DIR + "/viz")
    name = name.rsplit("/", 1)[-1].rsplit("_", 1)[0]
    file_name = cfg.SAVE_DIR + "/viz/" + name + "_round" + str(round_number) + ".png"

    plt.suptitle(name)
    plt.savefig(file_name)
    plt.close()
