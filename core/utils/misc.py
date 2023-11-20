import argparse
from collections import OrderedDict
import errno
import os
import numpy as np
from PIL import Image

import torch
from core.configs import cfg


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection.float().cpu(), bins=K, min=0, max=K - 1
    )
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def get_color_pallete(npimg, dataset="voc"):
    out_img = Image.fromarray(npimg.astype("uint8")).convert("P")
    if dataset == "city":
        cityspallete = [
            128,
            64,
            128,
            244,
            35,
            232,
            70,
            70,
            70,
            102,
            102,
            156,
            190,
            153,
            153,
            153,
            153,
            153,
            250,
            170,
            30,
            220,
            220,
            0,
            107,
            142,
            35,
            152,
            251,
            152,
            0,
            130,
            180,
            220,
            20,
            60,
            255,
            0,
            0,
            0,
            0,
            142,
            0,
            0,
            70,
            0,
            60,
            100,
            0,
            80,
            100,
            0,
            0,
            230,
            119,
            11,
            32,
        ]
        out_img.putpalette(cityspallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Domain Adaptive Semantic Segmentation Training"
    )
    parser.add_argument(
        "-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--proctitle",
        type=str,
        default="HALO",
        help="allow a process to change its title",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.opts is not None and args.opts != []:
        args.opts[-1] = args.opts[-1].strip("\r\n")

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    print("Saving to {}".format(cfg.SAVE_DIR))
    cfg.freeze()

    return args

# if "classifier.classifier.P_MLR" in model_weights:
#     model_weights["classifier.P_MLR"] = model_weights.pop(
#         "classifier.classifier.P_MLR"
#     )
# if "classifier.classifier.A_MLR" in model_weights:
#     model_weights["classifier.A_MLR"] = model_weights.pop(
#         "classifier.classifier.A_MLR"
#     )

def load_checkpoint(model, path, module="feature_extractor"):
    print("Loading checkpoint from {}".format(path))
    if str(path).endswith(".ckpt"):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
        # Create a mapping to match keys in the checkpoint to keys in the model
        key_mapping = {}
        for k, v in checkpoint.items():
            if k.startswith(module):
                new_key = k[len(module) + 1:]  # Remove "classifier." prefix
                key_mapping[new_key] = v
        model.load_state_dict(key_mapping)
    elif str(path).endswith(".pth"):
        checkpoint = torch.load(cfg.resume, map_location=torch.device("cpu"))
        model_weights = checkpoint[module]
        model_weights = strip_prefix_if_present(checkpoint[module], "module.")
        model.load_state_dict(model_weights)
    else:
        raise NotImplementedError("Only support .ckpt and .pth file")


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict
