import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils import data

pillow_interp_codes = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'box': Image.BOX,
    'lanczos': Image.LANCZOS,
    'hamming': Image.HAMMING
}


class ACDC(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=19,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
            cfg=None,
            empty=False,
    ):
        self.active = True if split == 'active' else False
        if split == 'active':
            split = 'train'
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.cfg = cfg
        self.empty = empty
        with open(data_list, "r") as handle:
            content = handle.readlines()

        self.data_list = []
        if empty:
            self.data_list.append(
                {
                    "img": "",
                    "label": "",
                    "label_mask": "",
                    "name": "",
                }
            )
        else:
            for fname in content:
                name = fname.strip()
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.data_root,
                            "images/%s/%s" % (self.split, name)
                        ),
                        "label": os.path.join(
                            self.data_root,
                            "gt/%s/%s"
                            % (
                                self.split,
                                name.split("_rgb_anon")[0]
                                + "_gt_labelIds.png"
                            ),
                        ),
                        "label_mask": os.path.join(
                            self.cfg.OUTPUT_DIR,
                            "gtMask/%s/%s"
                            % (
                                self.split,
                                name.split("_rgb_anon")[0]
                                + "_gt_labelIds.png",
                            ),
                        ),
                        "name": name,
                        'indicator': os.path.join(
                            cfg.OUTPUT_DIR,
                            "gtIndicator/%s/%s"
                            % (
                                "train",
                                name.split("_rgb_anon")[0]
                                + "_indicator.pth",
                            ),
                        )
                    }
                )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        # GTAV
        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
        if self.NUM_CLASS == 16:  # SYNTHIA
            self.id_to_trainid = {
                7: 0,
                8: 1,
                11: 2,
                12: 3,
                13: 4,
                17: 5,
                19: 6,
                20: 7,
                21: 8,
                23: 9,
                24: 10,
                25: 11,
                26: 12,
                28: 13,
                32: 14,
                33: 15,
            }
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }

        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)
        label_mask = None
        if self.split == 'train':
            label_mask = np.array(Image.open(datafiles["label_mask"]), dtype=np.uint8)
        else:
            # test or val, mask is useless
            label_mask = np.ones_like(label, dtype=np.uint8) * 255

        # for generate new mask
        origin_mask = torch.from_numpy(label_mask).long()

        active_indicator = torch.tensor([0])
        active_selected = torch.tensor([0])
        if self.active:
            indicator = torch.load(datafiles['indicator'])
            active_indicator = indicator['active']
            active_selected = indicator['selected']
            # if first time load, initialize it
            if active_indicator.size() == (1,):
                active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
                active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = np.array(label_copy, dtype=np.uint8)

        origin_label = torch.from_numpy(label).long()

        label.resize(label.shape[0], label.shape[1], 1)
        label_mask.resize(label_mask.shape[0], label_mask.shape[1], 1)

        h, w = label.shape[0], label.shape[1]

        mask_aggregation = np.concatenate((label, label_mask), axis=2)
        mask_aggregation = Image.fromarray(mask_aggregation)

        if self.transform is not None:
            image, mask_aggregation = self.transform(image, mask_aggregation)
            label = mask_aggregation[:, :, 0]
            label_mask = mask_aggregation[:, :, 1]

        ret_data = {
            "img": image,  # data
            'label': label,  # for test
            'mask': label_mask,  # for train
            'name': datafiles['name'],  # for test to store the results
            'path_to_mask': datafiles['label_mask'],  # for active to store new mask
            'path_to_indicator': datafiles['indicator'],  # store new indicator
            'size': torch.tensor([h, w]),  # for active to interpolate the output to original size
            'origin_mask': origin_mask,  # mask without transforms for active
            'origin_label': origin_label,  # label without transforms for active
            'active': active_indicator,  # indicate region or pixels can not be selected
            'selected': active_selected,  # indicate the pixel have been selected, can calculate the class-wise ratio of selected samples
        }

        return ret_data


class ACDC_old(data.Dataset):

    orig_dims = (1080, 1920)

    def __init__(
            self,
            root: str,
            stage: str = "train",
            condition: Union[List[str], str] = [
                "fog", "night", "rain", "snow"],
            load_keys: Union[List[str], str] = [
                "image_ref", "image", "semantic"],
            dims: Union[Tuple[int, int], List[int]] = (1080, 1920),
            transforms: Optional[Callable] = None,
            predict_on: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__()
        self.root = root
        self.dims = dims
        self.transforms = transforms

        assert stage in ["train", "val", "test", "predict"]
        self.stage = stage

        # mapping from stage to splits
        if self.stage == 'train':
            self.split = 'train'
        elif self.stage == 'val':
            self.split = 'val'
        elif self.stage == 'test':
            self.split = 'val'  # test on val split
        elif self.stage == 'predict':
            if not predict_on:
                self.split = 'test'  # predict on test split
            else:
                self.split = predict_on

        if isinstance(condition, str):
            self.condition = [condition]
        else:
            self.condition = condition

        if isinstance(load_keys, str):
            self.load_keys = [load_keys]
        else:
            self.load_keys = load_keys

        self.paths = {k: []
                      for k in ['image', 'image_ref', 'semantic']}

        self.images_dir = os.path.join(self.root, 'rgb_anon')
        self.semantic_dir = os.path.join(self.root, 'gt')
        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.semantic_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "condition" are inside the "root" directory')

        for cond in self.condition:
            img_parent_dir = os.path.join(self.images_dir, cond, self.split)
            semantic_parent_dir = os.path.join(
                self.semantic_dir, cond, self.split)
            for recording in os.listdir(img_parent_dir):
                img_dir = os.path.join(img_parent_dir, recording)
                semantic_dir = os.path.join(semantic_parent_dir, recording)
                for file_name in os.listdir(img_dir):
                    for k in ['image', 'image_ref', 'semantic']:
                        if k == 'image':
                            file_path = os.path.join(img_dir, file_name)
                        elif k == 'image_ref':
                            ref_img_dir = img_dir.replace(
                                self.split, self.split + '_ref')
                            ref_file_name = file_name.replace(
                                'rgb_anon', 'rgb_ref_anon')
                            file_path = os.path.join(
                                ref_img_dir, ref_file_name)
                        elif k == 'semantic':
                            semantic_file_name = file_name.replace(
                                'rgb_anon.png', 'gt_labelTrainIds.png')
                            file_path = os.path.join(
                                semantic_dir, semantic_file_name)
                        self.paths[k].append(file_path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        sample: Any = {}
        sample['filename'] = self.paths['image'][index].split('/')[-1]

        for k in self.load_keys:
            if k in ['image', 'image_ref']:
                data = Image.open(self.paths[k][index]).convert('RGB')
                if data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes['bilinear'])
            elif k == 'semantic':
                data = Image.open(self.paths[k][index])
                if data.size != self.dims[::-1]:
                    data = data.resize(
                        self.dims[::-1], resample=pillow_interp_codes['nearest'])
            else:
                raise ValueError('invalid load_key')
            sample[k] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(next(iter(self.paths.values())))
