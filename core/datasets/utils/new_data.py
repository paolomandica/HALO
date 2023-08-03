import os

import torch
from tqdm.auto import tqdm
from PIL import Image

from core.utils.misc import get_color_pallete

categories = ["bicycle", "bus", "light", "motorcycle", "rider"]
dirpath = 'datasets/halo_extra_data_10k'


def get_name2id_mapping():
    trainid2name = {
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
        255: "unknown",
    }

    name2id = {v: k for k, v in trainid2name.items()}
    name2id["tree"] = name2id["vegetation"]
    name2id["traffic light"] = name2id["light"]
    name2id["motorcycle"] = name2id["motocycle"]
    name2id["traffic sign"] = name2id["sign"]
    return name2id


def parse_info(info) -> dict:
    id2category = {0: 255}
    for elem in info:
        id2category[elem['id']] = elem['category_id']
    return id2category


def process_mask(mask, id2category, labels, name2id) -> torch.Tensor:
    unique = torch.unique(mask)
    new_mask = mask.clone()
    for i in unique:
        category_id = id2category[i.item()]
        if category_id == 255 or category_id == 0:
            new_mask[new_mask == i.item()] = 255
        else:
            category_name = labels[category_id]
            class_id = name2id[category_name]
            new_mask[new_mask == i] = class_id
    return new_mask


def generate_mask(data, category, labels, name2id):
    new_masks = torch.zeros((len(data), 640, 1280), dtype=torch.uint8)

    outdir = f'gtFine/train/{category}'
    outdir = os.path.join(dirpath, outdir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, (k, v) in enumerate(tqdm(data.items())):
        img_id = k
        mask = v['mask']
        info = v['info']
        id2category = parse_info(info)
        new_mask = process_mask(mask, id2category, labels, name2id)
        new_masks[i] = new_mask

        # save masks
        filepath_id = os.path.join(outdir, f'{category}_{img_id}_000019_gtFine_labelIds.png')
        filepath_color = os.path.join(outdir, f'{category}_{img_id}_000019_gtFine_color.png')

        png_id = new_mask.numpy()
        png_id = Image.fromarray(png_id)
        png_id = png_id.convert('L')
        png_id.save(filepath_id)

        png_color = new_mask.numpy()
        png_color = get_color_pallete(png_color)
        png_color.save(filepath_color)


def create_new_masks():
    print('\nCreating new masks\n')
    for category in categories:
        print(f'\nProcessing {category}')
        file_path = f'hipie/{category}_annotations.pt'
        file_path = os.path.join(dirpath, file_path)
        data = torch.load(file_path)
        labels = data.pop('labels')

        name2id = get_name2id_mapping()

        generate_mask(data, category, labels, name2id)


def create_txt_train_list():
    print('\nCreating txt train list\n')
    directory = 'leftImg8bit/train'
    directory = os.path.join(dirpath, directory)
    output_file = 'datasets/extra_train_list.txt'
    if '10k' in dirpath:
        output_file = 'datasets/extra_train_list_10k.txt'

    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.png'):
                    subdir = root.split('/')[-1]
                    f.write(os.path.join(subdir, filename) + '\n')


def rename_images():
    print('\nRenaming images\n')
    for category in categories:
        print(f'\nProcessing {category}')
        directory = f'leftImg8bit/train/{category}'
        directory = os.path.join(dirpath, directory)
        for root, dirs, files in os.walk(directory):
            for filename in tqdm(files):
                if filename.endswith('.png'):
                    file_id = filename.split('.')[0]
                    new_filename = f'{category}_{file_id}_000019_leftImg8bit.png'
                    os.rename(os.path.join(root, filename), os.path.join(root, new_filename))


def resize_labels():
    print('\nResizing labels\n')
    directory = 'gtFine/train'
    directory = os.path.join(dirpath, directory)
    for root, dirs, files in os.walk(directory):
        for filename in tqdm(files):
            if filename.endswith('.png'):
                filepath = os.path.join(root, filename)
                img = Image.open(filepath)
                img = img.resize((512, 512), Image.NEAREST)
                img.save(filepath)


if __name__ == '__main__':
    # rename_images()
    # create_txt_train_list()
    create_new_masks()
    # resize_labels()
