import os
import shutil
from tqdm import tqdm


def find_unique_subfolder_names(root_dir, middle_dir):
    condition_folders = os.listdir(root_dir)
    subdirs = dict()
    num_subdirs = 0
    total_subdirs = 0

    # check total number of subdirs
    for condition_dir in condition_folders:
        level2_path = os.path.join(root_dir, condition_dir, middle_dir)
        level3_folders = os.listdir(level2_path)
        total_subdirs += len(level3_folders)
    print(f"\nTotal number of subdirs: {total_subdirs}\n")

    for condition_dir in condition_folders:
        print(f"Checking {condition_dir}...")
        level2_path = os.path.join(root_dir, condition_dir, middle_dir)
        level3_folders = os.listdir(level2_path)
        for subfolder in level3_folders:
            if subfolder not in subdirs:
                subdirs[subfolder] = condition_dir
                num_subdirs += 1
            else:
                print(f"'{subfolder}' from '{condition_dir}' already exists in '{subdirs[subfolder]}'.")
    print("\n")
    print(f"Number of unique subfolders: {num_subdirs}")


def move_subfolders_to_condition_level(root_dir, middle_dirs):
    condition_folders = os.listdir(root_dir)

    for middle_dir in middle_dirs:
        print(f"Moving subfolders from {middle_dir}...")
        for condition_dir in condition_folders:
            if condition_dir in middle_dirs:
                continue

            print(f"Moving subfolders from {condition_dir}...")
            src_dir = os.path.join(root_dir, condition_dir, middle_dir)
            dest_dir = os.path.join(root_dir, middle_dir)

            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)

            subfolders = os.listdir(src_dir)
            for subfolder in tqdm(subfolders):
                src_subfolder = os.path.join(src_dir, subfolder)
                dest_subfolder = os.path.join(dest_dir, subfolder)
                # print(f"Moving {src_subfolder} to {dest_subfolder}...")
                shutil.move(src_subfolder, dest_subfolder)


def create_txt_train_list(dirpath, split='train'):
    directory = f"images/{split}"
    directory = os.path.join(dirpath, directory)
    output_file = f"datasets/acdc_{split}_list.txt"
    print(f'\nCreating {output_file}\n')

    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.png'):
                    subdir = root.split('/')[-1]
                    f.write(os.path.join(subdir, filename) + '\n')


if __name__ == "__main__":
    root_dir = "datasets/acdc"
    splits = ["train", "val"]
    # find_unique_subfolder_names(root_dir, middle_dir)
    # move_subfolders_to_condition_level(root_dir, splits)
    create_txt_train_list(root_dir, split='val')