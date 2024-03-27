 ---  
 
# Hyperbolic Active Learning

## Overview
We propose ...

## Usage

### Prerequisites

- Python 3.9
- Pytorch 1.13
- torchvision 0.14

### Step-by-step installation with conda

```bash
conda create --name halo -y python=3.9
conda activate halo

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# this installs required packages
pip install -r requirements.txt
```

### Step-by-step installation with Docker

```bash
export UID=$(id -u)
export GID=$(id -g)
export DATA_VOLUME="/mnt/homes/paolo/"

docker compose build
docker compose up -d
```

### Data Preparation

- Download [The Cityscapes Dataset](https://www.cityscapes-dataset.com/), [The GTAV Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/), and [The SYNTHIA Dataset](https://synthia-dataset.net/)

Symlink the required dataset

```bash
ln -s /path_to_cityscapes_dataset datasets/cityscapes
ln -s /path_to_gtav_dataset datasets/gtav
ln -s /path_to_synthia_dataset datasets/synthia
```

Generate the label static files for GTAV/SYNTHIA Datasets by running

```bash
python datasets/generate_gtav_label_info.py -d datasets/gtav -o datasets/gtav/
python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
```

The data folder should be structured as follows:

```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gtav/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
│   └──	synthia
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   └──	
```

### Usage

The config files for the different ADA training protocols can be found in the `configs` directory.

#### Training

```bash
export NCCL_IB_TIMEOUT=22
python train.py -cfg CONFIG_PATH
```

#### Testing

```bash
python test.py -cfg CONFIG_PATH
```


## Acknowledgements
This project is based on the [RIPU](https://github.com/BIT-DA/RIPU) open-source project, with our re-implementation in PyTorch Lightning for multi-gpu active learning. We thank the authors for making the source code publically available.
