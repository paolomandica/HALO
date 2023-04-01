# GTAV -> Cityscapes ra, deeplabv3+, 5%
python train.py -cfg configs/gtav/deeplabv3plus_r101_RA.yaml OUTPUT_DIR results/v3plus_gtav_ra_5.0_precent

# GTAV -> Cityscapes ra, deeplabv2, 2.2%
python train.py -cfg configs/gtav/deeplabv2_r101_RA.yaml OUTPUT_DIR results/v2_gtav_ra_2.2_precent

# GTAV -> Cityscapes pa, deeplabv2, 40 pixels
python train.py -cfg configs/gtav/deeplabv2_r101_PA.yaml OUTPUT_DIR results/v2_gtav_pa_40_pixel

# [source-free scenario] GTAV -> Cityscapes ra, deeplabv2, 2.2%
CUDA_VISIBLE_DEVICES=0 python train_source.py -cfg configs/gtav/deeplabv2_r101_src.yaml OUTPUT_DIR results/pretrain_gtav
CUDA_VISIBLE_DEVICES=1 python train_source.py -cfg configs/gtav/deeplabv2_r101_src_hyper.yaml OUTPUT_DIR results/pretrain_gtav_hyper
python train_source_free.py -cfg configs/gtav/deeplabv2_r101_RA_source_free.yaml OUTPUT_DIR results/v2_gtav_ra_2.2_precent_source_free resume results/source_free/gtav_source_only_iter020000.pth

# [source-only test] GTAV -> Cityscapes, deeplabv2
CUDA_VISIBLE_DEVICES=2 python test.py -cfg configs/gtav/deeplabv2_r101_RA_source_free.yaml resume results/pretrain_gtav/model_iter020000.pth OUTPUT_DIR results/test
CUDA_VISIBLE_DEVICES=2 python test.py -cfg configs/gtav/hyper_deeplabv2_r101_RA_source_free.yaml resume results/pretrain_gtav_hyper_10/model_iter020000.pth OUTPUT_DIR results/test