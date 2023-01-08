python3 tools/test.py --config_file='configs/HRNet32.yml' DATASETS.NAMES "('occluded_dukemtmc')" MODEL.NAME "('HRNet32')" \
MODEL.DEVICE_ID "('0')" CLUSTERING.PART_NUM "(4)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pl-HRNet32-OD-1')" \
TEST.WEIGHT "('/mnt/Data/dsg/ISP/log/Best-OD-mAP55.6-HRNet32/HRNet32_model_120.pth')" \
TEST.WITH_ARM "(True)" MODEL.SEG_F "('part_t')"

# /mnt/Data/dsg/ISP/log/TSCL-OD-HRNet32-1/HRNet32_model_120.pth
# /mnt/Data/dsg/ISP/log/Best-OD-mAP55.6-HRNet32/HRNet32_model_120.pth
# /mnt/Data/dsg/ISP/log/Best-OD-Rank165.6-HRNet32/HRNet32_model_120.pth
# MODEL.SEG_F "('part_t')"