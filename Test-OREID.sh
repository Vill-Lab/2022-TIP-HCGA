python3 tools/test.py --config_file='configs/HRNet32.yml' DATASETS.NAMES "('occluded_reid')" MODEL.NAME "('HRNet32')" \
MODEL.DEVICE_ID "('0')" CLUSTERING.PART_NUM "(4)" \
TEST.WITH_ARM "(True)" MODEL.SEG_F "('part_t')" \
TEST.WEIGHT "('/mnt/Data/dsg/ISP/log/HCNet-OREID-4/HRNet32_model_120.pth')"

# /mnt/Data/dsg/ISP/log/HCNet-OREID-4/HRNet32_model_120.pth
# /mnt/Data/dsg/ISP/log/Best-OREID-Rank187.2-HRNet32/HRNet32_model_120.pth