python3 tools/test.py --config_file='configs/HRNet32.yml' DATASETS.NAMES "('dukemtmc')" MODEL.NAME "('HRNet32')" \
MODEL.DEVICE_ID "('1')" CLUSTERING.PART_NUM "(4)" \
MODEL.SEG_F "('part_ts')" TEST.WITH_ARM "(False)" \
TEST.WEIGHT "('/mnt/Data/dsg/ISP/log/HCNet-Duke-4/HRNet32_model_120.pth')"

# /mnt/Data/dsg/ISP/log/HCNet-Duke-4/HRNet32_model_120.pth
# /mnt/Data/dsg/ISP/log/Best-Duke-4-Rank189.8-HRNet32/HRNet32_model_120.pth