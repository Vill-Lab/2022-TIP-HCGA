python3 tools/test.py --config_file='configs/HRNet32.yml' DATASETS.NAMES "('market1501')" MODEL.NAME "('HRNet32')" \
MODEL.DEVICE_ID "('1')" CLUSTERING.PART_NUM "(6)" \
MODEL.SEG_F "('part_t')" TEST.WITH_ARM "(False)" \
TEST.WEIGHT "('/mnt/Data/dsg/ISP/log/HCNet-market-4/HRNet32_model_120.pth')"

# /mnt/Data/dsg/ISP/log/HCNet-market-4/HRNet32_model_120.pth
# /mnt/Data/dsg/ISP/log/Best-Market-Rank195.6-HRNet32/HRNet32_model_120.pth