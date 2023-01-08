python3 tools/train.py --config_file='configs/HRNet32.yml' MODEL.DEVICE_ID "('0')" \
MODEL.IF_WITH_CENTER "('no')" MODEL.NAME "('HRNet32')" MODEL.IF_BIGG "(True)" DATASETS.NAMES "('cuhk03_np_detected')" \
DATASETS.ROOT_DIR "('/mnt/Data-HDD/dsg')" CLUSTERING.PART_NUM "(6)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pseudo_labels-HCNet-6')"  \
OUTPUT_DIR "('./log/HCNet-cuhk03-detected-6')" DFC.PIXEL_WEIGHT "(1.0)" DFC.IMAGE_WEIGHT "(0.5)" DFC.DIAG_WEIGHT "(0.0)" DFC.C_LOSS "('l1  ')" \
CLUSTERING.STOP "(101)" DFC.MODE "('mp')"