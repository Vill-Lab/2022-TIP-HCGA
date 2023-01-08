python3 tools/train.py --config_file='configs/HRNet32.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('HRNet32')" DATASETS.NAMES "('market1501')" \
CLUSTERING.PART_NUM "(6)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pseudo_labels-market-TSCL-4')" OUTPUT_DIR "('./log/HCNet-market-4')" \
DFC.PIXEL_WEIGHT "(1.0)" DFC.IMAGE_WEIGHT "(1.0)" DFC.DIAG_WEIGHT "(1.0)" DFC.C_LOSS "('l2')" MODEL.IF_WITH_CENTER "('on')" \
MODEL.SEG_F "('part_t')" VIS.ENV "('Market1501')" TEST.WITH_ARM "(True)"