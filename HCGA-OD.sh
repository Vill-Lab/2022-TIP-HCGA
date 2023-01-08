python3 tools/train.py --config_file='configs/HRNet32.yml' DATASETS.NAMES "('occluded_dukemtmc')" MODEL.NAME "('HRNet32')" \
MODEL.DEVICE_ID "('0')" MODEL.IF_WITH_CENTER "('no')" \
CLUSTERING.PART_NUM "(4)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pl-HRNet32-OD-0')" OUTPUT_DIR "('./log/HCGA-OD-HRNet32')" \
SOLVER.PARSING_LOSS_WEIGHT "(0.1)" TEST.WITH_ARM "(True)" \
DFC.PIXEL_WEIGHT "(2.0)" DFC.IMAGE_WEIGHT "(1.0)" DFC.DIAG_WEIGHT "(1.0)" \
CLUSTERING.STOP "(101)" DFC.C_LOSS "('l1')" MODEL.SEG_F "('adaptive')" VIS.ENV "('OD')" MODEL.T "(4)"