python3 tools/train.py --config_file='configs/HRNet32.yml' MODEL.DEVICE_ID "('0')" MODEL.NAME "('HRNet32')" \
DATASETS.NAMES "('occluded_reid')" CLUSTERING.PART_NUM "(4)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pseudo_labels-OREID-4')" OUTPUT_DIR "('./log/HCGA-OREID-4')" \
MODEL.IF_WITH_CENTER "('no')" DFC.PIXEL_WEIGHT "(2.0)" DFC.IMAGE_WEIGHT "(1.5)" DFC.DIAG_WEIGHT "(0.0)" \
TEST.WITH_ARM "(True)" CLUSTERING.STOP "(101)" MODEL.SEG_F "('part_t')" VIS.ENV "('OREID')" \
SOLVER.PARSING_LOSS_WEIGHT "(0.3)"