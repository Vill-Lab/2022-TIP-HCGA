python3 tools/train.py --config_file='configs/HRNet32.yml' MODEL.DEVICE_ID "('0')"  MODEL.NAME "('HRNet32')" \
DATASETS.NAMES "('dukemtmc')"  CLUSTERING.PART_NUM "(4)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pseudo_labels-Duke-HCNet-4')"  \
OUTPUT_DIR "('./log/HCNet-Duke-4')" MODEL.IF_WITH_CENTER "('on')" MODEL.SEG_F "('part_t')" VIS.ENV "('Duke')" \
DFC.PIXEL_WEIGHT "(2.0)" DFC.IMAGE_WEIGHT "(1.0)" DFC.DIAG_WEIGHT "(0.0)" DFC.C_LOSS "('l2')" \
TEST.WITH_ARM "(True)" SOLVER.PARSING_LOSS_WEIGHT "(0.3)"