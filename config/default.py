from yacs.config import CfgNode as CN

cfg = CN()


# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
cfg.TRAIN = CN()

cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.LR = 0.01
cfg.TRAIN.EPOCHS = 1
cfg.TRAIN.SNAPSHOT_BEST = ""

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
cfg.BACKBONE = CN()
cfg.BACKBONE.NAME = ''
cfg.BACKBONE.PRETRAINED = ''