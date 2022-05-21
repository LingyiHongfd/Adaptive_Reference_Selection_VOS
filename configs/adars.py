import torch
import argparse
import os
import sys
import cv2
import time


class Configuration():
    def __init__(self):
        self.EXP_NAME = 'ada_rs'

        self.DIR_ROOT = './'
        self.DIR_DATA = os.path.join(self.DIR_ROOT, 'datasets')
        self.DIR_DAVIS = os.path.join(self.DIR_DATA, 'DAVIS')
        self.DIR_RESULT = os.path.join(self.DIR_ROOT, 'result', self.EXP_NAME)
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')

        self.MODEL_BACKBONE = 'resnet'
        self.MODEL_MODULE = 'networks.adars.adars'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SEMANTIC_EMBEDDING_DIM = 100
        self.MODEL_HEAD_EMBEDDING_DIM = 256
        self.MODEL_PRE_HEAD_EMBEDDING_DIM = 64
        self.MODEL_GN_GROUPS = 32
        self.MODEL_GN_EMB_GROUPS = 25
        self.MODEL_MULTI_LOCAL_DISTANCE = [2, 4, 6, 8, 10, 12]
        self.MODEL_LOCAL_DOWNSAMPLE = True
        self.MODEL_REFINE_CHANNELS = 64  # n * 32
        self.MODEL_LOW_LEVEL_INPLANES = 256 if self.MODEL_BACKBONE == 'resnet' else 24
        self.MODEL_RELATED_CHANNELS = 64
        self.MODEL_EPSILON = 1e-5
        self.MODEL_MATCHING_BACKGROUND = True
        self.MODEL_GCT_BETA_WD = True
        self.MODEL_FLOAT16_MATCHING = False
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False

        self.TEST_GPU_ID = 1
        self.TEST_DATASET = 'davis2017'
        self.TEST_DATASET_FULL_RESOLUTION = False
        self.TEST_DATASET_SPLIT = ['val']
        self.TEST_CKPT_PATH = 'pretrain_models/adars.pth'
        self.TEST_CKPT_STEP = None  # if "None", evaluate the latest checkpoint.
        self.TEST_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MIN_SIZE = None
        self.TEST_MAX_SIZE = 800 * 1.3 if self.TEST_MULTISCALE == [1.] else 800
        self.TEST_WORKERS = 1
        self.TEST_GLOBAL_CHUNKS = 4
        self.TEST_GLOBAL_ATROUS_RATE = 1
        self.TEST_LOCAL_ATROUS_RATE = 1
        self.TEST_LOCAL_PARALLEL = True

        # dist
        self.DIST_ENABLE = False
        self.DIST_BACKEND = "nccl"
        self.DIST_URL = "tcp://127.0.0.1:13566"
        self.DIST_START_GPU = 0

        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
            raise ValueError('config.py: the number of GPU is 0')
        for path in [self.DIR_RESULT, self.DIR_CKPT, self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG,
                     self.DIR_TB_LOG]:
            if not os.path.isdir(path):
                os.makedirs(path)


cfg = Configuration()
