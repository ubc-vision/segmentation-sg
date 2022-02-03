import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import COCOEvaluator

class COCOEvaluatorWeakSegmentation(COCOEvaluator):
    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        # if cfg.MODEL.MASK_ON:
        #     tasks = tasks + ("segm",)
        # if cfg.MODEL.KEYPOINT_ON:
        #     tasks = tasks + ("keypoints",)
        return tasks