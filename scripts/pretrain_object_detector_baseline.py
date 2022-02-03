import sys
import os
import numpy as np
import torch
import logging

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, inference_on_dataset, print_csv_format, inference_context
from detectron2.checkpoint import DetectionCheckpointer

from segmentationsg.engine import ObjectDetectorTrainer
from segmentationsg.data import add_dataset_config, VisualGenomeTrainData, register_datasets


parser = default_argument_parser()

def setup(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_datasets(cfg)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = ObjectDetectorTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = ObjectDetectorTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = ObjectDetectorTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        # use the last 4 numbers in the job id as the id
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        # all ports should be in the 10k+ range
        default_port = int(default_port) + 15000

    except Exception:
        default_port = 59482
    
    args.dist_url = 'tcp://127.0.0.1:'+str(default_port)
    print (args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
