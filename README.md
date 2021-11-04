# Segmentation-Grounded Scene Graph Generation

This repository contains the code for the CVPR 2021 paper titled [**"Segmentation-Grounded Scene Graph Generation"**](https://arxiv.org/pdf/2104.14207.pdf).

## Bibtext
```
@inproceedings{khandelwal2021segmentation,
  title={Segmentation-grounded Scene Graph Generation},
  author={Khandelwal, Siddhesh and Suhail, Mohammed and Sigal, Leonid},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```

## Requirements
To setup the environment with all the required dependencies, follow the steps detailed in [INSTALL.md](https://github.com/ubc-vision/UniT/blob/main/INSTALL.md). Additionally, please rename the cloned repository from `segmentation-sg` to `segmentationsg`.

## Prepare Dataset
The approach requires access to Visual Genome and MS-COCO datasets. 
- MS-COCO is publicly available [here](https://cocodataset.org/#download). We use the 2017 Train/Val splits in our experiments.
- We use the Visual Genome filtered data widely used in the Scene Graph community. Please see the [Unbiased Scene Graph Generation repo](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md) on instructions to download this dataset.

## Pretrain Object Detector
Before the scene graph models can be trained, the first step involves jointly pre-training the object detector to accurately predict bounding boxes on Visual Genome and segmentation masks on MS-COCO. 

If using the ResNeXt-101 backbone, the pre-training can be achieved by running the following command
```python
python pretrain_object_detector_withcoco.py --config-file ../configs/pretrain_object_detector_coco.yaml --num-gpus 4 --resume DATASETS.VISUAL_GENOME.IMAGES <PATH TO VG_100K IMAGES> DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY <PATH TO VG-SGG-dicts-with-attri.json> DATASETS.VISUAL_GENOME.IMAGE_DATA <PATH TO image_data.json> DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 <PATH TO VG-SGG-with-attri.h5> DATASETS.MSCOCO.ANNOTATIONS <PATH TO MSCOCO ANNOTATIONS> DATASETS.MSCOCO.DATAROOT <PATH TO MSCOCO IMAGES> OUTPUT_DIR <PATH TO CHECKPOINT DIR>
```

If using the VGG-16 backbone, the pre-training can be achieved by running the following command
```python
python pretrain_object_detector_withcoco.py  --config-file ../configs/pretrain_object_detector_vgg_coco.yaml --num-gpus 4 --resume DATASETS.VISUAL_GENOME.IMAGES <PATH TO VG_100K IMAGES> DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY <PATH TO VG-SGG-dicts-with-attri.json> DATASETS.VISUAL_GENOME.IMAGE_DATA <PATH TO image_data.json> DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 <PATH TO VG-SGG-with-attri.h5> DATASETS.MSCOCO.ANNOTATIONS <PATH TO MSCOCO ANNOTATIONS> DATASETS.MSCOCO.DATAROOT <PATH TO MSCOCO IMAGES> OUTPUT_DIR <PATH TO CHECKPOINT DIR>
```

The jointly trained pre-trained weights can be found [here](https://drive.google.com/drive/folders/1YZ3ipSi_ao_Xl9UsMBbmro7sp2mi8bqr?usp=sharing).

## Train Scene Graph Model
Once the object detector pre-training is complete, prepare the pre-training weights to be used with scene graph training. Run the following script to achieve this
```python
import torch
pretrain_model = torch.load('<Path to Pretrained Model Weights (example: model_final.pth)>')
pretrain_weight = {}
pretrain_weight['model'] = pretrain_model['model']
with open('<Weight Save Path (example: model_weights.pth)>', 'wb') as f:
    torch.save(pretrain_weight, f)

```

Depending on the task, the scene graph training can then be run as follows. The training scripts are available in the `scripts` folder.
* Predicate Classification (PredCls)

```python
python train_SG_segmentation_head.py --config-file ../configs/sg_dev_masktransfer.yaml --num-gpus 4 --resume DATALOADER.NUM_WORKERS 2 \
MODEL.WEIGHTS <PATH TO PRETRAINED WEIGHTS> \
OUTPUT_DIR <PATH TO CHECKPOINT DIR> \
DATASETS.VISUAL_GENOME.IMAGES <PATH TO VG_100K IMAGES> DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY <PATH TO VG-SGG-dicts-with-attri.json> DATASETS.VISUAL_GENOME.IMAGE_DATA <PATH TO image_data.json> DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 <PATH TO VG-SGG-with-attri.h5> \
DATASETS.MSCOCO.ANNOTATIONS <PATH TO MSCOCO ANNOTATIONS> DATASETS.MSCOCO.DATAROOT <PATH TO MSCOCO IMAGES> \
MODEL.MASK_ON True  \
MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX True MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION True MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE 'Weighted' \
MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION True TEST.EVAL_PERIOD 100000 \
MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS False \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK True \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK False SOLVER.IMS_PER_BATCH 16 DATASETS.SEG_DATA_DIVISOR 2 \
MODEL.ROI_SCENEGRAPH_HEAD.PREDICTOR 'MotifSegmentationPredictorC' MODEL.ROI_HEADS.REFINE_SEG_MASKS False
```

- SceneGraph Classification (SGCls)
```python
python train_SG_segmentation_head.py --config-file ../configs/sg_dev_masktransfer.yaml --num-gpus 4 --resume DATALOADER.NUM_WORKERS 2 \
MODEL.WEIGHTS <PATH TO PRETRAINED WEIGHTS> \    
OUTPUT_DIR <PATH TO CHECKPOINT DIR> \
DATASETS.VISUAL_GENOME.IMAGES <PATH TO VG_100K IMAGES> DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY <PATH TO VG-SGG-dicts-with-attri.json> DATASETS.VISUAL_GENOME.IMAGE_DATA <PATH TO image_data.json> DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 <PATH TO VG-SGG-with-attri.h5> \
DATASETS.MSCOCO.ANNOTATIONS <PATH TO MSCOCO ANNOTATIONS> DATASETS.MSCOCO.DATAROOT <PATH TO MSCOCO IMAGES> \
MODEL.MASK_ON True  \
MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX True MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION True MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE 'Weighted' \
MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION True TEST.EVAL_PERIOD 100000 \
MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS False \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK True \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK False SOLVER.IMS_PER_BATCH 16 DATASETS.SEG_DATA_DIVISOR 2 \
MODEL.ROI_SCENEGRAPH_HEAD.PREDICTOR 'MotifSegmentationPredictorC' MODEL.ROI_HEADS.REFINE_SEG_MASKS False
```

- SceneGraph Prediction (SGPred)
```python
python train_SG_segmentation_head.py --config-file ../configs/sg_dev_masktransfer.yaml --num-gpus 4 --resume DATALOADER.NUM_WORKERS 2 \
MODEL.WEIGHTS <PATH TO PRETRAINED WEIGHTS> \
OUTPUT_DIR <PATH TO CHECKPOINT DIR> \
DATASETS.VISUAL_GENOME.IMAGES <PATH TO VG_100K IMAGES> DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY <PATH TO VG-SGG-dicts-with-attri.json> DATASETS.VISUAL_GENOME.IMAGE_DATA <PATH TO image_data.json> DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 <PATH TO VG-SGG-with-attri.h5> \
DATASETS.MSCOCO.ANNOTATIONS <PATH TO MSCOCO ANNOTATIONS> DATASETS.MSCOCO.DATAROOT <PATH TO MSCOCO IMAGES> \
MODEL.MASK_ON True  \
MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX False MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL False \
MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION True MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE 'Weighted' \
MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION True TEST.EVAL_PERIOD 100000 \
MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS False \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK True \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK False SOLVER.IMS_PER_BATCH 16 DATASETS.SEG_DATA_DIVISOR 2 \
MODEL.ROI_SCENEGRAPH_HEAD.PREDICTOR 'MotifSegmentationPredictorC' MODEL.ROI_HEADS.REFINE_SEG_MASKS False TEST.DETECTIONS_PER_IMAGE 40
```

Note that these commands augment our approach to Neural Motifs with ResNeXt 101 backbone. To use VCTree, use 
```python
MODEL.ROI_SCENEGRAPH_HEAD.PREDICTOR 'VCTreeSegmentationPredictorC'
```
To use VGG-16 backbone, use
```python
--config-file ../configs/sg_dev_masktransfer_vgg.yaml
```

## Evaluation

Evaluation can be done using the `--eval-only` flag. For example, evaluation can be run on the PredCLS model as follows,
```python
python train_SG_segmentation_head.py --eval-only --config-file ../configs/sg_dev_masktransfer.yaml --num-gpus 4 --resume DATALOADER.NUM_WORKERS 2 \
MODEL.WEIGHTS <PATH TO PRETRAINED WEIGHTS> \
OUTPUT_DIR <PATH TO CHECKPOINT DIR> \
DATASETS.VISUAL_GENOME.IMAGES <PATH TO VG_100K IMAGES> DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY <PATH TO VG-SGG-dicts-with-attri.json> DATASETS.VISUAL_GENOME.IMAGE_DATA <PATH TO image_data.json> DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 <PATH TO VG-SGG-with-attri.h5> \
DATASETS.MSCOCO.ANNOTATIONS <PATH TO MSCOCO ANNOTATIONS> DATASETS.MSCOCO.DATAROOT <PATH TO MSCOCO IMAGES> \
MODEL.MASK_ON True  \
MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX True MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_SCENEGRAPH_HEAD.USE_MASK_ATTENTION True MODEL.ROI_SCENEGRAPH_HEAD.MASK_ATTENTION_TYPE 'Weighted' \
MODEL.ROI_SCENEGRAPH_HEAD.SIGMOID_ATTENTION True TEST.EVAL_PERIOD 100000 \
MODEL.ROI_RELATION_FEATURE_EXTRACTORS.MULTIPLY_LOGITS_WITH_MASKS False \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.BOX_FEATURE_MASK True \
MODEL.ROI_BOX_FEATURE_EXTRACTORS.CLASS_LOGITS_WITH_MASK False SOLVER.IMS_PER_BATCH 16 DATASETS.SEG_DATA_DIVISOR 2 \
MODEL.ROI_SCENEGRAPH_HEAD.PREDICTOR 'MotifSegmentationPredictorC' MODEL.ROI_HEADS.REFINE_SEG_MASKS False
```

**Note**: The default training/testing assumes 4 GPUs. It can be modified to suit other GPU configurations, but would require changing the learning rate and batch sizes accordingly. Please look at `SOLVER.REFERENCE_WORLD_SIZE` parameter in the [detectron2 configurations](https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references) for details on how this can be done automatically.

