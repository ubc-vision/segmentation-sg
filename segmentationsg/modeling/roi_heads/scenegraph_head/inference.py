import torch
import torch.nn.functional as F
from torch import nn

from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from .utils import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
    ):
        """
        Arguments:
        """
        super(PostProcessor, self).__init__()
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres

    def forward(self, x, rel_pair_idxs, boxes, img_sizes, segmentation_vis=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image
        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box, img_size) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes, img_sizes
        )):
            
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, -1] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, :-1].max(dim=1)
            else:
                # apply late nms for object prediction
                obj_pred = obj_prediction_nms(box.boxes_per_cls, obj_logit, self.later_nms_pred_thres)
                obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                obj_scores = obj_class_prob.view(-1)[obj_score_ind]
            
            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            result = Instances(img_size)
            
            if self.use_gt_box:
                result.pred_boxes = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                #FIXME
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                result.pred_boxes = Boxes(box.boxes_per_cls[torch.arange(batch_size, device=device), regressed_box_idxs])

            result.pred_classes = obj_class
            result.scores = obj_scores
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, :-1].max(dim=1)

            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            result._rel_pair_idxs = rel_pair_idx # (#rel, 2)
            result._pred_rel_scores = rel_class_prob # (#rel, #rel_class)
            result._pred_rel_labels = rel_labels # (#rel, )
            if segmentation_vis:
                result._sorting_idx = sorting_idx
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            results.append(result)
        return results


def build_roi_scenegraph_post_processor(cfg):

    use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    postprocessor = PostProcessor(
        use_gt_box,
        later_nms_pred_thres,
    )
    return postprocessor