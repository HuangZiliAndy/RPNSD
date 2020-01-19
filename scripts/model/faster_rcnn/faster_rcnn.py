import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.n_classes = classes
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, feat, gt_boxes, num_boxes, stage):
        # feat: acoustic features (we use STFT) [batch_size, seq_len, feat_dim], default [8, 1000, 257]
        # gt_boxes: ground truth speech segments, the last dimension is (start_frame, end_frame, speaker index) 
        # [batch_size, padded_len, 3], default [8, 20, 3]
        # num_boxes: number of speech segments in each audio [batch_size], default [8]
        # stage: specify the stage (can be train, dev or test)
        batch_size, seq_len, feat_dim = feat.size(0), feat.size(1), feat.size(2)

        feat = torch.unsqueeze(feat, 1)
        feat = torch.transpose(feat, 2, 3)
        im_info = torch.from_numpy(np.array([[feat_dim, seq_len]]))
        im_info = im_info.expand(batch_size, im_info.size(1))

        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # base_feat: deep features after backbone (ResNet101)
        # [batch_size, num_channels, h, w], default [8, 1024, 16, 63]
        base_feat = self.RCNN_base(feat)

        # feed base feature map to RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, stage)
        # rois: region of interest(ROI), selected speech segment segments
        # The last dimension is (batch_idx, start_t, end_t)
        # [batch_size, number of rois, 3] default: [8, 100, 3]

        # if it is training phrase, then use ground truth bboxes for refining
        if stage == "train" or stage == "dev":
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # rois: selected ROIs to compute loss, the last dimension is (batch_idx, start_t, end_t)
            # [batch_size, number of rois, 3], default [8, 64, 3]

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        elif stage == "test":
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        else:
            raise ValueError("Condition not defined.")

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        rois_tmp = rois.new(rois.size(0), rois.size(1), 5).zero_()
        rois_tmp[:, :, np.array([0, 1, 3]).astype(int)] = rois
        rois_tmp[:, :, 4] = feat_dim - 1 

        # default is 'align'
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois_tmp.view(-1,5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois_tmp.view(-1,5))
        else:
            raise ValueError("Pooling mode not supported.")
        # pooled_feat: the pooled feature for speech segments
        # [batch_size * number of rois, number of channels, 7, 7], default [512, 1024, 7, 7]

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        # compute object classification probability
        bg_cls_score = self.RCNN_bg_cls_score(pooled_feat)
        bg_cls_prob = F.softmax(bg_cls_score, 1)
        seg_embed = self.RCNN_embed(pooled_feat)
        cls_score = self.RCNN_cls_score(F.relu(seg_embed))
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_bg_cls = 0
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_cls_spk = 0

        if stage == "train" or stage == "dev":
            # RCNN_loss_cls is the loss to classify fg/bg 
            rois_bg_label = (rois_label > 0).long()
            RCNN_loss_cls = F.cross_entropy(bg_cls_score, rois_bg_label)
            cls_score_nonzero, rois_label_nonzero = cls_score[rois_label != 0, :], rois_label[rois_label != 0]

            # RCNN_loss_cls_spk is the loss to classify different speakers
            RCNN_loss_cls_spk = F.cross_entropy(cls_score_nonzero, rois_label_nonzero)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        bg_cls_prob = bg_cls_prob.view(batch_size, rois.size(1), -1)
        return rois, bg_cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_cls_spk, RCNN_loss_bbox, rois_label, seg_embed

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
