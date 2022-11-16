# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:46:53 2020

@author: HED
"""

#%%

from .gfl_head import GFLHead
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
import pdb
import numpy

#%% 
def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

#%% 
@HEADS.register_module()
class GFLSMLBBoxHead(GFLHead):


    def __init__(self,
                 num_sml_classes=3,
                 loss_sml_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 **kwargs):
        
        super(GFLSMLBBoxHead, self).__init__(**kwargs)

        self.num_sml_classes = num_sml_classes
        self.loss_sml_cls = build_loss(loss_sml_cls) 
        
        self.cls_sml_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_sml_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            
        self.gfl_sml_cls = nn.Conv2d(
            self.feat_channels, self.num_sml_classes, 3, padding=1) 

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_sml_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)
        normal_init(self.gfl_sml_cls, std=0.01, bias=bias_cls)

        
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        cls_feat = x
        cls_sml_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for cls_sml_conv in self.cls_sml_convs:
            cls_sml_feat = cls_sml_conv(cls_sml_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            
        cls_score = self.gfl_cls(cls_feat)
        cls_sml_score = self.gfl_sml_cls(cls_sml_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, cls_sml_score, bbox_pred

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, cls_sml_score, labels, labels_sml, label_weights,
                    label_sml_weights, bbox_targets, stride, num_total_samples):

        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        #pdb.set_trace()
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        
        #pdb.set_trace()
        cls_sml_score = cls_sml_score.permute(0, 2, 3,
                                      1).reshape(-1, self.num_sml_classes)
        
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        labels_sml = labels_sml.reshape(-1)
        label_sml_weights = label_sml_weights.reshape(-1)
        '''
        print('Back in Loss function')
        print('Size of labels_sml ', labels_sml.shape)
        print('Unique Elements in labels_sml')
        print(numpy.unique(labels_sml.cpu().detach().numpy(), return_counts=True))
        print('Size of bbox_targets ', bbox_targets.shape)
        print('\n\n\n')
        '''
        #print(numpy.unique(label_sml_weights.cpu().detach().numpy(), return_counts=True))
        #pdb.set_trace()

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        #pdb.set_trace()

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            #pdb.set_trace()

            pos_sml_targets = labels_sml[pos_inds]
            pos_sml_pred = cls_sml_score[pos_inds]

            #pdb.set_trace()

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            loss_sml_cls = self.loss_sml_cls(
            pos_sml_pred, pos_sml_targets,
            #label_sml_weights,
            #avg_factor=avg_factor,
            reduction_override=None
            )
            lamda_sml = 0.1
            #print(loss_sml_cls)
            loss_sml_cls = loss_sml_cls*lamda_sml
            #print(loss_sml_cls)
            #pdb.set_trace()

            #pdb.set_trace()

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_sml_cls = 0
            weight_targets = torch.tensor(0).cuda()

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)
        #pdb.set_trace()

        # sml cls loss
        #avg_factor = max(torch.sum(label_sml_weights > 0).float().item(), 1.)
        #criterion = nn.CrossEntropyLoss()
        #loss_sml_cls = criterion(cls_sml_score, labels_sml)
        
        #loss_sml_cls = self.loss_sml_cls(
        #    cls_sml_score, labels_sml,
            #label_sml_weights,
            #avg_factor=avg_factor,
        #    reduction_override=None
        #    )
        #pdb.set_trace()
        return loss_cls, loss_bbox, loss_dfl, loss_sml_cls, weight_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_sml_score,
             gt_bboxes,
             gt_labels,
             gt_labels_sml,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        #print('Dhukse 1')
        #pdb.set_trace()
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        #pdb.set_trace()
        '''
        print('Now in Loss Function')
        print('In loss function, Length of gt_labels_sml is ', len(gt_labels_sml), ',One per image in a batch')
        print('One example for an image ', gt_labels_sml[6])
        print('\n\n\n')
        '''

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            gt_labels_sml_list = gt_labels_sml,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, labels_sml_list, label_sml_weights_list,
         bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        #pdb.set_trace()

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, losses_sml_cls,\
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                cls_sml_score,
                bbox_preds,
                labels_list,
                labels_sml_list,
                label_weights_list,
                label_sml_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_sml_cls = losses_sml_cls)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, anchors in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    gt_labels_sml_list = None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        #pdb.set_trace()
        '''
        print('Now in get_targets Function')
        print('In get_targets function, Length of gt_labels_sml_list is ', len(gt_labels_sml_list), ',One per image in a batch')
        print('One example for an image ', gt_labels_sml_list[6])
        print('\n\n\n')
        '''
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
            gt_labels_sml_list = [None for _ in range(num_imgs)]

        (all_anchors, all_labels, all_label_weights, all_labels_sml, all_label_sml_weights,
         all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             gt_labels_sml_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        labels_sml_list = images_to_levels(all_labels_sml, num_level_anchors)

        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        label_sml_weights_list = images_to_levels(all_label_sml_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list, labels_sml_list,
        label_sml_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           gt_labels_sml,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        '''
        print('Now in get_targets_single Function')
        print('In get_targets_single function, Length of gt_labels_sml is ', len(gt_labels_sml))
        #print('One example for an image ', gt_labels_sml[6])
        print('\n\n\n')
        '''
        #pdb.set_trace()
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        labels_sml = anchors.new_full((num_valid_anchors, ),
                                  self.num_sml_classes,
                                  dtype=torch.long)
        
        label_sml_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        '''
        print('Here the labels is filled with num_classes because that is the bg class')
        print('Bbox targets is filled with zeros completely')
        print('Unique elements of labels: ')
        print(numpy.unique(labels.cpu().detach().numpy(), return_counts=True))
        print('Unique elements of labels_sml: ')
        print(numpy.unique(labels_sml.cpu().detach().numpy(), return_counts=True))
        #print('First 50 elements of bbox_targets: ')
        #print(bbox_targets[0:49,:])
        '''

        #pdb.set_trace()

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        '''

        print('Number of positive examples: ', len(pos_inds))
        print('Number of negative examples: ', len(neg_inds))
        print('Number of labels: ', len(labels_sml))
        '''

        #pdb.set_trace()
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            #print('After assigning labels to positive examples, ')
            #print('First 50 elements of bbox_targets: ')
            #print(bbox_targets[0:49,:])
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
                labels_sml[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
                labels_sml[pos_inds] = gt_labels_sml[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
                label_sml_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
                label_sml_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
            label_sml_weights[neg_inds] = 1.0
            '''
        print('After assigning labels to positive examples, ')
        print('Unique elements of labels: ')
        print(numpy.unique(labels.cpu().detach().numpy(), return_counts=True))
        print('Unique elements of labels_sml: ')
        print(numpy.unique(labels_sml.cpu().detach().numpy(), return_counts=True))
        print('First 50 elements of bbox_targets: ')
        print(bbox_targets[0:500,:])
        '''
        #pdb.set_trace()

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            labels_sml = unmap(
                labels_sml, num_total_anchors, inside_flags, fill=self.num_sml_classes)
            label_sml_weights = unmap(label_sml_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        '''
        print('Unique elements of labels: ')
        print(numpy.unique(labels.cpu().detach().numpy(), return_counts=True))
        print('Unique elements of labels_sml: ')
        print(numpy.unique(labels_sml.cpu().detach().numpy(), return_counts=True))
        '''
        #pdb.set_trace()

        return (anchors, labels, label_weights, labels_sml, label_sml_weights,
         bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

        
        
        
        