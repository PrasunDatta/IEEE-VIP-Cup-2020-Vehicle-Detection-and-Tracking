import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from .base import BaseDetector
import pdb

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if hasattr(self.backbone, 'num_dn_classes'):
          [dn,x] = self.backbone(img)
          #print('Dhukse')
          #pdb.set_trace()
          if self.with_neck:
              x = self.neck(x)
          #pdb.set_trace()
          return dn, x
        else:
          x = self.backbone(img)
          #pdb.set_trace()
          if self.with_neck:
              x = self.neck(x)
          #pdb.set_trace()
          return x


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if hasattr(self.backbone, 'num_dn_classes'):
          [dn, x] = self.extract_feat(img)
          gt_labels_dn = torch.empty(8)
          #pdb.set_trace()
          for i in range(len(img_metas)):
            img_path = img_metas[i]['filename']
            img_path = img_path[43:]
            if img_path.startswith('CLIP'):
              gt_labels_dn[i] = torch.tensor([1], dtype=torch.long, device='cuda')
            else:
              gt_labels_dn[i] = torch.tensor([0], dtype=torch.long, device='cuda')
          gt_labels_dn = gt_labels_dn.cuda()

          #print(img_path)
          #pdb.set_trace()
          '''
          night_output = torch.exp(dn[:,0])/torch.sum(torch.exp(dn), dim=1)
          day_output = torch.exp(dn[:,1])/torch.sum(torch.exp(dn), dim=1)
          night_op = gt_labels_dn*torch.log(night_output+1e-20)
          day_op = (1-gt_labels_dn)*torch.log(day_output+1e-20)
          loss_dn_cls = -torch.sum(day_op + night_op)/8
          if loss_dn_cls==0:
            pdb.set_trace()
          print(loss_dn_cls)
          #pdb.set_trace()
          loss_dn = 1/loss_dn_cls

          if len(torch.unique(gt_labels_dn))==1:
            if gt_labels_dn[0]==1:
              pdb.set_trace() #represents all night images in gnd truth
          '''
          loss_dn_cls=nn.CrossEntropyLoss()
          loss_dn = loss_dn_cls(
              dn, gt_labels_dn.long(),
              )
          
          '''
          if loss_dn==0:
            pdb.set_trace()
          '''
          lamda = 0.1
          #print(loss_dn)
          loss_dn = 1/(loss_dn+0.2) #0.2 for handling loss zero case
          #pdb.set_trace()
          losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
          #print(loss_dn)
          losses['loss_dn_cls'] = loss_dn*lamda
          #print(losses['loss_dn_cls'])
          #print(losses)
          #pdb.set_trace()
        else:
          x = self.extract_feat(img)
          losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
          #pdb.set_trace()

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
