from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import pdb
import copy
import numpy


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)

        #######Need to write logic for size here and store it in gt_labels_sml
        ##gt_labels sml will be a list that will have length 8 for 8 images
        ##any image will have N number of labels depending on the number of vehicles in that image
        ##each vehicle can have any class between 1,2,3
        #pdb.set_trace()

        if hasattr(self, 'num_sml_classes'):
          #print('Dhukse')
          gt_labels_sml = copy.deepcopy(gt_labels)
          #pdb.set_trace()
          #print(gt_bboxes[0][0])
          #pdb.set_trace()
          for i in range(len(gt_bboxes)):
              for j in range(len(gt_bboxes[i])):
                      boxwidth = gt_bboxes[i][j][2] - gt_bboxes[i][j][0]
                      boxheight = gt_bboxes[i][j][3] - gt_bboxes[i][j][1]
                      gt_labels_sml[i][j] = checksize(boxwidth, boxheight) 
                    
        
        '''
        gt_labels_dn = copy.deepcopy(gt_labels)
        pdb.set_trace()
        for i in range(len(gt_labels_dn)):
            for j in range(len(gt_labels_dn[i])):
                print('a')
        #pdb.set_trace()
        '''

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
            
        else:
          if hasattr(self, 'num_sml_classes'):
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_labels_sml, img_metas)
          else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)

            #loss_inputs = outs + (gt_bboxes, gt_labels, gt_labels_sml, img_metas)
        
        #pdb.set_trace()
        ##Need to change the loss function inside gfl_head
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        #pdb.set_trace()
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list


def checksize(boxwidth, boxheight):
    import numpy 

    boxwidth = boxwidth.cpu().detach().numpy()
    boxheight = boxheight.cpu().detach().numpy()

    if boxwidth*boxheight<numpy.array([32*32]):
        return torch.tensor([0], device='cuda')
    elif boxwidth*boxheight>numpy.array([96*96]):
        return torch.tensor([2], device='cuda')
    else:
        return torch.tensor([1], device='cuda')


