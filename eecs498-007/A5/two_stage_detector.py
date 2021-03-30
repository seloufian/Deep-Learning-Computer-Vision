import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt
from single_stage_detector import GenerateAnchor, GenerateProposal, IoU


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")

class ProposalModule(nn.Module):
  def __init__(self, in_dim, hidden_dim=256, num_anchors=9, drop_ratio=0.3):
    super().__init__()

    assert(num_anchors != 0)
    self.num_anchors = num_anchors
    ##############################################################################
    # TODO: Define the region proposal layer - a sequential module with a 3x3    #
    # conv layer, followed by a Dropout (p=drop_ratio), a Leaky ReLU and         #
    # a 1x1 conv.                                                                #
    # HINT: The output should be of shape Bx(Ax6)x7x7, where A=self.num_anchors. #
    #       Determine the padding of the 3x3 conv layer given the output dim.    #
    ##############################################################################
    # Make sure that your region proposal module is called pred_layer
    self.pred_layer = None      
    # Replace "pass" statement with your code

    # Compute the last (2nd) convolution layer output channels: A*6
    last_cv_out = self.num_anchors * 6

    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
      nn.Dropout2d(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(hidden_dim, last_cv_out, kernel_size=1)
    )

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors

  def forward(self, features, pos_anchor_coord=None, \
              pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the proposal module.

    Inputs:
    - features: Tensor of shape (B, in_dim, H', W') giving features from the
      backbone network.
    - pos_anchor_coord: Tensor of shape (M, 4) giving the coordinates of
      positive anchors. Anchors are specified as (x_tl, y_tl, x_br, y_br) with
      the coordinates of the top-left corner (x_tl, y_tl) and bottom-right
      corner (x_br, y_br). During inference this is None.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of positive
      anchors. During inference this is None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indicdes of negative
      anchors. During inference this is None.

    The outputs from this module are different during training and inference.
    
    During training, pos_anchor_coord, pos_anchor_idx, and neg_anchor_idx are
    all provided, and we only output predictions for the positive and negative
    anchors. During inference, these are all None and we must output predictions
    for all anchors.

    Outputs (during training):
    - conf_scores: Tensor of shape (2M, 2) giving the classification scores
      (object vs background) for each of the M positive and M negative anchors.
    - offsets: Tensor of shape (M, 4) giving predicted transforms for the
      M positive anchors.
    - proposals: Tensor of shape (M, 4) giving predicted region proposals for
      the M positive anchors.

    Outputs (during inference):
    - conf_scores: Tensor of shape (B, A, 2, H', W') giving the predicted
      classification scores (object vs background) for all anchors
    - offsets: Tensor of shape (B, A, 4, H', W') giving the predicted transforms
      for all anchors
    """
    if pos_anchor_coord is None or pos_anchor_idx is None or neg_anchor_idx is None:
      mode = 'eval'
    else:
      mode = 'train'
    conf_scores, offsets, proposals = None, None, None
    ############################################################################
    # TODO: Predict classification scores (object vs background) and transforms#
    # for all anchors. During inference, simply output predictions for all     #
    # anchors. During training, extract the predictions for only the positive  #
    # and negative anchors as described above, and also apply the transforms to#
    # the positive anchors to compute the coordinates of the region proposals. #
    #                                                                          #
    # HINT: You can extract information about specific proposals using the     #
    # provided helper function self._extract_anchor_data.                      #
    # HINT: You can compute proposal coordinates using the GenerateProposal    #
    # function from the previous notebook.                                     #
    ############################################################################
    # Replace "pass" statement with your code

    # Compute the output from the prediction network, feeded with 'features'.
    # 'out_anchors' is a 4-D tensor of shape (B, 6*A, H', W')
    out_anchors = self.pred_layer(features)

    # Get some parameter values, needed for next operations.
    B, _, Hp, Wp = out_anchors.shape
    A = self.num_anchors

    # Reshape 'out_anchors' from (B, 6*A, H', W') to (B, A, 6, H', W')
    out_anchors = out_anchors.reshape(B, A, 6, Hp, Wp)

    # Get all confidence scores (situated in the two first positions, for each anchor).
    # 'all_conf_scores' is a 5-D tensor of shape (B, A, 2, H', W')
    all_conf_scores = out_anchors[:, :, :2, ...]

    # Get all offsets (situated in the four last positions, for each anchor).
    # 'all_offsets' is a 5-D tensor of shape (B, A, 4, H', W')
    all_offsets = out_anchors[:, :, 2:, ...]

    if mode == 'train':  # Training mode.
      # Confidence scores are retrieved for both positive and negative anchors.
      pos_anc_conf = self._extract_anchor_data(all_conf_scores, pos_anchor_idx)
      neg_anc_conf = self._extract_anchor_data(all_conf_scores, neg_anchor_idx)
      # Now, 'conf_scores' is a 2-D tensor of shape (2M, 2)
      conf_scores = torch.cat((pos_anc_conf, neg_anc_conf))

      # Offsets are retrieved only for positive anchors. It's shape is (M, 4)
      offsets = self._extract_anchor_data(all_offsets, pos_anchor_idx)

      # Reshape both 'offsets' and 'pos_anchor_coord' to (1, M, 1, 1, 4)
      # This transformation is needed in order to match GenerateProposal's input shape
      pos_anchor_offsets = offsets.reshape(1, -1, 1, 1, 4)
      pos_anchor_coord = pos_anchor_coord.reshape(1, -1, 1, 1, 4)
      # Get positive anchors proposals. Output shape is (M, 4)
      proposals = GenerateProposal(pos_anchor_coord, pos_anchor_offsets, method='FasterRCNN')
      proposals = proposals.squeeze()

    elif mode == 'eval':  # Inference mode.
      # Return needed info "as they are" (i.e. without positive/negative anchor distinction).
      conf_scores = all_conf_scores
      offsets = all_offsets

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    if mode == 'train':
      return conf_scores, offsets, proposals
    elif mode == 'eval':
      return conf_scores, offsets


def ConfScoreRegression(conf_scores, batch_size):
  """
  Binary cross-entropy loss

  Inputs:
  - conf_scores: Predicted confidence scores, of shape (2M, 2). Assume that the
    first M are positive samples, and the last M are negative samples.

  Outputs:
  - conf_score_loss: Torch scalar
  """
  # the target conf_scores for positive samples are ones and negative are zeros
  M = conf_scores.shape[0] // 2
  GT_conf_scores = torch.zeros_like(conf_scores)
  GT_conf_scores[:M, 0] = 1.
  GT_conf_scores[M:, 1] = 1.

  conf_score_loss = nn.functional.binary_cross_entropy_with_logits(conf_scores, GT_conf_scores, \
                                     reduction='sum') * 1. / batch_size
  return conf_score_loss


def BboxRegression(offsets, GT_offsets, batch_size):
  """"
  Use SmoothL1 loss as in Faster R-CNN

  Inputs:
  - offsets: Predicted box offsets, of shape (M, 4)
  - GT_offsets: GT box offsets, of shape (M, 4)
  
  Outputs:
  - bbox_reg_loss: Torch scalar
  """
  bbox_reg_loss = nn.functional.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
  return bbox_reg_loss


class RPN(nn.Module):
  def __init__(self):
    super().__init__()

    # READ ONLY
    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]])
    self.feat_extractor = FeatureExtractor()
    self.prop_module = ProposalModule(1280, num_anchors=self.anchor_list.shape[0])

  def forward(self, images, bboxes, output_mode='loss'):
    """
    Training-time forward pass for the Region Proposal Network.

    Inputs:
    - images: Tensor of shape (B, 3, 224, 224) giving input images
    - bboxes: Tensor of ground-truth bounding boxes, returned from the DataLoader
    - output_mode: One of 'loss' or 'all' that determines what is returned:
      If output_mode is 'loss' then the output is:
      - total_loss: Torch scalar giving the total RPN loss for the minibatch
      If output_mode is 'all' then the output is:
      - total_loss: Torch scalar giving the total RPN loss for the minibatch
      - pos_conf_scores: Tensor of shape (M, 1) giving the object classification
        scores (object vs background) for the positive anchors
      - proposals: Tensor of shape (M, 4) giving the coordiantes of the region
        proposals for the positive anchors
      - features: Tensor of features computed from the backbone network
      - GT_class: Tensor of shape (M,) giving the ground-truth category label
        for the positive anchors.
      - pos_anchor_idx: Tensor of shape (M,) giving indices of positive anchors
      - neg_anchor_idx: Tensor of shape (M,) giving indices of negative anchors
      - anc_per_image: Torch scalar giving the number of anchors per image.
    
    Outputs: See output_mode

    HINT: The function ReferenceOnActivatedAnchors from the previous notebook
    can compute many of these outputs -- you should study it in detail:
    - pos_anchor_idx (also called activated_anc_ind)
    - neg_anchor_idx (also called negative_anc_ind)
    - GT_class
    """
    # weights to multiply to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 5 # for offsets

    assert output_mode in ('loss', 'all'), 'invalid output mode!'
    total_loss = None
    conf_scores, proposals, features, GT_class, pos_anchor_idx, anc_per_img = \
      None, None, None, None, None, None
    ##############################################################################
    # TODO: Implement the forward pass of RPN.                                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, proposals through the region proposal    #
    #     module                                                                 #
    # v) Compute the total_loss for RPN which is formulated as:                  #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss,                     #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression. Note that RPN does not predict any class info.          #
    #    We have written this part for you which you've already practiced earlier#
    # HINT: Do not apply thresholding nor NMS on the proposals during training   #
    #       as positive/negative anchors have been explicitly targeted.          #
    ##############################################################################
    # Replace "pass" statement with your code

    batch_size = images.shape[0]

    # i) Image feature extraction.
    features = self.feat_extractor(images)

    # ii) Grid and anchor generation.
    grid = GenerateGrid(batch_size)
    anchors = GenerateAnchor(self.anchor_list, grid)

    # Compute the number of anchors per image.
    # 'anchors' shape is (B, A, H, W, 4). that is, 'anc_per_img' equals to: A*H*W
    _, A, H, W, _ = anchors.shape
    anc_per_img = A * H * W

    # iii) Determine activated (positive), negative anchors, etc.
    iou_mat = IoU(anchors, bboxes)
    pos_anchor_idx, neg_anchor_idx, _, GT_offsets, GT_class, activated_anc_coord, _ = \
        ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, method='FasterRCNN')

    # iv) Compute conf_scores, offsets and proposals through the RPN (with "train" mode).
    conf_scores, offsets, proposals = self.prop_module(features, activated_anc_coord,
                                                        pos_anchor_idx, neg_anchor_idx)

    # v) Compute the total loss.
    conf_loss = ConfScoreRegression(conf_scores, batch_size)
    reg_loss = BboxRegression(offsets, GT_offsets, batch_size)

    total_loss = w_conf * conf_loss + w_reg * reg_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    if output_mode == 'loss':
      return total_loss
    else:
      return total_loss, conf_scores, proposals, features, GT_class, \
              pos_anchor_idx, neg_anchor_idx, anc_per_img


  def inference(self, images, thresh=0.5, nms_thresh=0.7, mode='RPN'):
    """
    Inference-time forward pass for the Region Proposal Network.

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - thresh: Threshold value on confidence scores. Proposals with a predicted
      object probability above thresh should be kept. HINT: You can convert the
      object score to an object probability using a sigmoid nonlinearity.
    - nms_thresh: IoU threshold for non-maximum suppression
    - mode: One of 'RPN' or 'FasterRCNN' to determine the outputs.

    The region proposal network can output a variable number of region proposals
    per input image. We assume that the input image images[i] gives rise to
    P_i final propsals after thresholding and NMS.

    NOTE: NMS is performed independently per-image!

    Outputs:
    - final_proposals: List of length B, where final_proposals[i] is a Tensor
      of shape (P_i, 4) giving the coordinates of the predicted region proposals
      for the input image images[i].
    - final_conf_probs: List of length B, where final_conf_probs[i] is a
      Tensor of shape (P_i,) giving the predicted object probabilities for each
      predicted region proposal for images[i]. Note that these are
      *probabilities*, not scores, so they should be between 0 and 1.
    - features: Tensor of shape (B, D, H', W') giving the image features
      predicted by the backbone network for each element of images.
      If mode is "RPN" then this is a dummy list of zeros instead.
    """
    assert mode in ('RPN', 'FasterRCNN'), 'invalid inference mode!'

    features, final_conf_probs, final_proposals = None, None, None
    ##############################################################################
    # TODO: Predicting the RPN proposal coordinates `final_proposals` and        #
    # confidence scores `final_conf_probs`.                                     #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Threshold the conf_scores based on the threshold value `thresh`.     #
    # Then, apply NMS to the filtered proposals given the threshold `nms_thresh`.#
    # HINT: Use `torch.no_grad` as context to speed up the computation.          #
    ##############################################################################
    # Replace "pass" statement with your code

    batch_size = images.shape[0]

    with torch.no_grad():  # "no_grad" context used to speed up the computation.
      # Image feature extraction (using the RPN).
      features = self.feat_extractor(images)

      # Grid and anchor generation.
      grid = GenerateGrid(batch_size)
      anchors = GenerateAnchor(self.anchor_list, grid)

      B, A, Hp, Wp, _ = anchors.shape

      # Pass 'features' through the RPN network (with "inference" mode).
      conf_scores, offsets = self.prop_module(features)

    # Reshape 'conf_scores' from (B, A, 2, H', W') to (B, A, H', W', 2)
    conf_scores = torch.transpose(conf_scores, 2, 4)
    # Reshape 'conf_scores' from (B, A, H', W', 2) to (B, A*H'*W', 2)
    conf_scores = torch.flatten(conf_scores, start_dim=1, end_dim=-2)
    # Transform 'conf_scores' into probabilities, by squashing them into (0,1) range.
    conf_probs = torch.sigmoid(conf_scores)
    # Retrieve only object's probability, we don't care about the 'background' prob.
    # Now, 'conf_probs' will have a shape of (B, A*H'*W')
    conf_probs = conf_probs[..., 0]

    # Reshape 'offsets' from (B, A, 4, H', W') to (B, A, H', W', 4)
    offsets = torch.transpose(offsets, 2, 4)

    proposals = GenerateProposal(anchors, offsets, method='FasterRCNN')
    # Reshape 'proposals' from (B, A, H', W', 4) to (B, A*H'*W', 4)
    proposals = torch.flatten(proposals, start_dim=1, end_dim=-2)

    final_conf_probs, final_proposals = [], []

    for idx in range(batch_size):
      # Get current image's proposals and conf_probs.
      cr_proposal = proposals[idx]     # Tensor's shape: (A*H'*W', 4)
      cr_conf_probs = conf_probs[idx]  # Tensor's shape: (A*H'*W',)

      # Define a boolean mask which indicates indexes to delete.
      del_idx_mask = ~ (cr_conf_probs < thresh)

      # Apply the mask on current proposals, conf_probs and class_indices.
      cr_conf_probs = cr_conf_probs[del_idx_mask]
      # Get the number of 'cr_conf_probs' that have been [K]ept.
      K = cr_conf_probs.shape[0]

      # Reshape 'del_idx_mask' from (A*H'*W',) to (A*H'*W', 1)
      del_idx_mask = del_idx_mask.unsqueeze(1)
      # Reshape via broadcasting 'del_idx_mask' from (A*H'*W', 1) to (A*H'*W', 4)
      del_idx_mask = torch.broadcast_to(del_idx_mask, (A*Hp*Wp, 4))

      # 'cr_proposal' will have a shape of (K, 4)
      cr_proposal = cr_proposal[del_idx_mask].reshape(K, 4)

      # Get indices of kept proposals (using NMS).
      icr_proposal = torchvision.ops.nms(cr_proposal, cr_conf_probs, nms_thresh)

      final_conf_probs.append(cr_conf_probs[icr_proposal].unsqueeze(1))
      final_proposals.append(cr_proposal[icr_proposal])

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    if mode == 'RPN':
      features = [torch.zeros_like(i) for i in final_conf_probs] # dummy class
    return final_proposals, final_conf_probs, features


class TwoStageDetector(nn.Module):
  def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
               roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0)
    self.num_classes = num_classes
    self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
    ##############################################################################
    # TODO: Declare your RPN and the region classification layer (in Fast R-CNN).#
    # The region classification layer is a sequential module with a Linear layer,#
    # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
    # Linear layer that predicts classification scores for each proposal.        #
    # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
    # hidden_dim -> num_classes.                                                 #
    ##############################################################################
    # Your RPN and classification layers should be named as follows
    self.rpn = None
    self.cls_layer = None

    # Replace "pass" statement with your code

    self.rpn = RPN()

    self.cls_layer = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.Dropout(p=drop_ratio),
      nn.ReLU(),
      nn.Linear(hidden_dim, num_classes)
    )

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def forward(self, images, bboxes):
    """
    Training-time forward pass for our two-stage Faster R-CNN detector.

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - bboxes: Tensor of shape (B, N, 5) giving ground-truth bounding boxes
      and category labels, from the dataloader.

    Outputs:
    - total_loss: Torch scalar giving the overall training loss.
    """
    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of TwoStageDetector.                      #
    # A few key steps are outlined as follows:                                   #
    # i) RPN, including image feature extraction, grid/anchor/proposal           #
    #       generation, activated and negative anchors determination.            #
    # ii) Perform RoI Align on proposals and meanpool the feature in the spatial #
    #     dimension.                                                             #
    # iii) Pass the RoI feature through the region classification layer which    #
    #      gives the class probilities.                                          #
    # iv) Compute class_prob through the prediction network and compute the      #
    #     cross entropy loss (cls_loss) between the prediction class_prob and    #
    #      the reference GT_class. Hint: Use F.cross_entropy loss.               #
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = rpn_loss + cls_loss.                                       #
    ##############################################################################
    # Replace "pass" statement with your code

    # i) Pass images through the RPN.
    rpn_loss, _, proposals, features, GT_class, pos_anchor_idx, _, anc_per_img = \
                                      self.rpn(images, bboxes, output_mode='all')

    # ii) Perform RoI Align on proposals.
    # RoI Align requires a 2-D tensor (i.e. a matrix), in which columns have 5 elements:
    #   - Image index: indicate to which image (from the batch) the proposal belongs.
    #   - 4 coordinates (xy of top-left and right-bottom).
    # For that, we'll perform a some sort of "bining" of 'pos_anchor_idx' into 'batch_size' bins.
    # 'im_idx' is a 2-D tensor of shape (<len(pos_anchor_idx)>, 1)
    im_idx = (pos_anchor_idx // anc_per_img).unsqueeze(1)
    # Concatenate 'im_idx' to 'proposals' (on the "column" axis, in first place).
    # 'proposals_ibatch' is a 2-D tensor of shape (proposals.shape[0], 5)
    proposals_ibatch = torch.cat((im_idx, proposals), dim=1)
    # Perform RoI Align.
    roi_features = torchvision.ops.roi_align(features, proposals_ibatch,
                              output_size=(self.roi_output_w, self.roi_output_h))

    # Meanpool the feature in the spatial dimesion (i.e. the two last dimensions).
    # Now, 'roi_features' is a 2-D tensor (instead of 4-D before the meanpool).
    roi_features = torch.mean(roi_features, (2, 3))

    # iii) Pass the RoI feature through the region classification layer.
    class_prob = self.cls_layer(roi_features)

    # iv) compute the cross entropy loss between the prediction class_prob and GT_class.
    cls_loss = nn.functional.cross_entropy(class_prob, GT_class)

    # v) Compute the total_loss
    total_loss = rpn_loss + cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return total_loss

  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for our two-stage Faster R-CNN detector

    Inputs:
    - images: Tensor of shape (B, 3, H, W) giving input images
    - thresh: Threshold value on NMS object probabilities
    - nms_thresh: IoU threshold for NMS in the RPN

    We can output a variable number of predicted boxes per input image.
    In particular we assume that the input images[i] gives rise to P_i final
    predicted boxes.

    Outputs:
    - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
      of shape (P_i, 4) giving the coordinates of the final predicted boxes for
      the input images[i]
    - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
      Tensor of shape (P_i,) giving the predicted probabilites that the boxes
      in final_proposals[i] are objects (vs background)
    - final_class: List of length (B,), where final_class[i] is an int64 Tensor
      of shape (P_i,) giving the predicted category labels for each box in
      final_proposals[i].
    """
    final_proposals, final_conf_probs, final_class = None, None, None
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_probs`, and the class index `final_class`.   #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Use the RPN inference function to perform thresholding and NMS, and  #
    # to compute final_proposals and final_conf_probs. Use the predicted class   #
    # probabilities from the second-stage network to compute final_class.        #
    ##############################################################################
    # Replace "pass" statement with your code

    # Pass images through the RPN.
    final_proposals, final_conf_probs, features = self.rpn.inference(images,
                          thresh=thresh, nms_thresh=nms_thresh, mode='FasterRCNN')

    final_class = []

    for idx, cr_proposal in enumerate(final_proposals):
      # Get current image features, and add a unit dimension to match the RoI Align
      # input dimensions.
      cr_features = features[idx].unsqueeze(dim=0)

      # Perform RoI Align on the current image proposals.
      roi_features = torchvision.ops.roi_align(cr_features, [cr_proposal],
                                output_size=(self.roi_output_w, self.roi_output_h))
      # Meanpool the feature in the spatial dimesion (i.e. the two last dimensions).
      # Now, 'roi_features' is a 2-D tensor (instead of 4-D before the meanpool).
      roi_features = torch.mean(roi_features, (2, 3))

      with torch.no_grad():  # "no_grad" context used to speed up the computation.
        # Pass the RoI feature through the region classification layer.
        class_prob = self.cls_layer(roi_features)

      if class_prob.shape[0] == 0:  # Current image doesn't contain any box.
        # Initialize an empty tensor (with "int" type, on the GPU).
        cr_final_class = torch.tensor([], dtype=torch.int32, device='cuda')
      else:  # Current image contains at least one box.
        # Get the maximum score indices (i.e. most probable class) for each box.
        # 'cr_final_class' has shape of (<num_boxes>, 1) where each value is in
        # range [0,<num_classes>] (in our case: num_classes=20).
        cr_final_class = torch.argmax(class_prob, dim=1, keepdim=True)

      final_class.append(cr_final_class)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_probs, final_class
