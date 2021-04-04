"""
Implements a network visualization in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

# import os
import torch
# import torchvision
# import torchvision.transforms as T
# import random
# import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from a4_helper import *


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from network_visualization.py!')

def compute_saliency_maps(X, y, model):
  """
  Compute a class saliency map using the model for images X and labels y.

  Input:
  - X: Input images; Tensor of shape (N, 3, H, W)
  - y: Labels for X; LongTensor of shape (N,)
  - model: A pretrained CNN that will be used to compute the saliency map.

  Returns:
  - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
  images.
  """
  # Make input tensor require gradient
  X.requires_grad_()
  
  saliency = None
  ##############################################################################
  # TODO: Implement this function. Perform a forward and backward pass through #
  # the model to compute the gradient of the correct class score with respect  #
  # to each input image. You first want to compute the loss over the correct   #
  # scores (we'll combine losses across a batch by summing), and then compute  #
  # the gradients with a backward pass.                                        #
  # Hint: X.grad.data stores the gradients                                     #
  ##############################################################################
  # Replace "pass" statement with your code

  # Make a forward pass of X (which contains N images) through the model.
  # The output (scores) has shape (N, C): For each image, get its unnormalized
  # scores (for each class of the dataset), e.g. C=1000 for a model trained on ImageNet.
  scores = model(X)

  # Get the -unnormalized- score of the correct class for each image.
  # "cscores" has shape of (N,)
  cscores = scores.gather(1, y.view(-1, 1)).squeeze()

  # Compute the loss over the correct scores.
  # As mentioned above, the loss is the sum across batch correct class scores.
  loss = torch.sum(cscores)
  # Apply the backward pass, which computes the gradient of the loss
  # w.r.t. our model's parameters (among others, the input X).
  loss.backward()

  # Note that we can apply the backward pass directly from "cscores" by using:
  # >>> cscores.backward(gradient=torch.ones_like(y))
  # The reason: The sub-computational graph for the "sum" method is:
  # -----
  # Forward pass:                 cscores   ---> [sum] ---> loss
  # Backward pass (gradiants):  [1, ..., 1] <--------------   1
  # -----
  # That is, we can directly start from "cscores" gradient, which is a tensor of
  # ones with the shape (N,). Actually: ones_like(y) == ones_like(cscores)

  # Compute the absolute value of the X gradients.
  # Saliency Maps requires nonnegative values (gradients).
  # For now, "saliency" has shape of: (N, 3, H, W)
  saliency = X.grad.abs()
  # Take the maximum value over the 3 input channels (for each of N images).
  # Now, "saliency" has shape of: (N, H, W)
  saliency = torch.max(saliency, dim=1).values

  ##############################################################################
  #               END OF YOUR CODE                                             #
  ##############################################################################
  return saliency

def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
  """
  Generate an adversarial attack that is close to X, but that the model classifies
  as target_y.

  Inputs:
  - X: Input image; Tensor of shape (1, 3, 224, 224)
  - target_y: An integer in the range [0, 1000)
  - model: A pretrained CNN
  - max_iter: Upper bound on number of iteration to perform
  - verbose: If True, it prints the pogress (you can use this flag for debugging)

  Returns:
  - X_adv: An image that is close to X, but that is classifed as target_y
  by the model.
  """
  # Initialize our adversarial attack to the input image, and make it require gradient
  X_adv = X.clone()
  X_adv = X_adv.requires_grad_()
  
  learning_rate = 1
  ##############################################################################
  # TODO: Generate an adversarial attack X_adv that the model will classify    #
  # as the class target_y. You should perform gradient ascent on the score     #
  # of the target class, stopping when the model is fooled.                    #
  # When computing an update step, first normalize the gradient:               #
  #   dX = learning_rate * g / ||g||_2                                         #
  #                                                                            #
  # You should write a training loop.                                          #
  #                                                                            #
  # HINT: For most examples, you should be able to generate an adversarial     #
  # attack in fewer than 100 iterations of gradient ascent.                    #
  # You can print your progress over iterations to check your algorithm.       #
  ##############################################################################
  # Replace "pass" statement with your code

  # Training loop: Apply gradient ascent 100 times, in maximum.
  for epoch in range(100):
    # Forward pass, "scores" shape is (1, 1000)
    scores = model(X_adv)

    # Get the predicted class (pred) and its socre (pred_score).
    pred_score, pred = torch.max(scores, axis=1)
    pred_score, pred = pred_score.item(), pred.item()

    # Get the "target_y" score.
    target_score = scores[:, target_y].squeeze()

    # Display some information about the current epoch (iteration).
    print('Iteration %2d: target score %.3f, max score %.3f' \
          % (epoch+1, target_score.item(), pred_score))

    # Check if the model is fooled, i.e. "predicted class" equals "target_y".
    if pred == target_y:
      print('\nThe model is fooled.')
      break

    # Apply the backward pass: Compute the gradient of "target score" w.r.t.
    # model's trainable parameters (among others, "X_adv").
    target_score.backward()

    # Normalize the gradient (Note that "L2 norm" was used in the division).
    X_adv.grad *= learning_rate / torch.linalg.norm(X_adv.grad)

    # Compute an update step: Apply the gradient ascent.
    # Note that an addition is used (+=) insted of substraction (-=), because
    # the goal is to maximize "target_y" predicted score.
    X_adv.data += X_adv.grad.data

    # Re-initialize the gradient of "X_adv" to zero (for the next epoch).
    X_adv.grad.data.zero_()

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the 
    score of target_y under a pretrained model.
  
    Inputs:
    - img: random image with jittering as a PyTorch tensor  
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code

    # Forward pass, "scores" shape is (1, 1000)
    scores = model(img)

    # Get the "target_y" score.
    target_score = scores[:, target_y].squeeze()
    # Add the regularization term (Note that the L2 norm is squared).
    target_score -= l2_reg * torch.square(torch.linalg.norm(img))

    # Apply the backward pass: Compute the gradient of "target score" w.r.t.
    # model's trainable parameters (among others, "img").
    target_score.backward()

    # Compute an update step: Apply the gradient ascent.
    # Note that an addition is used (+=) insted of substraction (-=), because
    # the goal is to maximize "target_y" predicted score.
    img.data += learning_rate * img.grad.data

    # Re-initialize the gradient of "img" to zero.
    img.grad.data.zero_()

    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
