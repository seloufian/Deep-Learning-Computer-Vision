import torch
import random
import torchvision.transforms as T
import numpy as np
from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d

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
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Training loop: Apply gradient ascent 100 times, in maximum.
    for epoch in range(100):
      # Forward pass, "scores" shape is (1, 1000)
      scores = model(X_fooling)

      # Get the predicted class (pred) and its socre (pred_score).
      pred_score, pred = torch.max(scores, axis=1)
      pred_score, pred = pred_score.item(), pred.item()

      # Get the "target_y" score.
      target_score = scores[:, target_y].squeeze()

      # Display some information about the current epoch (iteration).
      print('Epoch: %2d ' % (epoch+1) + \
            '| Predicted class: %3d (Score: %.2f) ' % (pred, pred_score) + \
            '| Target class: %d (Score: %.2f)' % (target_y, target_score.item()))

      # Check if the model is fooled, i.e. "predicted class" equals "target_y".
      if pred == target_y:
        print('\nThe model is fooled.')
        break

      # Apply the backward pass: Compute the gradient of "target score" w.r.t.
      # model's trainable parameters (among others, "X_fooling").
      target_score.backward()

      # Normalize the gradient (Note that "L2 norm" was used in the division).
      X_fooling.grad *= learning_rate / torch.linalg.norm(X_fooling.grad)

      # Compute an update step: Apply the gradient ascent.
      # Note that an addition is used (+=) insted of substraction (-=), because
      # the goal is to maximize "target_y" predicted score.
      X_fooling.data += X_fooling.grad.data

      # Re-initialize the gradient of "X_fooling" to zero (for the next epoch).
      X_fooling.grad.data.zero_()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X
