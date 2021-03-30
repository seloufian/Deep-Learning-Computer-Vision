from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Gradient for the Softmax function.
    # Check: https://stackoverflow.com/a/53972798

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        correct_class_score = scores[y[i]]
        numerator = np.exp(correct_class_score)

        denominator = 0.0
        for j in range(num_classes):
            denominator += np.exp(scores[j])

        loss += - np.log(numerator / denominator)

        dW[:, y[i]] += (numerator / denominator - 1) * X[i]
        for j in range(num_classes):
            if j != y[i]:
              dW[:, j] += (np.exp(scores[j]) / denominator) * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W # Derivate of L2 norm

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.exp(X.dot(W))

    sum_scores = scores.sum(axis=1)

    y_scores = scores[np.arange(len(scores)), y]

    loss = - np.log(y_scores / sum_scores)
    loss = np.mean(loss)
    loss += reg * np.sum(W * W)

    scores /= sum_scores.reshape(-1, 1)
    scores[np.arange(len(scores)), y] -= 1

    dW = X.T.dot(scores)
    dW /= X.shape[0]
    dW += reg * 2 * W # Derivate of L2 norm

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
