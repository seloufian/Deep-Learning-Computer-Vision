from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)

        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        z1, z1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])

        a2, a2_cache = affine_forward(z1, self.params['W2'], self.params['b2'])

        scores = a2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, d_scores = softmax_loss(scores, y)

        reg_weights = np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2)

        loss += 0.5 * self.reg * reg_weights

        d_z1, d_W2, d_b2 = affine_backward(d_scores, a2_cache)

        d_x, d_W1, d_b1 = affine_relu_backward(d_z1, z1_cache)

        grads['b2'] = d_b2
        grads['W2'] = d_W2 + self.reg * self.params['W2']

        grads['b1'] = d_b1
        grads['W1'] = d_W1 + self.reg * self.params['W1']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


############################################################################
# Define additional helper layer which takes into account the Batch/Layer  #
# Normalization (Forward and Backward pass).                               #
############################################################################
def affine_norm_relu_forward(x, w, b, gamma, beta, n_param, normalization):
    """
    Convenience layer that perorms an affine transform followed by a Batch/Layer
    Normalization, and a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta, n_param: Batch/Layer Normalization parameters
    - normalization: Indicate the Normalization type, Batch or Layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a1, fc_cache = affine_forward(x, w, b)

    if normalization == 'batchnorm':
      a2, norm_cache = batchnorm_forward(a1, gamma, beta, n_param)

    elif normalization == 'layernorm':
      a2, norm_cache = layernorm_forward(a1, gamma, beta, n_param)

    out, relu_cache = relu_forward(a2)
    cache = (fc_cache, norm_cache, relu_cache)

    return out, cache


def affine_norm_relu_backward(dout, cache, normalization):
    """
    Backward pass for the affine-norm-relu convenience layer
    """
    fc_cache, n_cache, relu_cache = cache

    da1 = relu_backward(dout, relu_cache)

    if normalization == 'batchnorm':
      da2, dgamma, dbeta = batchnorm_backward_alt(da1, n_cache)

    elif normalization == 'layernorm':
      da2, dgamma, dbeta = layernorm_backward(da1, n_cache)

    dx, dw, db = affine_backward(da2, fc_cache)

    return dx, dw, db, dgamma, dbeta

############################################################################
#                        END OF THE ADDITIONAL CODE                        #
############################################################################


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        hidden_dims.append(num_classes)

        for layer_idx, layer_dim in enumerate(hidden_dims):
          if layer_idx == 0:
            prev_layer_dim = input_dim
          else:
            prev_layer_dim = hidden_dims[layer_idx-1]

          weights_name = 'W' + str(layer_idx+1)
          weights_dim = (prev_layer_dim, layer_dim)

          biases_name = 'b' + str(layer_idx+1)

          self.params[weights_name] = np.random.normal(0.0, weight_scale, weights_dim)
          self.params[biases_name] = np.zeros(layer_dim)

          # Add Batch/Layer Normalization layer parameter initialization (Beta and Gamma).
          # Don't apply Batch/Layer Normalization on the output layer.
          if self.normalization and layer_idx < self.num_layers-1:
            self.params['gamma' + str(layer_idx+1)] = np.ones(layer_dim)
            self.params['beta' + str(layer_idx+1)] = np.zeros(layer_dim)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache_layers, dp_cache_layers, last_output = [], [], None
        reg_weights = 0

        for layer_idx in range(self.num_layers):
          weights = self.params['W' + str(layer_idx+1)]
          biases = self.params['b' + str(layer_idx+1)]

          if layer_idx == 0:
            last_output = X

          if layer_idx == self.num_layers-1:
            last_output, cache = affine_forward(last_output, weights, biases)

          else:
            if self.normalization:
              last_output, cache = affine_norm_relu_forward(last_output,
                                    weights, biases,
                                    self.params['gamma' + str(layer_idx+1)],
                                    self.params['beta' + str(layer_idx+1)],
                                    self.bn_params[layer_idx],
                                    self.normalization)
            else:
              last_output, cache = affine_relu_forward(last_output, weights, biases)

            if self.use_dropout:
              last_output, dp_cache = dropout_forward(last_output, self.dropout_param)
              dp_cache_layers.append(dp_cache)

          cache_layers.append(cache)

          reg_weights += np.sum(weights**2)

        scores = last_output

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, d_scores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * reg_weights

        d_layer = None

        for layer_idx in range(self.num_layers-1, -1, -1):
          weights_name = 'W' + str(layer_idx+1)
          biases_name = 'b' + str(layer_idx+1)

          if layer_idx == self.num_layers-1:
            d_layer, d_W, d_b = affine_backward(d_scores, cache_layers[layer_idx])
          else:
            if self.use_dropout:
              d_layer = dropout_backward(d_layer, dp_cache_layers[layer_idx])

            if self.normalization:
              d_layer, d_W, d_b, d_gamma, d_beta = affine_norm_relu_backward(d_layer,
                                                    cache_layers[layer_idx],
                                                    self.normalization)
              grads['gamma' + str(layer_idx+1)] = d_gamma
              grads['beta' + str(layer_idx+1)] = d_beta

            else:
              d_layer, d_W, d_b = affine_relu_backward(d_layer, cache_layers[layer_idx])

          grads[weights_name] = d_W + self.reg * self.params[weights_name]
          grads[biases_name] = d_b

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
