from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    D = np.prod(x.shape[1:])

    x_input = x.reshape(N, D)

    out = x_input @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, M = dout.shape

    dx = dout @ w.T
    dx = dx.reshape(*x.shape)

    D = np.prod(x.shape[1:])
    x_input = x.reshape(N, D)

    dw = x_input.T @ dout

    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.where(x<0, 0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.where(x<0, 0, 1) * dout

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Batch normalization algorithm rewritten in the way to store intermediate
        # values in "cache" for backpropagation.

        # Compute sample mean for each feature.
        xmean = x.mean(axis=0)

        xmean_diff = x - xmean

        xmean_diff_square = xmean_diff**2

        # Compute sample variance for each feature.
        xvar = xmean_diff_square.mean(axis=0)

        # Compute sample standard deviation for each feature.
        xstd = np.sqrt(xvar + eps)

        xstd_inv = 1 / xstd

        # Normalize.
        xhat = xmean_diff * xstd_inv

        # Scale and shift.
        out = gamma * xhat + beta

        # Update 'running_mean' and 'running_var'.
        running_mean = momentum * running_mean + (1 - momentum) * xmean
        running_var = momentum * running_var + (1 - momentum) * xvar

        # Save intermediates needed for the backward pass.
        cache = {
          'xmean_diff': xmean_diff,
          'xvar': xvar,
          'eps': eps,
          'xstd': xstd,
          'xstd_inv': xstd_inv,
          'xhat': xhat,
          'gamma': gamma,
          'beta': beta
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Normalize using 'running_mean' and 'running_var'.
        out = (x - running_mean) / (np.sqrt(running_var) + eps)

        # Scale and shift.
        out = gamma * out + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # In order to fully understand the backward pass, computational graph of
    # batch normalization layer must be drawn.

    N, D = dout.shape

    # Derivate from: out = xscaled * beta
    dbeta = dout.sum(axis=0)

    # Derivate from: xscaled = xhat * gamma
    dgamma = dout * cache['xhat']
    dgamma = dgamma.sum(axis=0)

    # Derivate from: xscaled = xhat * gamma
    d_xhat = dout * cache['gamma']

    # Derivate from: xhat = xstd_inv * xmean_diff
    d_xstd_inv = cache['xmean_diff'] * d_xhat
    d_xstd_inv = d_xstd_inv.sum(axis=0)

    # Derivate from: xstd_inv = 1 / xstd
    d_xstd = (-1 / cache['xstd']**2) * d_xstd_inv

    # Derivate from: xstd = sqrt(xvar + eps)
    d_xvar = (1 / (2 * np.sqrt(cache['xvar'] + cache['eps']))) * d_xstd

    # Derivate from: xvar = xmean_diff_square.mean(axis=0)
    d_xmean_diff_square = d_xvar / N

    # Derivate from: xmean_diff_square = xmean_diff ^ 2
    d_xmean_diff_2 = (2 * cache['xmean_diff']) * d_xmean_diff_square

    # Derivate from: xstd_inv * xmean_diff
    d_xmean_diff_1 = cache['xstd_inv'] * d_xhat

    # Sum the two derivates.
    d_xmean_diff = d_xmean_diff_1 + d_xmean_diff_2

    # Derivate from: d_xmean_diff = x - xmean
    d_xmean = - d_xmean_diff.sum(axis=0)

    # Derivate from: xmean = x.mean(axis=0)
    d_x2 = d_xmean / N

    # Derivate from: d_xmean_diff = x - xmean
    d_x1 = d_xmean_diff

    # Sum the two derivates.
    dx = d_x1 + d_x2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape

    # Unpack needed intermediates
    inv_var, xhat, gamma = cache['xstd_inv'], cache['xhat'], cache['gamma']

    # Derivate from: out = xscaled * beta
    dbeta = dout.sum(axis=0)

    # Derivate from: xscaled = xhat * gamma
    dgamma = np.sum(dout * xhat, axis=0)

    # Derivate from: xscaled = xhat * gamma
    d_xhat = dout * gamma

    # Result formula from work-out the derivatives for the batch normalization
    # backward pass on paper, including simplification.
    dx = inv_var / N * \
        (N*d_xhat - d_xhat.sum(axis=0) - xhat*np.sum(d_xhat*xhat, axis=0))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Layer normalization algorithm rewritten in the way to store intermediate
    # values in "cache" for backpropagation.

    # Compute the mean for each datapoint.
    xmean = x.mean(axis=1, keepdims=True)

    xmean_diff = x - xmean

    xmean_diff_square = xmean_diff**2

    # Compute the variance for each datapoint.
    xvar = xmean_diff_square.mean(axis=1, keepdims=True)

    # Compute the standard deviation for each datapoint.
    xstd = np.sqrt(xvar + eps)

    xstd_inv = 1 / xstd

    # Normalize.
    xhat = xmean_diff * xstd_inv

    # Scale and shift.
    out = gamma * xhat + beta

    # Save intermediates needed for the backward pass.
    cache = {
      'xstd_inv': xstd_inv,
      'xhat': xhat,
      'gamma': gamma
    }

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape

    # Unpack needed intermediates
    inv_var, xhat, gamma = cache['xstd_inv'], cache['xhat'], cache['gamma']

    # Derivate from: out = xscaled * beta
    dbeta = dout.sum(axis=0)

    # Derivate from: xscaled = xhat * gamma
    dgamma = np.sum(dout * xhat, axis=0)

    # Derivate from: xscaled = xhat * gamma
    d_xhat = dout * gamma

    # Formula obtained by slightly modifying the one used in 'batchnorm_backward_alt'
    dx = inv_var / D * \
        (D*d_xhat - d_xhat.sum(axis=1, keepdims=True) - \
        xhat*np.sum(d_xhat*xhat, axis=1, keepdims=True))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack 'stride' and 'pad' values.
    stride, pad = conv_param['stride'], conv_param['pad']

    # Get different size information from 'x' and 'w'.
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Define padding filter: Pad only the width and height of each 'x' sample.
    npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    # Apply the padding filter to 'x'.
    xpad = np.pad(x, npad)

    # Compute H' and W'.
    Hprime = int(1 + (H + 2 * pad - HH) / stride)
    Wprime = int(1 + (W + 2 * pad - WW) / stride)

    # Initialize the output.
    out = np.zeros((N, F, Hprime, Wprime))

    # Maximum start point in width and height from which the convolution
    # can be applied, to not exceed 'x' datapoints size.
    Hmax_xpad = 1 + (H + 2 * pad - HH)
    Wmax_xpad = 1 + (W + 2 * pad - WW)

    # Loop over all datapoints (samples).
    # Track current sample number (xnum) and the sample itself (xsample).
    for xnum, xsample in enumerate(xpad):
      # Loop over all weights (filters).
      # Track current filter number (wnum) and the filter itself (wfilter).
      for wnum, wfilter in enumerate(w):
        # Loop over 'xsample' width, where the convolution can be applied.
        # Take into account the padding, move per 'stride'.
        for iout, i in enumerate(range(0, Hmax_xpad, stride)):
          # Loop over 'xsample' height, where the convolution can be applied.
          # Take into account the padding, move per 'stride'.
          for jout, j in enumerate(range(0, Wmax_xpad, stride)):
            # 'xsample' part on which 'wfilter' will be applied.
            xpart = xsample[:, i:i+HH, j:j+WW]
            # Apply the convolution 'wfilter' on 'xpart', and add the bias.
            out[xnum, wnum, iout, jout] = (wfilter * xpart).sum() + b[wnum]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache

    # Get different size information from 'x', 'w' and 'dout'.
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, Hprime, Wprime = dout.shape

    # Unpack 'stride' and 'pad' values.
    stride, pad = conv_param['stride'], conv_param['pad']

    # Initialize 'dx', 'dw' and 'db'.
    dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)

    # Define padding filter: Pad only the width and height of each 'dx' sample.
    npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    # Apply the padding filter to 'dx', as 'out' was computed using padded 'x'.
    dxpad = np.pad(dx, npad)
    # Apply the padding filter to 'x', which will be used to compute 'dw'.
    xpad = np.pad(x, npad)

    # Maximum start point in width and height from which the convolution
    # can be applied, to not exceed the 'dx' size.
    Hmax_xpad = 1 + (H + 2 * pad - HH)
    Wmax_xpad = 1 + (W + 2 * pad - WW)

    # Loop over the output (result of application of the convolutional 
    # filters on 'x') derivative. Track datapoint number (xnum) to which 
    # current output derivate (xdout) belongs.
    for xnum, xdout in enumerate(dout):
      # Loop over 'xdout'.
      for dnum, dfilter in enumerate(xdout):
        # Loop over 'dfilter' lines (height).
        for i, idx in enumerate(range(0, Hmax_xpad, stride)):
          # Loop over 'xsample' height, where the convolution can be applied.
          # Take into account the padding, move per 'stride'.
          for j, jdx in enumerate(range(0, Wmax_xpad, stride)):
            # Get the current 'dfilter' value.
            dfilterval = dfilter[i, j]
            # Update 'db', 'dxpad' and 'dw'.
            db[dnum] += dfilterval
            # Note that we update 'dxpad' instead of 'dx'.
            dxpad[xnum, :, idx:idx+HH, jdx:jdx+WW] += w[dnum, ...] * dfilterval
            # Note that we update 'dw' using 'xpad' instead of 'x'.
            dw[dnum, ...] += xpad[xnum, :, idx:idx+HH, jdx:jdx+WW] * dfilterval

    # Shrink the padding from 'dx' lines/columns, in order to match 'x' size.
    dx = dxpad[..., pad:-pad, pad:-pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack max-pooling layer parameters: 'width', 'height' and 'stride'.
    pool_width, pool_height = pool_param['pool_width'], pool_param['pool_height']
    stride = pool_param['stride']

    # Get different size information from 'x'.
    N, C, H, W = x.shape

    # Compute H' and W'.
    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)

    # Initialize the output.
    out = np.zeros((N, C, Hprime, Wprime))

    # Maximum start point in width and height from which the max-pooling
    # can be applied, to not exceed 'x' datapoints size.
    Hmaxx = 1 + (H - pool_height)
    Wmaxx = 1 + (W - pool_width)

    # Loop over all datapoints (samples).
    # Track current sample number (xnum) and the sample itself (xsample).
    for xnum, xsample in enumerate(x):
      # Loop over all channels.
      # Track current channel number (cnum) and the channel itself (channel).
      for cnum, channel in enumerate(xsample):
        # Loop over 'channel' width, where the max-pooling can be applied.
        for iout, i in enumerate(range(0, Hmaxx, stride)):
          # Loop over 'channel' height, where the max-pooling can be applied.
          for jout, j in enumerate(range(0, Wmaxx, stride)):
            # 'channel' part on which max-pooling will be applied.
            chpart = channel[i:i+pool_height, j:j+pool_width]
            # Get the maximum value in 'chpart'
            out[xnum, cnum, iout, jout] = chpart.max()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack max-pooling layer cache.
    x, pool_param = cache

    # Unpack max-pooling layer parameters: 'width', 'height' and 'stride'.
    pool_width, pool_height = pool_param['pool_width'], pool_param['pool_height']
    stride = pool_param['stride']

    # Get different size information from 'x'.
    N, C, H, W = x.shape

    # Compute H' and W'.
    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)

    # Initialize 'dx' (which has the same shape as 'x').
    dx = np.zeros(x.shape)

    # Maximum start point in width and height, to not exceed 'x' datapoints size.
    Hmaxx = 1 + (H - pool_height)
    Wmaxx = 1 + (W - pool_width)

    # Loop over all datapoints (samples).
    # Track current sample number (xnum) and the sample itself (xsample).
    for xnum, xsample in enumerate(x):
      # Loop over all channels.
      # Track current channel number (cnum) and the channel itself (channel).
      for cnum, channel in enumerate(xsample):
        # Loop over 'channel' width.
        for idout, i in enumerate(range(0, Hmaxx, stride)):
          # Loop over 'channel' height.
          for jdout, j in enumerate(range(0, Wmaxx, stride)):
            # 'channel' part from which the maximum coordinates will be retrieved.
            chpart = channel[i:i+pool_height, j:j+pool_width]
            # Get the maximum coordinates: Line (maxln) and column (maxcol)
            maxln, maxcol = np.unravel_index(np.argmax(chpart), chpart.shape)
            # Assign the current maximum from 'dout' to 'dx', based on
            # the 'maximum coordinates' retrieved previously.
            dx[xnum, cnum, i+maxln, j+maxcol] = dout[xnum, cnum, idout, jdout]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # N, C, H, W = Datapoints, Channels, Height, Width
    N, C, H, W = x.shape

    # Transpose 'x', from (N, C, H, W) shape to (N, H, W, C). Then, reshape it
    # to (N*H*W, C). That is, forward Spatial BN is applied per color channel.
    x = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    # Reshape and transpose 'out' to match the 'x' shape.
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape

    # Transpose 'dout', from (N, C, H, W) shape to (N, H, W, C). Then, reshape it
    # to (N*H*W, C). That is, backward Spatial BN is applied per color channel.
    dout = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)

    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

    # Reshape and transpose 'dx' to match the 'dout' shape.
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    # In spatial group norm, channels (with their height and width) are
    # devided into groups, each one contains C/G channels (integer value).
    # So, each group ('elsgrp') contains C/G channels with their H and W.
    elsgrp = C//G * H * W

    # Reshape 'x' to match the logic behind the spatial group norm.
    # Note that in the original implementation (https://arxiv.org/abs/1803.08494),
    # 'x' was reshaped into (N, G, C//G, H, W), however, my implementation is also
    # correct. It allows to lighten per-axis mean computations, as we perform it on
    # the 2nd axis only, instead of (2, 3, 4) as in the original implementation.
    x = x.reshape([N, G, elsgrp])

    # Compute the mean for each group.
    xmean = x.mean(axis=2, keepdims=True)

    xmean_diff = x - xmean

    xmean_diff_square = xmean_diff**2

    # Compute the variance for each group.
    xvar = xmean_diff_square.mean(axis=2, keepdims=True)

    # Compute the standard deviation for each group.
    xstd = np.sqrt(xvar + eps)

    xstd_inv = 1 / xstd

    # Normalize.
    xhat = xmean_diff * xstd_inv

    # Reshape 'xhat' to its original shape, from (N, G, elsgrp) to (N, C, H, W).
    xhat = xhat.reshape(N, C, H, W)

    # Scale and shift.
    out = gamma * xhat + beta

    cache = {
      'G': G,
      'xstd_inv': xstd_inv,
      'xhat': xhat,
      'gamma': gamma
    }

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape

    # Unpack needed intermediates
    inv_var, xhat, gamma = cache['xstd_inv'], cache['xhat'], cache['gamma']
    G = cache['G']

    # Derivate from: out = xscaled * beta
    dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)

    # Derivate from: xscaled = xhat * gamma
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3), keepdims=True)

    # Derivate from: xscaled = xhat * gamma
    d_xhat = dout * gamma

    # Similarly to the forward pass, compute number of elements per group.
    elsgrp = C//G * H * W

    # Similarly to the forward pass, reshape 'xhat' and 'd_xhat' to match the logic
    # behind spatial group norm.
    xhat = xhat.reshape([N, G, elsgrp])
    d_xhat = d_xhat.reshape([N, G, elsgrp])

    # Formula obtained by slightly modifying the one used in 'layernorm_backward'
    dx = inv_var / elsgrp * \
        (elsgrp*d_xhat - d_xhat.sum(axis=2, keepdims=True) - \
        xhat*np.sum(d_xhat*xhat, axis=2, keepdims=True))

    # Reshape 'dx' to match original 'x' shape, from (N, G, elsgrp) to (N, C, H, W).
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
