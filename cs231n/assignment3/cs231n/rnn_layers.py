from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the input dot product (inprod), output shape is (N, H).
    inprod = x @ Wx
    # Compute the hidden dot product (hprod), output shape is (N, H).
    hprod = prev_h @ Wh
    # Compute the pre-activation (preact) sum.
    preact = inprod + hprod + b
    # Apply the RNN formula (with "tanh" activation function).
    next_h = np.tanh(preact)
    # Store the needed cache variables for the backward pass.
    cache = (x, Wx, prev_h, Wh, b, preact)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack cached variables (from the forward pass).
    x, Wx, prev_h, Wh, b, preact = cache

    # Compute the gradiant from: next_h = np.tanh(preact)
    # "tanh" derivate is: 1 - tanh^2
    d_preact = (1 - np.tanh(preact)**2) * dnext_h

    # Compute "db" from: preact = ... + b
    # Apply "sum" function to match the initial "b" shape.
    db = d_preact.sum(axis=0)

    # Compute "dWh" and "dprev_h" from: hprod = prev_h @ Wh
    # "Transpose" operation applied to match the initial shape.
    dWh = prev_h.T @ d_preact
    dprev_h = d_preact @ Wh.T

    #Compute "dWx" and "dx" from: inprod = x @ Wx
    dWx = x.T @ d_preact
    dx = d_preact @ Wx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get dimension values from "x" and "h0"
    N, T, D = x.shape
    H = h0.shape[1]

    # Initialize the hidden states array for the entire timeseries.
    h = np.zeros((N, T, H))
    # Initialize the cache. It will contain the cache for all timeseries.
    cache = []
    # Initialize the previous hidden state (prev_h) with the initial one.
    prev_h = h0

    # Loop over timeseries. Current timeserie is "ts" (integer).
    for ts in range(T):
      # Get the minibatch for current "ts".
      ts_x = x[:, ts, :]
      # Apply the forward step.
      ts_h, ts_cache = rnn_step_forward(ts_x, prev_h, Wx, Wh, b)
      # Currect timeserie hidden state (ts_h) will be 'previous' in the next "ts".
      prev_h = ts_h
      # Add "ts_h" to the hidden states array.
      h[:, ts, :] = ts_h
      # Add the current timestep cache (ts_cache) to the global cache array.
      cache.append(ts_cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get dimension values from "dh"
    N, T, H = dh.shape
    # Get the "D" value from the first timestep, first cached variable (the "x").
    D = cache[0][0].shape[1]

    # Initialize gradients with zeros.
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    # Initialize the derivate of the previous timestep hidden output (dprev_tsh)
    # with the last "dh", because:
    # - In the RNN backward pass, we must iterate over timesteps from the end
    # to the beginnig.
    # - The last hidden output derivate is the derivative of the input to the
    # loss function only (since there is no another timestep).
    dprev_tsh = dh[:, T-1, :]

    # Loop through timesteps in descending order. Current timeserie is "ts" (integer).
    for ts in range(T-1, -1, -1):
      # Apply the backward step.
      out_backward = rnn_step_backward(dprev_tsh, cache[ts])

      # Assign current timestep 'x' gadient.
      dx[:, ts, :] = out_backward[0]
      # Sum 'dWx', 'dWh' and 'db' gradients.
      dWx += out_backward[2]
      dWh += out_backward[3]
      db += out_backward[4]

      # Get the current hidden timestep input 'first' gradient.
      dprev_h = out_backward[1]
      # Check if we got to the first timestep (i.e. The last backward-ed timestep).
      if ts == 0:
        dh0 = dprev_h
      else:
        # The current hidden timestep input is the sum of the "1st input gradient"
        # and the "upcoming gradient from the loss".
        dprev_tsh = dprev_h + dh[:, ts-1, :]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = W[x]

    # Only 'x' and 'shape of W' must be cached ('W' itself is not needed).
    cache = (x, W.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, Wshape = cache

    # 'dW' must have the same shape as 'W'.
    dW = np.zeros(Wshape)

    np.add.at(dW, x, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the input dot product (inprod), output shape is (N, 4H).
    inprod = x @ Wx
    # Compute the hidden dot product (hprod), output shape is (N, 4H).
    hprod = prev_h @ Wh
    # Compute the pre-activation (preact) sum, output shape is (N, 4H).
    preact = inprod + hprod + b

    # Split "preact" along the column-axis into 4 parts [each of shape (N, H)].
    ai, af, ao, ag = tuple(np.split(preact, 4, axis=1))

    # Compute respectively 'input', 'forget', 'output' and 'block' gates.
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)

    # Compute the next cell state (next_c).
    next_c = f * prev_c + i * g
    # Compute the next hidden state (next_h).
    ncth = np.tanh(next_c)
    next_h = o * ncth

    # Store the needed cache variables for the backward pass.
    cache = (x, prev_h, prev_c, Wx, Wh, b, preact, i, f, o, g, ncth)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack cached variables (from the forward pass).
    x, prev_h, prev_c, Wx, Wh, b, preact, i, f, o, g, ncth = cache

    # Compute "do" and "d_ncth" from: next_h = o * ncth
    do = ncth * dnext_h
    d_ncth = o * dnext_h

    # Compute the derivate of "next_c" used to compute "next_h".
    d_next_c_h = d_ncth * (1 - ncth**2)
    # Sum output "dnext_c" with the one used to compute "next_h".
    dnext_c += d_next_c_h

    # Compute "dprev_c" and "df" from: next_c = f * prev_c + ...
    dprev_c = f * dnext_c
    df = prev_c * dnext_c

    # Compute "di" and "dg" from: next_c = ... + i * g
    di = g * dnext_c
    dg = i * dnext_c

    # Compute gates derivates: "dai", "daf", "dao" and "dag".
    # Derivate of the "sigmoid" is: sig(x)' = sig(x) * (1 - sig(x))
    # Derivate of the "tanh" is: tanh(x)' = 1 - tanh(x)^2
    dai = (i * (1 - i)) * di
    daf = (f * (1 - f)) * df
    dao = (o * (1 - o)) * do
    dag = (1 - g**2) * dg

    # Stack horizontally gates derivates [each of shape (N, H)].
    # "d_preact" shape is (N, 4H)
    d_preact = np.hstack((dai, daf, dao, dag))

    # Compute "db" from: preact = ... + b
    # Apply "sum" function to match the initial "b" shape.
    db = d_preact.sum(axis=0)

    # Compute "dWh" and "dprev_h" from: hprod = prev_h @ Wh
    # "Transpose" operation applied to match the initial shape.
    dWh = prev_h.T @ d_preact
    dprev_h = d_preact @ Wh.T

    #Compute "dWx" and "dx" from: inprod = x @ Wx
    dWx = x.T @ d_preact
    dx = d_preact @ Wx.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get dimension values from "x" and "h0"
    N, T, D = x.shape
    H = h0.shape[1]

    # Initialize the hidden states array for the entire timeseries.
    h = np.zeros((N, T, H))
    # Initialize the cache. It will contain the cache for all timeseries.
    cache = []
    # Initialize the previous hidden state (prev_h) with the initial one.
    prev_h = h0
    # Initialize the cell state with zeros.
    prev_c = np.zeros((N, H))

    # Loop over timeseries. Current timeserie is "ts" (integer).
    for ts in range(T):
      # Get the minibatch for current "ts".
      ts_x = x[:, ts, :]
      # Apply the forward step.
      ts_h, ts_c, ts_cache = lstm_step_forward(ts_x, prev_h, prev_c, Wx, Wh, b)
      # Current timeserie hidden state (ts_h) will be 'previous' in the next "ts".
      # The same for "ts_c".
      prev_h, prev_c = ts_h, ts_c
      # Add "ts_h" to the hidden states array.
      h[:, ts, :] = ts_h
      # Add the current timestep cache (ts_cache) to the global cache array.
      cache.append(ts_cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get dimension values from "dh"
    N, T, H = dh.shape
    # Get the "D" value from the first timestep, first cached variable (the "x").
    D = cache[0][0].shape[1]

    # Initialize gradients with zeros.
    dx = np.zeros((N, T, D))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    # Initialize the derivate of the previous timestep hidden output (dprev_tsh)
    # with the last "dh", because:
    # - In the LSTM backward pass, we must iterate over timesteps from the end
    # to the beginnig.
    # - The last hidden output derivate is the derivative of the input to the
    # loss function only (since there is no another timestep).
    dprev_tsh = dh[:, T-1, :]
    # Initialize the last cell state gradient (dprev_c) with zeros. Since the last 
    # cell state is not used anywhere. So, its derivate equals to zero (matrix form).
    dprev_c = np.zeros((N, H))

    # Loop through timesteps in descending order. Current timeserie is "ts" (integer).
    for ts in range(T-1, -1, -1):
      # Apply the backward step.
      # "out_backward" contains (respectively): dx, dprev_h, dprev_c, dWx, dWh, db
      out_backward = lstm_step_backward(dprev_tsh, dprev_c, cache[ts])

      # Assign current timestep 'x' gadient.
      dx[:, ts, :] = out_backward[0]
      # Sum 'dWx', 'dWh' and 'db' gradients.
      dWx += out_backward[3]
      dWh += out_backward[4]
      db += out_backward[5]

      dprev_c = out_backward[2]

      # Get the current hidden timestep input 'first' gradient.
      dprev_h = out_backward[1]
      # Check if we got to the first timestep (i.e. The last backward-ed timestep).
      if ts == 0:
        dh0 = dprev_h
      else:
        # The current hidden timestep input is the sum of the "1st input gradient"
        # and the "upcoming gradient from the loss".
        dprev_tsh = dprev_h + dh[:, ts-1, :]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
