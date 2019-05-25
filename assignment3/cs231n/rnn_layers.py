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
    # linear_out = x.dot(Wx) + prev_h.dot(Wh) + b
    # next_h = np.tanh(linear_out)
    w_x = x.dot(Wx)            # 1
    w_x_b = w_x + b            # 2
    w_h = prev_h.dot(Wh)       # 3 
    lin_out = w_x_b + w_h      # 4
    next_h = np.tanh(lin_out)  # 5

    cache = w_x, w_x_b, w_h, lin_out, prev_h, Wh, Wx, b, x
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
    w_x, w_x_b, w_h, lin_out, prev_h, Wh, Wx, b, x = cache
    dlin_out = (1 - np.tanh(lin_out)**2) * dnext_h # 5
    dw_x_b = 1 * dlin_out  # 4
    dw_h = 1 * dlin_out  # 4
    dWh = prev_h.T.dot(dw_h)  # 3
    dprev_h = dw_h.dot(Wh.T)  # 3
    db = np.sum(dw_x_b, axis=0)  # 2
    dw_x = 1 * dw_x_b  # 2
    dWx = x.T.dot(dw_x)  # 1
    dx = dw_x.dot(Wx.T)  # 1
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
    N, T, D = x.shape
    #H = b.shape

    hidden_states = list()
    caches_timestep = list()

    current_h = h0
    for t_i in range(T):
        next_h, next_cache = rnn_step_forward(x[:,t_i,:], current_h, Wx, Wh, b)
        hidden_states.append(next_h)
        caches_timestep.append(next_cache)
        current_h = next_h
    h = np.array(hidden_states).transpose((1,0,2))
    cache = caches_timestep
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
    gradients_timestep = list()
    T = len(cache)

    dx, dprev_h, dWx, dWh, db, dWx_tot, dWh_tot, db_tot = [0]*8
    dx_tot, dprev_h_tot = [None] * T, [None] * T

    for t_i in reversed(range(T)):
        dnext_h = dh[:,t_i,:] + dprev_h
        dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache[t_i])
        dx_tot[t_i] = dx
        dprev_h_tot[t_i] = dprev_h
        dWx_tot += dWx
        dWh_tot += dWh
        db_tot += db

    dx_tot = np.array(dx_tot).transpose(1,0,2)
    dx, dh0, dWx, dWh, db = dx_tot, dprev_h_tot[0], dWx_tot, dWh_tot, db_tot
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
    out = W[x.flatten(),:].reshape((*(x.shape),-1))
    cache = (W, x)  # TODO - needed?
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
    W, x = cache
    N = x.shape[0]
    dW = np.zeros_like(W)
    # Solution from https://github.com/mesuvash
    for i in range(N):
        indices = x[i, :]
        np.add.at(dW, indices, dout[i, :, :])
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
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
    # 1 Computes all in one
    activation_vector = x.dot(Wx) + prev_h.dot(Wh) + b  
    a_i, a_f, a_o, a_g = np.array_split(activation_vector, 4, axis=1)
    # 2 Compute all non-linearities
    g_i, g_f, g_o, g_g = sigmoid(a_i),  sigmoid(a_f), sigmoid(a_o), np.tanh(a_g)
    # 3-4 Compute intermediate values
    g_f_prev_c = g_f * prev_c
    g_i_g_g = g_i * g_g
    # 5-7 Compute output
    next_c = g_f_prev_c + g_i_g_g
    tanh_next_c = np.tanh(next_c)
    next_h = g_o * tanh_next_c
    cache = (g_f, prev_c, g_o, next_c, tanh_next_c, a_o, g_i, g_g, a_i, a_f, a_g, 
             Wx, x, Wh, prev_h)
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
    N = dnext_h.shape[0]
    (g_f, prev_c, g_o, next_c, tanh_next_c, a_o, g_i, g_g, a_i, a_f, a_g, 
     Wx, x, Wh, prev_h) = cache
    
    # Calculate derivatives
    dtanh_next_c = g_o * dnext_h                           # 1
    dg_o = tanh_next_c * dnext_h                           # 1
    dnext_c_2 = (1 - (np.tanh(next_c))**2) * dtanh_next_c  # 2
    dnext_c_total = dnext_c + dnext_c_2  # Accumulate gradients from branches
    dg_f_prev_c = dnext_c_total                            # 3
    dg_i_g_g = dnext_c_total                               # 3
    dprev_c = g_f * dg_f_prev_c                            # 4
    dg_f = prev_c * dg_f_prev_c                            # 4
    da_o = sigmoid(a_o) * (1 - sigmoid(a_o)) * dg_o        # 5
    dg_g = g_i * dg_i_g_g                                  # 6
    dg_i = g_g * dg_i_g_g                                  # 6
    da_g = (1 - (np.tanh(a_g))**2) * dg_g                  # 7
    da_i = sigmoid(a_i) * (1 - sigmoid(a_i)) * dg_i        # 8
    da_f = sigmoid(a_f) * (1 - sigmoid(a_f)) * dg_f        # 9

    dactivation_vector = np.concatenate([da_i, da_f, da_o, da_g], axis=1)
    dx = dactivation_vector.dot(Wx.T)
    dWx = x.T.dot(dactivation_vector)
    dprev_h = dactivation_vector.dot(Wh.T)
    dWh = prev_h.T.dot(dactivation_vector)
    db = np.sum(dactivation_vector, axis = 0)
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
    N, T, D = x.shape

    hidden_states = list()
    caches_timestep = list()

    next_h = h0
    next_c = np.zeros_like(h0)
    for t_i in range(T):
        next_h, next_c, cache = lstm_step_forward(x[:,t_i,:], next_h, next_c, Wx, Wh, b)
        hidden_states.append(next_h)
        caches_timestep.append(cache)
    h = np.array(hidden_states).transpose((1,0,2))
    cache = caches_timestep
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
    gradients_timestep = list()
    T = len(cache)

    dx, dprev_h, dWx, dWh, db, dWx_tot, dWh_tot, db_tot = [0]*8
    dx_tot, dprev_h_tot, dprev_c_tot = [None] * T, [None] * T, [None] * T
    dprev_c = dprev_h
    for t_i in reversed(range(T)):
        dnext_h = dh[:,t_i,:] + dprev_h
        dnext_c = dprev_c
        dx, dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache[t_i])
        dx_tot[t_i] = dx
        dprev_h_tot[t_i] = dprev_h
        dprev_c_tot[t_i] = dprev_c
        dWx_tot += dWx
        dWh_tot += dWh
        db_tot += db

    dx_tot = np.array(dx_tot).transpose(1,0,2)
    dx, dh0, dWx, dWh, db = dx_tot, dprev_h_tot[0], dWx_tot, dWh_tot, db_tot
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

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

