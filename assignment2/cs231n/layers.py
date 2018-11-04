from builtins import range
import numpy as np


def convolute_x(x, f_spatial_ext, stride):
    """
    Convolutes a 2D layer into a x_col layer.
    
    Parameters
    ----------
    
    x : 3D Numpy array. Shape ( C, H, W)
        Data to be convoluted.
    
    f_spatial_ext : int
        Filtersize
        
    stride : int
        Stride 
        
    Returns
    -------
    
    x_col : 2D Numpy array
        Convoluted array.
    """
    width , height = x.shape
    convolution = list()
    for h_ix in range(0, height-f_spatial_ext+1, stride):
        for w_ix in range(0, width-f_spatial_ext+1, stride):
            convolution.append(x[h_ix:h_ix+f_spatial_ext, w_ix:w_ix+f_spatial_ext].flatten())
    x_col = np.stack(convolution, axis=0).T
    return x_col


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
    out = x.reshape(x.shape[0], -1).dot(w) + b
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
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0, keepdims=True)
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
    out = np.maximum(0, x)
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
    mask = np.maximum(0, x)
    mask[mask > 0] = 1
    dx = dout * mask
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
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    # Variable referencing
    X = x
    if mode == 'train':
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
        
        # Batch norm layer - forward pass
        mu = np.mean(X, axis=0)          # 1.
        X_cent = X - mu                  # 2.
        X_num = X_cent                   # 2.
        X_centsq = X_cent ** 2           # 3.
        var = np.mean(X_centsq, axis=0)  # 4.
        sd = np.sqrt(var + eps)          # 5.
        den = 1 / sd                     # 6.
        X_hat = X_num * den              # 7.
        X_gamma = gamma * X_hat          # 8.
        X_out = X_gamma + beta           # 9.

        # Calculate running mean
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        # Stuff to return
        out = X_out
        cache = (X_cent, var, sd, X_num, den, X_hat, X_gamma, gamma, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # Batch norm layer - forward pass
        mu = running_mean                # (1.)
        X_cent = X - mu                  # 2.
        X_num = X_cent                   # 2.
        X_centsq = X_cent ** 2           # 3.
        var = running_var                # (4.)
        sd = np.sqrt(var + eps)          # 5.
        den = 1 / sd                     # 6.
        X_hat = X_num * den              # 7.
        X_gamma = gamma * X_hat          # 8.
        X_out = X_gamma + beta           # 9.

        # Stuff to return
        out = X_out
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

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

    X_cent, var, sd, X_num, den, X_hat, X_gamma, gamma, eps = cache

    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    N, D = dout.shape
    d_X_out = dout

    # Batch norm layer - backward pass
    d_beta = np.sum(d_X_out, axis = 0)                # 9.
    d_X_gamma = d_X_out                               # 9.
    d_X_hat = gamma * d_X_gamma                       # 8.
    d_gamma = np.sum(X_hat * d_X_gamma, axis=0)       # 8.
    d_X_num = den * d_X_hat                           # 7.
    d_den = np.sum(X_num * d_X_hat, axis=0)           # 7.
    d_sd = -1. / sd**2 * d_den                        # 6.
    d_var = 0.5 * 1 / np.sqrt(var + eps) * d_sd       # 5.
    d_X_centsq = 1 / N * np.ones_like(d_var) * d_var  # 4.
    d_X_cent = 2 * X_cent * d_X_centsq                # 3.
    d_X_1 = d_X_cent + d_X_num                        # 2.
    d_mu  = -1 * np.sum(d_X_cent + d_X_num, axis=0)   # 2.
    d_X_2 = 1 / N * np.ones_like(d_X_out) * d_mu      # 1.
    d_X = (d_X_1 + d_X_2)                             # 0.

    dgamma = d_gamma
    dbeta = d_beta
    dx = d_X
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
    X_cent, var, sd, X_num, den, X_hat, X_gamma, gamma, eps = cache
    d_X_out = dout
    N, D = dout.shape

    dbeta = np.sum(d_X_out, axis=0) # Upstream gradient summed over each observation.
    dgamma = np.sum(d_X_out * X_hat, axis=0)

    d_X_hat = d_X_out*gamma  #intermediate calculation
    dx = (1/(N*sd) * 
           (N* d_X_hat - np.sum(d_X_hat, axis=0) - X_hat * np.sum(d_X_hat*X_hat, axis=0))
          )
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
    eps = ln_param.get('eps', 1e-5)
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
    X = x.T
    mu = np.mean(X, axis=0)                        # 1.
    X_cent = X - mu                                # 2.
    X_num = X_cent                                 # 2.
    X_centsq = X_cent ** 2                         # 3.
    var = np.mean(X_centsq, axis=0)                # 4.
    sd = np.sqrt(var + eps)                        # 5.
    den = 1 / sd                                   # 6.
    X_hat = X_num * den                            # 7.
    X_gamma = gamma.reshape(-1,1) * X_hat          # 8.
    X_out = X_gamma + beta.reshape(-1,1)           # 9.

    # Stuff to return
    out = X_out.T
    cache = (X_cent, var, sd, X_num, den, X_hat, X_gamma, gamma, eps)
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
    X_cent, var, sd, X_num, den, X_hat, X_gamma, gamma, eps = cache

    d_X_out = dout.T
    N, D = d_X_out.shape

    # Batch norm layer - backward pass
    d_beta = np.sum(d_X_out, axis = 1)                # 9.
    d_X_gamma = d_X_out                               # 9.
    d_X_hat = gamma.reshape(-1,1) * d_X_gamma         # 8.
    d_gamma = np.sum(X_hat * d_X_gamma, axis=1)       # 8.
    d_X_num = den * d_X_hat                           # 7.
    d_den = np.sum(X_num * d_X_hat, axis=0)           # 7.
    d_sd = -1. / sd**2 * d_den                        # 6.
    d_var = 0.5 * 1 / np.sqrt(var + eps) * d_sd       # 5.
    d_X_centsq = 1 / N * np.ones_like(d_var) * d_var  # 4.
    d_X_cent = 2 * X_cent * d_X_centsq                # 3.
    d_X_1 = d_X_cent + d_X_num                        # 2.
    d_mu  = -1 * np.sum(d_X_cent + d_X_num, axis=0)   # 2.
    d_X_2 = 1 / N * np.ones_like(d_X_out) * d_mu      # 1.
    d_X = (d_X_1 + d_X_2)                             # 0.

    dgamma = d_gamma
    dbeta = d_beta
    dx = d_X
    
    
    # X_cent, var, sd, X_num, den, X_hat, X_gamma, gamma, eps = cache
    # d_X_out = dout.T
    # N, D = d_X_out.shape

    # dbeta = np.sum(d_X_out, axis=1) # Upstream gradient summed over each observation.
    # dgamma = np.sum(d_X_out * X_hat, axis=1)

    # d_X_hat = d_X_out*gamma.reshape(-1,1)  #intermediate calculation
    # dx = (1/(N*sd) * 
    #     (N* d_X_hat - np.sum(d_X_hat, axis=0) - X_hat * np.sum(d_X_hat*X_hat, axis=0))
    #     )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx.T, dgamma, dbeta


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
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
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
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
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
    # Get parameters
    n, channels, height, width = x.shape
    k_filters, channels, f_spatial_ext, _ = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    images_convoluted= list()
    results = list()
    w_n = w.reshape(w.shape[0], -1)
    b = b.reshape(-1,1)


    check_size = (((width-f_spatial_ext+2*pad)/stride+1) * 1.)
    if not check_size.is_integer():
        print("Neurons do not fit.")

    # Pad input data with zeros.
    x_padded = np.pad(x, 
                  pad_width=((0,0), (0,0),(pad,pad), (pad,pad)), 
                  mode="constant", constant_values=0)

    height_padded = height + 2 * pad
    width_padded = width + 2 * pad

    # Convolve input data
    for ix in range(n):
        convolution = list()
        for w_ix in range(0, width_padded-f_spatial_ext+1, stride):
            for h_ix in range(0, height_padded-f_spatial_ext+1, stride):
                convolution.append(x_padded[ix,:, h_ix:h_ix+f_spatial_ext, w_ix:w_ix+f_spatial_ext].flatten())
        x_col = np.stack(convolution, axis=0).T
        images_convoluted.append(x_col)

    h_out = int(1 + (height + 2 * pad - f_spatial_ext) / stride)
    w_out = int(1 + (width + 2 * pad - f_spatial_ext) / stride)

    for img in images_convoluted:
        tmp = np.transpose((w_n.dot(img)+b).reshape(k_filters, h_out, w_out), 
                           axes=[0,2,1])
        results.append(tmp)
        
    out = np.stack(results, axis=0)
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
    x, w, b, conv_param = cache

    # Get parameters
    n, channels, height, width = x.shape
    k_filters, channels, f_spatial_ext, _ = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    
    # Pad x data
    x_padded = np.pad(x, 
                pad_width=((0,0), (0,0),(pad,pad), (pad,pad)), 
                mode="constant", constant_values=0)

    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)
    for img_ix in range(n):
        for filter_ix in range (k_filters):
            for channel_ix in range(channels):
                x_col = convolute_x(x_padded[img_ix, channel_ix], f_spatial_ext, stride).T  
                dw[filter_ix, channel_ix] += np.dot(dout[img_ix, filter_ix].reshape(1,-1), x_col).reshape(dw[filter_ix, channel_ix].shape)
        db += np.sum(dout[img_ix].reshape(k_filters,-1), axis=1).reshape(-1,1)

    # preparation to get dx
    pad_dout = pad
    dout_padded = np.pad(dout, pad_width=((0,0), (0,0), (pad_dout,pad_dout), (pad_dout,pad_dout )), mode="constant", constant_values=0)

    # get derivative with respect to x
    for img_ix in range(n):
        for filter_ix in range (k_filters):
            for channel_ix in range(channels):
                        
                w_flipped = np.rot90(w[filter_ix, channel_ix], 2)
                dout_padded_conv = convolute_x(x=dout_padded[img_ix, filter_ix,], f_spatial_ext=f_spatial_ext, stride=stride)
                dx[img_ix, channel_ix] += np.dot(w_flipped.reshape(1,-1), dout_padded_conv).reshape(width, height)
    
    dw = dw
    db = db.flatten()
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
    pass
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
    pass
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
    pass
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
    pass
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
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
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
    pass
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
