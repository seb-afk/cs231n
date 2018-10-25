from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

def softmax(x):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    return probs

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform, followed by a batch
    normalisation layer, followed by a ReLU.

    Parameters
    ----------
    x : Numpy array (N, D)
        Input to the affine layer
    w : Numpy array
        Weights for the affine layer
    b : Numpy array
        Bias terms for the affine layer
    gamma : Numpy array (D,)
        Scaling parameter.
    beta : Numpy array (D,)
        Bias parameter.
    bn_param: Dictionary with the following keys:
        mode: 'train' or 'test'; required
        eps: Constant for numeric stability
        momentum: Constant for running mean / variance.
        running_mean: Array of shape (D,) giving running mean of features
        running_var Array of shape (D,) giving running variance of features

    Returns
    -------
    out : Numpy array
        Output from the ReLU.
    cache : Tuple (affine, batch-norm, relu)
        Tuple containing the cached items for each layer (affine, batch-norm, relu)
        to give to the backwards pass.
    """
    y_affine, fc_cache = affine_forward(x, w, b)
    y_bn, bn_cache = batchnorm_forward(y_affine, gamma, beta, bn_param)
    out, relu_cache = relu_forward(y_bn)

    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Convenience layer that performs the backward pass for an affine - bn - relu
    block.

    Parameters
    ----------
    dout : Numpy array
        Upstream gradient
    cache : Tuple (fc_cache, bn_cache, relu_cache)
        Tuple containing the cached items for each layer (affine, batch-norm, relu)
        to give to the backwards pass

    Returns
    -------
    dx, dw, db, dgamma, dbeta : Numpy array
        Gradients with respect to layer parameters and inputs.
    """
    fc_cache, bn_cache, relu_cache = cache
    dy_bn = relu_backward(dout, relu_cache)
    dy_affine, dgamma, dbeta  = batchnorm_backward(dy_bn, bn_cache)
    dx, dw, db = affine_backward(dy_affine, fc_cache)
    
    return dx, dw, db, dgamma, dbeta

def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    Convenience layer that performs an affine transform, followed by a layer
    normalisation layer, followed by a ReLU.

    Parameters
    ----------
    x : Numpy array (N, D)
        Input to the affine layer
    w : Numpy array
        Weights for the affine layer
    b : Numpy array
        Bias terms for the affine layer
    gamma : Numpy array (D,)
        Scaling parameter.
    beta : Numpy array (D,)
        Bias parameter.
    ln_param: Dictionary with the following keys:
        eps: Constant for numeric stability

    Returns
    -------
    out : Numpy array
        Output from the ReLU.
    cache : Tuple (affine, layer-norm, relu)
        Tuple containing the cached items for each layer (affine, batch-norm, relu)
        to give to the backwards pass.
    """
    y_affine, fc_cache = affine_forward(x, w, b)
    y_bn, ln_cache = layernorm_forward(y_affine, gamma, beta, ln_param)
    out, relu_cache = relu_forward(y_bn)

    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache

def affine_ln_relu_backward(dout, cache):
    """
    Convenience layer that performs the backward pass for an affine - ln - relu
    block.

    Parameters
    ----------
    dout : Numpy array
        Upstream gradient
    cache : Tuple (fc_cache, ln_cache, relu_cache)
        Tuple containing the cached items for each layer (affine, layer-norm, relu)
        to give to the backwards pass

    Returns
    -------
    dx, dw, db, dgamma, dbeta : Numpy array
        Gradients with respect to layer parameters and inputs.
    """
    fc_cache, ln_cache, relu_cache = cache
    dy_ln = relu_backward(dout, relu_cache)
    dy_affine, dgamma, dbeta  = layernorm_backward(dy_ln, ln_cache)
    dx, dw, db = affine_backward(dy_affine, fc_cache)
    
    return dx, dw, db, dgamma, dbeta

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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
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
        # self.params = {}
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
        self.params = {'W1': weight_scale * np.random.randn(input_dim, hidden_dim),
                'b1': np.zeros(hidden_dim),
                'W2': weight_scale * np.random.randn(hidden_dim, num_classes),
                'b2': np.zeros(num_classes)}
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
        
        # Get parameters
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        # Forward pass
        Z1, cache_z1 = affine_forward(X, W1, b1)
        A1, cache_a1 = relu_forward(Z1)

        Z2, cache_z2 = affine_forward(A1, W2, b2)
        scores = Z2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        reg = self.reg
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
        # Compute loss
        data_loss, da2 = softmax_loss(Z2, y)
        regularization_loss = 0.5 * (reg * np.sum(W1*W1) + reg * np.sum(W2*W2))
        loss = data_loss + regularization_loss


        # Compute gradients
        da1, dw2, db2 = affine_backward(da2, cache_z2)
        dz1 = relu_backward(da1,cache_a1)
        dx, dw1, db1 = affine_backward(dz1, cache_z1)

        # Add regularization term
        dw1 += W1 * reg
        dw2 += W2 * reg

        # Updates gradients dictionary
        grads["W1"] = dw1
        grads["b1"] = db1
        grads["W2"] = dw2
        grads["b2"] = db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

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

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
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

        self.params = {}
        dims = [input_dim] + hidden_dims + [num_classes]
        for layer_index in range(1, self.num_layers+1):
            l_size = dims[layer_index]
            l_size_prev = dims[layer_index-1]
            self.params["W"+str(layer_index)] = weight_scale * np.random.randn(l_size_prev, l_size)
            self.params["b"+str(layer_index)] = np.zeros(l_size)

        if (normalization == "batchnorm") or (normalization == "layernorm"):
            for layer_index in range(1, self.num_layers):
                l_size = dims[layer_index]
                # Initialise batchnorm layers
                self.params["gamma"+str(layer_index)] = np.ones(l_size)
                self.params["beta"+str(layer_index)] = np.zeros(l_size)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
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
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
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
        
        activation = X
        cache_list = list()
        cache_dropout = list()

        # (AFFINE -> [BatchNorm | Layernorm] -> RELU -> [Dropout]) * (L-1)
        for layer_index in range(1, self.num_layers):
            activation_prev = activation
            
            if self.normalization == None:
                activation, cache = affine_relu_forward(
                    activation_prev, 
                    self.params["W"+str(layer_index)],
                    self.params["b"+str(layer_index)])
            
            elif self.normalization == "batchnorm":
                activation, cache = affine_bn_relu_forward(
                    activation_prev, 
                    self.params["W"+str(layer_index)],
                    self.params["b"+str(layer_index)],
                    self.params["gamma"+str(layer_index)],
                    self.params["beta"+str(layer_index)],
                    self.bn_params[layer_index-1])

            elif self.normalization == "layernorm":
                activation, cache = affine_ln_relu_forward(
                    activation_prev, 
                    self.params["W"+str(layer_index)],
                    self.params["b"+str(layer_index)],
                    self.params["gamma"+str(layer_index)],
                    self.params["beta"+str(layer_index)],
                    self.bn_params[layer_index-1])

            if self.use_dropout:
                activation, do_cache = dropout_forward(activation, self.dropout_param)
                cache_dropout.append(do_cache)

            cache_list.append(cache)

        # [AFFINE->SOFTMAX]
        scores, cache = affine_forward(activation,
                                self.params["W"+str(self.num_layers)],
                                self.params["b"+str(self.num_layers)])
        cache_list.append(cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
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
        # Compute loss
        data_loss, dout = softmax_loss(scores, y)
        regularization_loss_total = 0
        for layer_index in range(1, self.num_layers+1):
            regularization_loss_layer = .5 * self.reg * np.sum(self.params["W"+str(layer_index)]**2)
            regularization_loss_total += regularization_loss_layer
        loss = data_loss + regularization_loss_total

        # Compute gradients
        
        # Last layer
        dout, dw, db = affine_backward(dout, cache)
        grads["W"+str(layer_index)] = dw + self.reg * cache[1]
        grads["b"+str(self.num_layers)] = db

        # (AFFINE -> [BatchNorm] -> RELU) * (L-1) 
        for layer_index in reversed(range(1, self.num_layers)):
            if self.use_dropout:
                do_cache = cache_dropout[layer_index-1]
                dout = dropout_backward(dout, do_cache)

            cache = cache_list[layer_index-1]
            if self.normalization == None:
                dout, dw, db = affine_relu_backward(dout, cache)

            elif self.normalization == "batchnorm":
                dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache)
                grads["gamma"+str(layer_index)] = dgamma
                grads["beta"+str(layer_index)] = dbeta

            elif self.normalization == "layernorm":
                dout, dw, db, dgamma, dbeta = affine_ln_relu_backward(dout, cache)
                grads["gamma"+str(layer_index)] = dgamma
                grads["beta"+str(layer_index)] = dbeta

            grads["W"+str(layer_index)] = dw + self.reg * cache[0][1]
            grads["b"+str(layer_index)] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
