import numpy as np
from random import shuffle


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

    # Get the number of classes and number of examples
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # Loop through training examples
    for i in range(num_train):

        # Compute the probabilities of each class
        logits = X[i].dot(W)
        logits -= np.max(logits) # To avoid numerical issues
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Compute the loss
        current_target = y[i]
        loss += -np.log(probs[current_target])

        # Compute the gradient
        for class_i in range(num_classes):
            if class_i == current_target:
                dW[:, class_i] += X[i] * (probs[class_i] - 1)
            else:
                dW[:, class_i] += X[i] * probs[class_i]

    # Add regularization term
    loss += 0.5*reg*np.sum(W*W)
    dW  += W * reg

    # Average loss, average gradient
    # loss = loss / num_train
    dW = dW / num_train

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Get the number of classes and number of examples
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # Compute probs
    logits = X.dot(W)
    logits -= np.amax(logits, axis=1)[:, None]
    normalisation_term = np.sum(np.exp(logits), axis=1)
    probs = np.exp(logits) / normalisation_term[:,None]

    # Compute loss
    loss = np.sum(-np.log(probs[range(num_train), y]))

    # Compute gradient
    probs[range(num_train), y] -= 1
    dW = np.dot(X.T, probs)

    # Add regularization term
    loss += 0.5 * reg * np.sum(W * W)
    dW += W * reg

    # Average loss, average gradient
    # loss = loss / num_train
    dW = dW / num_train

    return loss, dW
