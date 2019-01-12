import numpy as np
from random import shuffle
from copy import deepcopy


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):  # Loop through training examples
        scores = X[i].dot(W)  # The class score is a dot product between X and W
        correct_class_score = scores[y[i]]  # Score assigned to the correct class
        for j in range(
                num_classes):  # Compute hinge loss by looping though classes
            if j == y[i]:  # no need to compute loss for correct class
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:  # we only add to the loss if margin exceed 0
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, targets, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """

    # Initialise
    delta = 1
    total_loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    N = X.shape[0]

    # Compute scores, margins and loss
    score_mat = X.dot(W)
    correct_class_scores = score_mat[range(N), targets].reshape(N, 1)
    margins = score_mat - correct_class_scores + delta
    margins[range(N), targets] = 0
    margins = np.clip(margins, 0, None)
    data_loss = np.mean(np.sum(margins, axis=1))
    regularization_loss = 0.5 * reg * np.sum(W * W)
    total_loss = data_loss + regularization_loss

    # Compute dL/dW
    mask = deepcopy(margins)
    mask[mask > 0] = 1
    row_sum = np.sum(mask, axis=1)
    mask[range(N), targets] -= row_sum.T
    dW = np.dot(X.T, mask) / N
    dW += reg * W

    return total_loss, dW
