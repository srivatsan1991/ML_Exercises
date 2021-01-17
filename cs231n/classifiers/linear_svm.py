from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for I in range(num_train):
        scores = X[I].dot(W)
        correct_class_score = scores[y[I]]
        for b in range(num_classes):
            if b == y[I]:
                continue
            
            margin = scores[b] - correct_class_score + 1 # note delta = 1
            
            if margin > 0:
                loss += margin
                dW[:, b] += (X[I, :]/num_train)
                dW[:, y[I]] -= (X[I, :]/num_train)


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += (2 * reg * W)
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    scores = np.dot(X, W)
    
    correct_scores = np.array([scores[I, y[I]] for I in range(num_train)]).reshape((num_train,1))

    lm = scores - correct_scores + 1.0  # a matrix of loss values where the positive entries need to be summed

    # a bit mask for the locations with positive values
    mask = np.array(lm)
    mask[mask > 0] = 1
    mask[mask < 0] = 0

    # a bit mask where location (I,alpha) is nonzero iff y[I]=alpha
    correct_mask = np.zeros(lm.shape)
    for I in range(num_train):
        correct_mask[I,y[I]] = 1
    

    # set the negative elements to zero in the loss matrix
    lm[lm < 0] = 0
    loss = (np.sum(lm)/num_train) - 1.0  # the subtraction of 1.0 is for the diagonal elements of lm
    
    dW += np.dot(np.transpose(X), mask)/num_train
    dW -= np.dot(np.multiply(np.transpose(X), np.sum(mask, axis=1)), correct_mask)/num_train

    dW += 2 * reg * W

    return loss, dW
