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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  ##writing code here
  #(loss = -log(exp(fi) / sumj(exp(fj))))
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    #fi to mean the i-th element of the vector of class scores
    fi = np.dot(X[i,:], W)
    # to solve Numeric instability
    log_c = np.max(fi)
    fi = fi - log_c
    sumj = np.sum(np.exp(fi))
    pk = np.exp(fi) / sumj
    loss = loss - np.log(pk[y[i]])

    #computing the gradient
    for k in range(num_classes):
      if k == y[i]:
          dW[:,k] = dW[:,k] + (pk[k] - 1) * X[i]
      else:
        dW[:,k] = dW[:,k] + (pk[k]) * X[i]    

  #regularization and normalization
  #done by compute average then regularize by adding reg_term
  #R(W)--> regularization term = sum(W*W)
  loss = loss / num_train + 0.5*reg*np.sum(W*W)
  dW = dW / num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]

  f = np.dot(X, W)
  log_c = np.max(f, axis = 1, keepdims = True)
  f = f - log_c
  sumf = np.sum(np.exp(f), axis=1, keepdims = True)
  pk = np.exp(f) / sumf
  all_logs = -np.log(pk[np.arange(num_train), y])
  loss = np.sum(all_logs)
  loss = loss / num_train + 0.5*reg*np.sum(W*W)

  dpk = np.zeros_like(pk)
  dpk[np.arange(num_train), y] = 1
  dW = np.dot(X.T, pk-dpk)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

