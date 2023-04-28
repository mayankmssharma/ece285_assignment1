"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_
            train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        y = -np.ones((N, self.n_class), dtype=int)
        y[np.arange(N), y_train] = 1
#         print(y)
        self.w = weights
        for epoch in range(self.epochs):
            
            
            sig = self.sigmoid(- y.T *  (self.w @ X_train.T))
            gradient = -(( sig * y.T) @ X_train)/N
            gradient = gradient +   (self.weight_decay * self.w)

            self.w = self.w - self.lr * gradient
#             y_hat = self.sigmoid(-y.T*(self.w@X_train.T))
            
#             # Gradient
#             dw = ((y_hat*y.T)@X_train)/N
            
#             # Update with regularization
#             self.w = self.w + self.lr * ((2*(self.weight_decay*self.w)) + dw)
        
        
        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        return np.argmax(self.sigmoid(self.w.dot(X_test.T)), axis=0)