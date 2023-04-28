"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights
#         print("X shape",X_train.shape)
        
#         print("w shape", self.w.shape)
#         print("y_train shape", y_train.shape)
        
        # Convert labels to one-hot encoding
        y_train_onehot = np.zeros((N,self.w.shape[0] ))
        y_train_onehot[np.arange(N), y_train] = 1
        
        
        for epoch in range(self.epochs):
        # Compute the predictions
            y_pred = np.dot(X_train, self.w.T)

            # Compute the gradient
#             print("X train shape", X_train.T.shape)
#             print("y train one hot shape", y_train_onehot.shape)
#             print("y pred shape", y_pred.shape)
#             print("self weight decay", self.weight_decay)
#             y_pred_final = np.argmax(y_pred)
  
            gradient = -2 * np.dot(X_train.T, y_train_onehot - y_pred) / N
            gradient += 2 * self.weight_decay * self.w.T
            
#             print("gradient shape", gradient.shape)
            # Update the weights
            self.w -= self.lr * gradient.T

#         TODO: implement me

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
        # TODO: implement me
        y_pred = np.dot(X_test, self.w.T)
        return np.argmax(y_pred, axis=1)
