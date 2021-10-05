from abc import ABC, abstractmethod
import numpy as np


class BaseMLP(ABC):
    """
    Base class for MLP classification.
    """
    __slots__ = ('batch_size', 'learning_rate', 'max_iter', 'hidden_layer_sizes', 'random_state', 'momentum')

    def __init__(self, batch_size, learning_rate, max_iter, hidden_layer_sizes, random_state, momentum):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.momentum = momentum

    @abstractmethod
    def fit(self, x, y) -> None:
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        x : list or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : list of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels for classification).

        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    @abstractmethod
    def predict(self, x):
        """Predict using the multi-layer perceptron classifier

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : list, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        raise NotImplementedError("Method not implemented")