import numpy as np


class LinearRegression:
    """
    A linear regression model

    """
    def __init__(self, dimension):
        self.dimension = dimension
        self.initialize_weights()

    def initialize_weights(self):
        weights = np.zeros((1, self.dimension))
        bias = 0

    # noinspection PyMethodMayBeStatic
    def sigmoid(self, x):
        """
        Sigmoid function : 1/(1+e^-x)

        Parameters
        ----------
        x : float or nd.array
            input
        Returns
        -------
        float or nd.array
            the result of the function applied to the input
        """
        s = 1.0 / (1.0 + np.exp(-x))
        return s

    def compute_cost(self):
        pass

    def gradient_descent(self):
        pass

    def update_weights(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass
