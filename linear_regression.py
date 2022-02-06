import numpy as np


class LinearRegression:
    """
    A linear regression model

    """
    def __init__(self, dimension):
        """
        Initialization of the model

        Parameters
        ----------
        dimension : int
            dimension of the model
        """
        self.bias = None
        self.weights = None
        self.dimension = dimension
        self.initialize_weights()

    def initialize_weights(self):
        """
        Sets all the weights and bias to zeros

        Returns
        -------

        """
        self.weights = np.zeros((1, self.dimension))
        self.bias = 0

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

    def compute_cost(self, x, y):
        """
        calculates the output of the model and the cost of output compares to desired output

        Parameters
        ----------
        x : nd.array
            input vector
        y : nd.array
            label vector

        Returns
        -------
        sigma : float
            output
        cost : float
            the cost of the output compares to the label vector
        """
        sigma = self.sigmoid(np.dot(self.weights, x.T) + self.bias)
        cost = - (1 / x.shape[0]) * np.sum(np.dot(y, np.log(sigma).T) + np.dot((1 - y), np.log(1 - sigma).T))
        return sigma, cost

    def gradient_descent(self):
        pass

    def update_weights(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass
