import numpy as np


class LinearRegression:
    """
    A linear regression model

    """
    def __init__(self, dimension, iterations, learning_rate):
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
        self.iterations = iterations
        self.learning_rate = learning_rate
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

    def gradient_descent(self, x, y, sigma):
        """
        calculates the gradients of the weights with respect to the loss of the model

        Parameters
        ----------
        x : nd.array
            input vector
        y : nd.array
            label vector
        sigma : nd.array
            model output

        Returns
        -------
        gradients for the weights and the bias
        """
        dw = 1/x.shape[0] * np.dot((sigma - y), x)
        db = 1 / x.shape[0] * np.sum(sigma - y)
        return dw, db

    def update_weights(self, x, y):
        """
        Updates the weights of the model

        Parameters
        ----------
        x : nd.array
            input vector
        y : nd.array
            label vector

        Returns
        -------
        cost : float
            returns the cost of the current performance of the model
        """
        sigma, cost = self.compute_cost(x, y)
        dw, db = self.gradient_descent(x, y, sigma)
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db
        return cost

    def fit(self, x, y):
        pass

    def predict(self, x):
        """
        Makes a predictions on a nd.array based on the weights and the bias of the model

        Parameters
        ----------
        x : nd.array
            input vector

        Returns
        -------
        predictions : nd.array
            return the predictions of the model
        """
        sigma = self.sigmoid(np.dot(self.weights, x.T) + self.bias)
        predictions = np.around(sigma)
        return predictions
