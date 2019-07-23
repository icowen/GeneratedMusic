import numpy as np
from enum import Enum


def normalize(output_probabilities):
    total = sum(output_probabilities)

    def get_new_p(p):
        return p / total

    return list(map(get_new_p, output_probabilities))


def log(y_predicted, *args):
    return np.log(y_predicted)


def log_derivative(y_predicted, y_true, is_correct_index, sum_of_output_activations):
    if is_correct_index:
        return -(sum_of_output_activations - y_predicted) / \
               (sum_of_output_activations * y_predicted)
    else:
        return 1 / sum_of_output_activations


def mean_squared_error_derivative(y_predicted, y_true, *args):
    return -2 * (y_true - y_predicted)


def mean_squared_error(y_predicted, y_true, *args):
    return (y_true - y_predicted) ** 2


def log_loss_derivative(y_predicted, y_true, *args):
    if y_true == 1:
        return -1 / y_predicted
    else:
        return 1 / (1 - y_predicted)


def logloss(y_predicted, y_true, *args):
    if y_true == 1:
        return -np.log(y_predicted)
    else:
        return -np.log(1 - y_predicted)


def mse_loss(y_predicted, y_true, *args):
    return ((y_true - y_predicted) ** 2).mean()


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ActivationFunction(Enum):
    LOG = (log, log_derivative)
    MEAN_SQUARED_ERROR = (mean_squared_error, mean_squared_error_derivative)
    LOG_LOSS = (logloss, log_loss_derivative)

    def __init__(self, loss_function, derivative):
        self.loss_function = loss_function
        self.derivative = derivative
