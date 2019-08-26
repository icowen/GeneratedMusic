import numpy as np
from enum import Enum
import re
from string import ascii_lowercase


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


def letter_dict(index):
    letters_dict = dict()
    for c in ascii_lowercase:
        letters_dict[get_index_for_char_array(c)] = c
    if index in letters_dict.keys():
        return letters_dict[index]
    else:
        return ' '


def get_index_for_char_array(char):
    return 26 if char == ' ' else ord(char) - 97


def convert_char(char):
    c = np.zeros((27,), dtype=int)
    c[get_index_for_char_array(char)] = 1
    return c


def convert_list_to_ascii_by_highest_probability(letter_list):
    index = np.argmax(letter_list)
    return convert_index_number_to_ascii_letter(index)


def convert_index_number_to_ascii_letter(index):
    if index == 26:
        return '\' \''
    return chr(index + 97)


def clean_word_list(word_list):
    clean = []
    for line in word_list:
        for word in line.split():
            word = re.sub('[^a-zA_Z]', '', word.lower())
            word = re.sub('([\s])', ' ', word)
            clean.append(word)
    return clean


