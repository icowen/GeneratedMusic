import random
import numpy as np
from sklearn.metrics import log_loss


class Weight:
    def __init__(self, weight, from_neuron):
        self.weight = weight
        self.from_neuron = from_neuron
        self.to_neuron = None
        self.change = 0

    def __str__(self):
        return f'WEIGHT--- Weight: {self.weight}, ' \
               f'From neuron: {self.from_neuron.id if self.from_neuron else "none" }, ' \
               f'To Neuron: {self.to_neuron.id if self.to_neuron else "none"}'

    def set_to_neuron(self, to_neuron):
        self.to_neuron = to_neuron

    def set_weight(self, weight):
        self.weight = weight

    def add_change(self, change):
        self.change = change

    def update(self):
        self.weight = self.weight - self.change * .01


class Neuron:
    def __init__(self, bias, layer, position_in_layer):
        self.bias = bias
        self.layer = layer
        self.position_in_layer = position_in_layer
        self.id = f'Layer: {self.layer} Position: {self.position_in_layer}'
        self.activation = 0
        self.sum = 0
        self.change_in_bias = 0
        self.change_in_activation = 0
        self.weights = {'weights_from_this_neuron': [],
                        'weights_to_this_neuron': []}

    def __str__(self):
        from_weights = ''
        to_weights = ''
        for weight in self.weights["weights_from_this_neuron"]:
            from_weights += f'{weight.weight}  '
        for weight in self.weights["weights_to_this_neuron"]:
            to_weights += f'{weight.weight}  '
        weight_str = f'Starts here: {from_weights if from_weights else "None"} ' \
                     f'Ends here: {to_weights if to_weights else "None"}'
        return f'NEURON--- Layer: {self.layer}, ' \
               f'Position: {self.position_in_layer}, ' \
               f'Activation: {self.activation}, ' \
               f'Bias: {self.bias}, ' \
               f'Weights: {weight_str}'

    def set_activation(self, activation):
        self.activation = activation

    def set_sum(self, sum_value):
        self.sum = sum_value

    def set_bias(self, bias):
        self.bias = bias

    def get_activation(self):
        return self.activation

    def get_bias(self):
        return self.bias

    def get_sum(self):
        return self.sum

    def get_layer(self):
        return self.layer

    def get_position_in_layer(self):
        return self.position_in_layer

    def get_change_in_bias(self):
        return self.change_in_bias

    def get_change_in_activation(self):
        return self.change_in_activation

    def add_weight_to_this_neuron(self, weight):
        self.weights['weights_to_this_neuron'].append(weight)

    def add_weight_from_this_neuron(self, weight):
        self.weights['weights_from_this_neuron'].append(weight)

    def add_change_in_bias(self, change_in_bias):
        self.change_in_bias = change_in_bias

    def add_change_in_activation(self, change_in_activation):
        self.change_in_activation = change_in_activation

    def update(self):
        self.bias = self.bias - self.change_in_bias * .01


class NeuralNet:
    def __init__(self,
                 number_of_input_neurons: int,
                 number_of_layers: int,
                 number_of_neurons_per_layer: int,
                 number_of_output_neurons: int = 1,
                 hidden_neuron_biases: list = None,
                 output_neuron_biases: list = None,
                 weights=None):
        self.neurons = []
        self.weights = []
        self.number_of_layers = number_of_layers
        self.number_of_output_neurons = number_of_output_neurons
        self.number_of_neurons_per_layer = number_of_neurons_per_layer
        self.number_of_neurons = number_of_input_neurons + \
                                 (number_of_layers - 2) * number_of_neurons_per_layer + \
                                 number_of_output_neurons
        self.number_of_weights = number_of_input_neurons * number_of_neurons_per_layer + \
                                 (number_of_layers - 3) * number_of_neurons_per_layer + \
                                 number_of_neurons_per_layer * number_of_output_neurons

        self.set_input_neurons(number_of_input_neurons)
        self.set_hidden_neurons(hidden_neuron_biases)
        self.set_output_neurons(output_neuron_biases)

        self.set_weight_connections(weights)

        self.print_net()

    def set_input_neurons(self, number_of_input_neurons):
        input_neurons = []
        for n in range(number_of_input_neurons):
            neuron = Neuron(random.random(), 1, n)
            input_neurons.append(neuron)
        self.neurons.append(input_neurons)

    def set_hidden_neurons(self, hidden_neuron_biases):
        self.check_for_correct_number_of_hidden_neurons(hidden_neuron_biases)
        i = 0
        for layer_number in range(self.number_of_layers - 2):
            layer = []
            for neuron_position in range(self.number_of_neurons_per_layer):
                bias = hidden_neuron_biases[i] if hidden_neuron_biases else random.random()
                i += 1
                neuron = Neuron(bias, layer_number + 2, neuron_position)
                layer.append(neuron)
            self.neurons.append(layer)

    def set_output_neurons(self, output_neuron_biases):
        self.check_for_correct_number_of_output_neurons(output_neuron_biases)
        layer = []
        for neuron_position in range(self.number_of_output_neurons):
            bias = output_neuron_biases[neuron_position] if output_neuron_biases else random.random()
            neuron = Neuron(bias, self.number_of_layers, neuron_position)
            layer.append(neuron)
        self.neurons.append(layer)

    def set_weight_connections(self, weights):
        self.check_for_correct_number_of_weights(weights)
        i = 0
        for layer_number in range(self.number_of_layers):
            for neuron in self.neurons[layer_number]:
                if layer_number + 1 < self.number_of_layers:
                    next_layer = self.neurons[layer_number + 1]
                    for neuron_in_next_layer in next_layer:
                        weight_value = weights[i] if weights else random.random()
                        i += 1
                        weight = Weight(weight_value, neuron)
                        weight.set_to_neuron(neuron_in_next_layer)
                        neuron.add_weight_from_this_neuron(weight)
                        neuron_in_next_layer.add_weight_to_this_neuron(weight)
                        self.weights.append(weight)

    def check_for_correct_number_of_output_neurons(self, output_neuron_biases):
        if output_neuron_biases and not len(output_neuron_biases) == self.number_of_output_neurons:
            raise ValueError(f'number of biases for hidden nodes ({len(output_neuron_biases)}) '
                             f'is not the same length as number of '
                             f'hidden nodes ({self.number_of_output_neurons})')

    def check_for_correct_number_of_hidden_neurons(self, hidden_neuron_biases):
        number_of_hidden_neurons = self.number_of_neurons_per_layer * (self.number_of_layers - 2)
        if hidden_neuron_biases and not len(hidden_neuron_biases) == number_of_hidden_neurons:
            raise ValueError(f'number of biases for hidden nodes ({len(hidden_neuron_biases)}) '
                             f'is not the same length as number of hidden nodes ({number_of_hidden_neurons})')

    def check_for_correct_number_of_weights(self, weights):
        if weights and not len(weights) == self.number_of_weights:
            raise ValueError(f'number of given weights ({len(weights)}) is not the same length as '
                             f'number of connections between neurons ({self.number_of_weights})')

    def get_weights(self):
        weights = []
        for weight in self.weights:
            weights.append(weight.weight)
        return weights

    def get_biases(self):
        biases = []
        for layer in self.neurons[1:]:
            for neuron in layer:
                biases.append(neuron.get_bias())
        return biases

    def print_net(self):
        for layer in self.neurons:
            print(f'\n-------------------Layer: {self.neurons.index(layer) + 1}-------------------')
            for neuron in layer:
                print(neuron)
                for weight in neuron.weights["weights_from_this_neuron"]:
                    print(weight)

    @staticmethod
    def get_new_sum_and_activation(neuron):
        new_sum_value = 0
        for weight in neuron.weights['weights_to_this_neuron']:
            activation = weight.from_neuron.get_activation()
            new_sum_value += weight.weight * activation
        new_sum_value += neuron.get_bias()
        neuron.set_sum(new_sum_value)
        neuron.set_activation(sigmoid(new_sum_value))

    def predict(self, input_data):
        for i in range(len(input_data)):
            self.neurons[0][i].set_activation(input_data[i])
        for layer in self.neurons:
            if self.neurons.index(layer) != 0:
                for neuron in layer:
                    self.get_new_sum_and_activation(neuron)
        y_predicted = max(neuron.get_activation() for neuron in self.get_output_layer())
        output = []
        prediction = 0
        for neuron in self.get_output_layer():
            output.append(neuron.get_activation())
        print(f'Output probability vector: {output}')
        return y_predicted

    def train(self, input_data: np.ndarray, output_data: np.ndarray, epochs=10000):
        print(f'\n\n-------------------------------TRAINING NET FOR {epochs} TIMES-------------------------------')
        self.check_for_valid_input(input_data, output_data)
        for epoch in range(epochs):
            for x, y in zip(input_data, output_data):
                for i in range(len(x)):
                    self.neurons[0][i].set_activation(x[i])
                for layer in self.neurons:
                    if self.neurons.index(layer) != 0:
                        for neuron in layer:
                            self.get_new_sum_and_activation(neuron)

                correct_index = 0
                for i in range(len(y)):
                    if y[i] == 1:
                        correct_index = i
                output_neuron = self.get_output_layer()[correct_index]
                y_predicted = output_neuron.get_activation()

                # derivative_of_loss = log_derivative(y_predicted)
                # derivative_of_loss = mean_squared_error_derivative(y[correct_index], y_predicted)
                derivative_of_loss = log_loss_derivative(y[correct_index], y_predicted)

                output_probabilities = []
                for output_neuron in self.get_output_layer():
                    output_neuron_activation = output_neuron.get_activation()
                    output_probabilities.append(output_neuron_activation)

                    output_neuron.add_change_in_activation(derivative_of_loss)
                    change_in_bias = derivative_of_loss * sigmoid_derivative(output_neuron.get_sum())
                    output_neuron.add_change_in_bias(change_in_bias)
                    self.calculate_partial_derivatives_for_weights(output_neuron)

                # output_neuron = self.neurons[self.number_of_layers - 1][0]
                # y_predicted = output_neuron.get_activation()

                self.calculate_partial_derivatives()
                self.update_net()

            # if epoch % 10 == 0:
            #     y_predictions = np.apply_along_axis(self.predict, 1, input_data)
            #     self.print_loss_function(output_data, y_predictions, epoch)
        self.print_net()

    def update_net(self):
        for layer in self.neurons:
            for neuron in layer:
                for weight in neuron.weights["weights_from_this_neuron"]:
                    weight.update()
                neuron.update()

    def calculate_partial_derivatives(self):
        for layer in self.neurons[:self.number_of_layers - 1]:
            for neuron in layer:
                self.calculate_partial_derivative_for_bias_and_activation(neuron)
                self.calculate_partial_derivatives_for_weights(neuron)

    @staticmethod
    def calculate_partial_derivative_for_bias_and_activation(neuron):
        new_bias_value = sigmoid_derivative(neuron.get_sum())
        bias_values_from_stuff_to_the_right = 0
        dl_da = 0
        for from_weight in neuron.weights["weights_from_this_neuron"]:
            bias_values_from_stuff_to_the_right += from_weight.to_neuron.get_change_in_activation() * \
                                                   sigmoid_derivative(from_weight.to_neuron.get_sum()) * \
                                                   from_weight.weight
            dl_da += from_weight.to_neuron.get_change_in_activation() * \
                     sigmoid_derivative(from_weight.to_neuron.get_sum()) * \
                     from_weight.weight
        neuron.add_change_in_activation(dl_da)
        if bias_values_from_stuff_to_the_right != 0:
            new_bias_value *= bias_values_from_stuff_to_the_right
            neuron.add_change_in_bias(new_bias_value)

    @staticmethod
    def calculate_partial_derivatives_for_weights(neuron):
        for to_weight in neuron.weights["weights_to_this_neuron"]:
            new_weight_value = sigmoid_derivative(neuron.get_sum()) * \
                               to_weight.from_neuron.get_activation() * \
                               neuron.get_change_in_activation()
            to_weight.add_change(new_weight_value)

    @staticmethod
    def print_loss_function(output_data, y_predictions, epoch):
        loss_function = log_loss(output_data, y_predictions)
        # print(f"Epoch {epoch} loss_function: {round(loss_function, 3)}")

    def get_output_layer(self):
        return self.neurons[self.number_of_layers - 1]

    def check_for_valid_input(self, input_data, output_data):
        if not isinstance(input_data, np.ndarray):
            raise TypeError(f'Use a list for input data instead of {type(input_data)}')
        if not len(input_data[0]) == len(self.neurons[0]):
            raise ValueError(f'input_data ({len(input_data[0])}) '
                             f'is not the same length as number of input nodes ({len(self.neurons[0])})')
        if not len(output_data[0]) == len(self.get_output_layer()):
            raise ValueError(f'output_data ({len(output_data[0])}) '
                             f'is not the same length as number of output nodes ({len(self.get_output_layer())})')


def log_derivative(y_predicted):
    return -1 / y_predicted


def mean_squared_error_derivative(y_true, y_predicted):
    return -2 * (y_true - y_predicted)


def log_loss_derivative(y_true, y_predicted):
    if y_true == 1:
        return -1 / y_predicted
    else:
        return 1 / (1 - y_predicted)


def logloss(true_label, predicted_prob):
    if true_label == 1:
        return -np.log(predicted_prob)
    else:
        return -np.log(1 - predicted_prob)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def run_two_neuron_net_for_EQUALS():
    x_test = np.array([[1], [0]])
    y_test = np.array([[1], [0]])
    net = NeuralNet(1, 2, 1)
    net.train(x_test, y_test)
    test = [1]
    test1 = [0]
    test2 = [1]
    test3 = [0]
    print('\n\n-----------PREDICTIONS FOR EQUALS WITH ZERO HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: 1")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: 0")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: 1")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: 0")


def run_two_neuron_net_for_NOT():
    x_test = np.array([[1], [0]])
    y_test = np.array([[0], [1]])
    net = NeuralNet(1, 2, 1)
    net.train(x_test, y_test)
    test = [1]
    test1 = [0]
    test2 = [1]
    test3 = [0]
    print('\n\n-----------PREDICTIONS FOR NOT WITH ZERO HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: 0")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: 1")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: 0")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: 1")


def run_three_neuron_net_with_one_hidden_for_EQUALS():
    x_test = np.array([[1], [0]])
    y_test = np.array([[1], [0]])
    net = NeuralNet(1, 3, 1)
    net.train(x_test, y_test)
    test = [1]
    test1 = [0]
    test2 = [1]
    test3 = [0]
    print('\n\n-----------PREDICTIONS FOR EQUALS WITH 1 HIDDEN NEURON-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: 1")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: 0")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: 1")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: 0")


def run_three_neuron_net_with_one_hidden_for_NOT():
    x_test = np.array([[1], [0]])
    y_test = np.array([[0], [1]])
    net = NeuralNet(1, 3, 1)
    net.train(x_test, y_test)
    test = [1]
    test1 = [0]
    test2 = [1]
    test3 = [0]
    print('\n\n-----------PREDICTIONS FOR NOT WITH ONE HIDDEN NEURON-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: 0")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: 1")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: 0")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: 1")


def run_four_neuron_net_with_two_hidden_for_EQUALS():
    x_test = np.array([[1], [0]])
    y_test = np.array([[1], [0]])
    net = NeuralNet(1, 3, 2)
    net.train(x_test, y_test)
    test = [1]
    test1 = [0]
    test2 = [1]
    test3 = [0]
    print('\n\n-----------PREDICTIONS FOR EQUALS WITH 2 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: 1")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: 0")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: 1")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: 0")


def run_four_neuron_net_with_two_hidden_for_NOT():
    x_test = np.array([[1], [0]])
    y_test = np.array([[0], [1]])
    net = NeuralNet(1, 3, 2)
    net.train(x_test, y_test)
    test = [1]
    test1 = [0]
    test2 = [1]
    test3 = [0]
    print('\n\n-----------PREDICTIONS FOR NOT WITH 2 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: 0")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: 1")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: 0")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: 1")


def run_three_neuron_net_with_zero_hidden_for_AND():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[0], [0], [0], [1]])
    net = NeuralNet(2, 2, 2)
    net.train(x_test, y_test)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR AND WITH ZERO HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {0}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {0}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {1}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_five_neuron_net_with_two_hidden_for_AND():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[0], [0], [0], [1]])
    net = NeuralNet(2, 3, 2)
    net.train(x_test, y_test)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR AND WITH 2 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {0}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {0}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {1}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_three_neuron_net_with_zero_hidden_for_OR():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[1], [0], [1], [1]])
    net = NeuralNet(2, 2, 2)
    net.train(x_test, y_test)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR OR WITH ZERO HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {1}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {1}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {1}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_five_neuron_net_with_two_hidden_for_OR():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[1], [0], [1], [1]])
    net = NeuralNet(2, 3, 2)
    net.train(x_test, y_test)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR OR WITH 2 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {1}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {1}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {1}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_seven_neuron_net_with_four_hidden_for_AND():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[0], [0], [0], [1]])
    net = NeuralNet(2, 3, 4)
    net.train(x_test, y_test)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR AND WITH 4 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {0}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {0}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {1}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_seven_neuron_net_with_four_hidden_for_OR():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[1], [0], [1], [1]])
    net = NeuralNet(2, 3, 4)
    net.train(x_test, y_test)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR OR WITH 4 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {1}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {1}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {1}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_five_neuron_net_with_two_hidden_for_MOD2():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[1], [0], [1], [0]])
    net = NeuralNet(2, 3, 2)
    net.train(x_test, y_test, 20000)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR MOD 2 WITH 2 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {1}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {1}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {0}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def run_seven_neuron_net_with_four_hidden_for_MOD2():
    x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
    y_test = np.array([[1], [0], [1], [0]])
    net = NeuralNet(2, 3, 4)
    net.train(x_test, y_test, 20000)
    test = [1, 0]
    test1 = [0, 1]
    test2 = [1, 1]
    test3 = [0, 0]
    print('\n\n-----------PREDICTIONS FOR MOD 2 WITH 4 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {1}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {1}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {0}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {0}")


def test_for_multiple_outputs():
    x_test = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]])
    y_test = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
    net = NeuralNet(3, 3, 3, 3)
    net.train(x_test, y_test, 10000)
    test = [1, 0, 1]
    test1 = [0, 1, 1]
    test2 = [1, 1, 1]
    test3 = [0, 0, 1]
    print('\n\n-----------PREDICTIONS FOR MOD 2 WITH 2 HIDDEN NEURONS-----------')
    print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {[0, 0, 1]}")
    print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {[1, 0, 0]}")
    print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {[1, 0, 0]}")
    print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {[0, 0, 1]}")


# run_two_neuron_net_for_EQUALS()
# run_two_neuron_net_for_NOT()
# run_three_neuron_net_with_one_hidden_for_EQUALS()
# run_three_neuron_net_with_one_hidden_for_NOT()
# run_four_neuron_net_with_two_hidden_for_EQUALS()
# run_four_neuron_net_with_two_hidden_for_NOT()
# run_three_neuron_net_with_zero_hidden_for_AND()
# run_five_neuron_net_with_two_hidden_for_AND()
# run_three_neuron_net_with_zero_hidden_for_OR()
# run_five_neuron_net_with_two_hidden_for_OR()
# run_seven_neuron_net_with_four_hidden_for_AND()
# run_seven_neuron_net_with_four_hidden_for_OR()
# run_five_neuron_net_with_two_hidden_for_MOD2()
# run_seven_neuron_net_with_four_hidden_for_MOD2()
test_for_multiple_outputs()
