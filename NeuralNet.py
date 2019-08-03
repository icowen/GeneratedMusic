import random
from Helpers import *
from Neuron import Neuron
from Weight import Weight
import time


class NeuralNet:
    def __init__(self,
                 number_of_input_neurons: int,
                 number_of_layers: int,
                 number_of_neurons_per_layer: int,
                 number_of_output_neurons: int = 1,
                 hidden_neuron_biases: list = None,
                 output_neuron_biases: list = None,
                 weights=None,
                 activation_function=ActivationFunction.LOG_LOSS,
                 learning_rate=0.1):
        self.neurons = []
        self.weights = []
        self.number_of_layers = number_of_layers
        self.number_of_output_neurons = number_of_output_neurons
        self.number_of_neurons_per_layer = number_of_neurons_per_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
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
        self.check_activation_function(activation_function)

        self.print_net()

    def train(self, input_data: np.ndarray,
              output_data: np.ndarray,
              epochs=10000):
        start_time = time.time()
        print(f'\n\n-------------------------------TRAINING NET FOR {epochs} TIMES-------------------------------')
        self.check_for_valid_input(input_data, output_data)
        for epoch in range(epochs):
            for x, y in zip(input_data, output_data):
                self.input_data_into_net(x)

                correct_index = self.get_correct_index(y)
                output_probabilities = self.get_output_probabilities()

                y_predicted = self.get_output_layer()[correct_index].get_activation()
                sum_of_output_probabilities = sum(output_probabilities)
                y_true = y[correct_index]

                for i in range(len(self.get_output_layer())):
                    output_neuron = self.get_output_layer()[i]
                    derivative_of_loss = self.activation_function.derivative(y_predicted,
                                                                             y_true,
                                                                             correct_index == i,
                                                                             sum_of_output_probabilities)
                    self.calculate_partial_derivative_for_output_neuron(derivative_of_loss, output_neuron)
                    self.calculate_partial_derivatives_for_weights(output_neuron)

                self.calculate_partial_derivatives()
                self.update_net()
        self.print_net()
        print(f'\n\n Net trained for {epochs} epochs in {time.time() - start_time} seconds.')

    def predict(self, input_data):
        self.input_data_into_net(input_data)
        output_layer = self.get_output_layer()
        if len(output_layer) == 1:
            return output_layer[0].get_activation()
        else:
            output = []
            for neuron in output_layer:
                output.append(neuron.get_activation())
            output = normalize(output)
            index_with_max_probability = output.index(max(output))
            return index_with_max_probability

    def get_output_probabilities(self):
        output_probabilities = []
        for neuron in self.get_output_layer():
            output_probabilities.append(neuron.get_activation())
        return output_probabilities

    def input_data_into_net(self, x):
        for i in range(len(x)):
            self.neurons[0][i].set_activation(x[i])
        for layer in self.neurons:
            if self.neurons.index(layer) != 0:
                for neuron in layer:
                    self.get_new_sum_and_activation(neuron)

    def update_net(self):
        for layer in self.neurons:
            for neuron in layer:
                for weight in neuron.weights["weights_from_this_neuron"]:
                    weight.update()
                neuron.update()

    def calculate_partial_derivatives(self):
        layer_index = self.number_of_layers - 2
        while layer_index > 0:
            for neuron in self.neurons[layer_index]:
                self.calculate_partial_derivative_for_bias_and_activation(neuron)
                self.calculate_partial_derivatives_for_weights(neuron)
            layer_index -= 1

    @staticmethod
    def calculate_partial_derivative_for_output_neuron(derivative_of_loss, output_neuron):
        output_neuron.add_change_in_activation(derivative_of_loss)
        change_in_bias = derivative_of_loss * sigmoid_derivative(output_neuron.get_sum())
        output_neuron.add_change_in_bias(change_in_bias)

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
        # print(f'{neuron}: \n\t {dl_da}')
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
            # print(f'{to_weight}: \n\t {new_weight_value}')
            to_weight.add_change(new_weight_value)

    @staticmethod
    def get_new_sum_and_activation(neuron):
        new_sum_value = 0
        for weight in neuron.weights['weights_to_this_neuron']:
            activation = weight.from_neuron.get_activation()
            new_sum_value += weight.weight * activation
        new_sum_value += neuron.get_bias()
        neuron.set_sum(new_sum_value)
        neuron.set_activation(sigmoid(new_sum_value))

    @staticmethod
    def get_correct_index(y):
        for i in range(len(y)):
            if y[i] == 1:
                return i
        return 0

    def get_output_layer(self):
        return self.neurons[self.number_of_layers - 1]

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

    def set_input_neurons(self, number_of_input_neurons):
        input_neurons = []
        for n in range(number_of_input_neurons):
            neuron = Neuron(random.random(), 1, n, self.learning_rate)
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
                neuron = Neuron(bias, layer_number + 2, neuron_position, self.learning_rate)
                layer.append(neuron)
            self.neurons.append(layer)

    def set_output_neurons(self, output_neuron_biases):
        self.check_for_correct_number_of_output_neurons(output_neuron_biases)
        layer = []
        for neuron_position in range(self.number_of_output_neurons):
            bias = output_neuron_biases[neuron_position] if output_neuron_biases else random.random()
            neuron = Neuron(bias, self.number_of_layers, neuron_position, self.learning_rate)
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
                        weight = Weight(weight_value, neuron, self.learning_rate)
                        weight.set_to_neuron(neuron_in_next_layer)
                        neuron.add_weight_from_this_neuron(weight)
                        neuron_in_next_layer.add_weight_to_this_neuron(weight)
                        self.weights.append(weight)

    @staticmethod
    def check_activation_function(activation_function):
        if not isinstance(activation_function, ActivationFunction):
            raise TypeError("Activation function must be an enumeration of ActivationFunction.")

    def check_for_valid_input(self, input_data, output_data):
        if not isinstance(input_data, np.ndarray):
            raise TypeError(f'Use a list for input data instead of {type(input_data)}')
        if not len(input_data[0]) == len(self.neurons[0]):
            raise ValueError(f'input_data ({len(input_data[0])}) '
                             f'is not the same length as number of input nodes ({len(self.neurons[0])})')
        if not len(output_data[0]) == len(self.get_output_layer()):
            raise ValueError(f'output_data ({len(output_data[0])}) '
                             f'is not the same length as number of output nodes ({len(self.get_output_layer())})')

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

    def print_net(self):
        for layer in self.neurons:
            print(f'\n-------------------Layer: {self.neurons.index(layer) + 1}-------------------')
            for neuron in layer:
                print(neuron)
                for weight in neuron.weights["weights_from_this_neuron"]:
                    print(weight)
