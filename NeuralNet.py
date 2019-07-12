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
                 number_of_input_neurons,
                 number_of_layers,
                 number_of_neurons_per_layer):
        self.neurons = []
        self.weights = []
        self.number_of_layers = number_of_layers

        self.set_input_neurons(number_of_input_neurons)
        self.set_hidden_neurons(number_of_layers, number_of_neurons_per_layer)
        self.set_output_neurons(number_of_layers)
        self.set_weight_connections()

        self.print_net()

    def set_input_neurons(self, number_of_input_neurons):
        input_neurons = []
        for n in range(number_of_input_neurons):
            neuron = Neuron(random.random(), 1, n)
            input_neurons.append(neuron)
        self.neurons.append(input_neurons)

    def set_hidden_neurons(self, number_of_layers, number_of_neurons_per_layer):
        biases = [-9, -1]
        bias_index = 0
        for layer_number in range(number_of_layers - 2):
            layer = []
            for neuron_position in range(number_of_neurons_per_layer):
                neuron = Neuron(biases[bias_index], layer_number + 2, neuron_position)
                bias_index += 1
                layer.append(neuron)
            self.neurons.append(layer)

    def set_output_neurons(self, number_of_layers):
        output_neuron = Neuron(2, number_of_layers, 0)
        self.neurons.append([output_neuron])

    def set_weight_connections(self):
        weights = [9, 1, 9, 1, 8, -13]
        weight_index = 0
        for layer_number in range(self.number_of_layers):
            for neuron in self.neurons[layer_number]:
                if layer_number + 1 < self.number_of_layers:
                    next_layer = self.neurons[layer_number + 1]
                    for neuron_in_next_layer in next_layer:
                        weight = Weight(weights[weight_index], neuron)
                        weight_index += 1
                        weight.set_to_neuron(neuron_in_next_layer)
                        neuron.add_weight_from_this_neuron(weight)
                        neuron_in_next_layer.add_weight_to_this_neuron(weight)

    def print_net(self):
        for layer in self.neurons:
            print(f'\n-------------------Layer: {self.neurons.index(layer) + 1}-------------------')
            for neuron in layer:
                print(neuron)
                for weight in neuron.weights["weights_from_this_neuron"]:
                    print(weight)
                for weight in neuron.weights["weights_to_this_neuron"]:
                    print(weight)

    def get_new_sum_and_activation(self, neuron):
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
        y_predicted = self.neurons[self.number_of_layers - 1][0].get_activation()
        return y_predicted

    def train(self, input_data, output_data):
        for epoch in range(1):
            for x, y in zip(input_data, output_data):
                for i in range(len(x)):
                    self.neurons[0][i].set_activation(x[i])
                for layer in self.neurons:
                    if self.neurons.index(layer) != 0:
                        for neuron in layer:
                            self.get_new_sum_and_activation(neuron)

                output_neuron = self.neurons[self.number_of_layers - 1][0]
                y_predicted = output_neuron.get_activation()

                if y[0] == 1:
                    derivative_of_loss = -1 / y_predicted
                else:
                    derivative_of_loss = 1 / (1 - y_predicted)

                # derivative_of_loss = -2 * (y[0] - y_predicted)
                output_neuron.add_change_in_activation(derivative_of_loss)
                output_neuron.add_change_in_bias(derivative_of_loss * sigmoid_derivative(output_neuron.get_sum()))
                for to_weight in output_neuron.weights["weights_to_this_neuron"]:
                    new_weight_value = sigmoid_derivative(output_neuron.get_activation()) * \
                                       to_weight.from_neuron.get_activation() * \
                                       output_neuron.get_change_in_activation()
                    to_weight.add_change(new_weight_value)

                i = self.number_of_layers - 2
                while i > 0:
                    for neuron in self.neurons[i]:
                        new_bias_value = sigmoid_derivative(neuron.get_activation())
                        bias_values_from_stuff_to_the_right = 0
                        dl_dv = 0
                        for from_weight in neuron.weights["weights_from_this_neuron"]:
                            bias_values_from_stuff_to_the_right += from_weight.to_neuron.get_change_in_activation() * \
                                                                   sigmoid_derivative(from_weight.to_neuron.get_activation()) * \
                                                                   from_weight.weight
                            dl_dv += from_weight.to_neuron.get_change_in_activation() * \
                                     sigmoid_derivative(from_weight.to_neuron.get_activation()) * \
                                     from_weight.weight
                        neuron.add_change_in_activation(dl_dv)
                        if bias_values_from_stuff_to_the_right != 0:
                            new_bias_value *= bias_values_from_stuff_to_the_right
                            neuron.add_change_in_bias(new_bias_value)
                        for to_weight in neuron.weights["weights_to_this_neuron"]:
                            new_weight_value = sigmoid_derivative(neuron.get_activation()) * \
                                               to_weight.from_neuron.get_activation() *  \
                                               neuron.get_change_in_activation()
                            to_weight.add_change(new_weight_value)

                    i -= 1
                for layer in self.neurons:
                    for neuron in layer:
                        for weight in neuron.weights["weights_from_this_neuron"]:
                            weight.update()
                        neuron.update()
            if epoch % 10 == 0:
                y_predictions = np.apply_along_axis(self.predict, 1, input_data)
                loss_function = log_loss(output_data, y_predictions)
                print(f"Epoch {epoch} loss_function: {round(loss_function, 3)}")
        self.print_net()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
y_test = np.array([[1], [0], [1], [0]])

net = NeuralNet(2, 3, 2)
net.train(x_test, y_test)
test = [1, 0]
test1 = [0, 1]
test2 = [1, 1]
test3 = [0, 0]
print(f"Input {test},  Prediction: {net.predict(test)},  Actual: {(test[0] + test[1]) % 2}")
print(f"Input {test1},  Prediction: {net.predict(test1)},  Actual: {(test1[0] + test1[1]) % 2}")
print(f"Input {test2},  Prediction: {net.predict(test2)},  Actual: {(test2[0] + test2[1]) % 2}")
print(f"Input {test3},  Prediction: {net.predict(test3)},  Actual: {(test3[0] + test3[1]) % 2}")
