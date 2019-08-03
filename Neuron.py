class Neuron:
    def __init__(self, bias, layer, position_in_layer, learning_rate):
        self.bias = bias
        self.layer = layer
        self.position_in_layer = position_in_layer
        self.id = f'Layer: {self.layer} Position: {self.position_in_layer}'
        self.activation = 0
        self.sum = 0
        self.change_in_bias = 0
        self.change_in_activation = 0
        self.learning_rate = learning_rate
        self.weights = {'weights_from_this_neuron': [],
                        'weights_to_this_neuron': []}

    def __str__(self):
        return f'NEURON--- Layer: {self.layer}, ' \
               f'Position: {self.position_in_layer}, ' \
               f'Activation: {self.activation}, ' \
               f'Bias: {self.bias}, '

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

    def get_learning_rate(self):
        return self.learning_rate

    def add_weight_to_this_neuron(self, weight):
        self.weights['weights_to_this_neuron'].append(weight)

    def add_weight_from_this_neuron(self, weight):
        self.weights['weights_from_this_neuron'].append(weight)

    def add_change_in_bias(self, change_in_bias):
        self.change_in_bias = change_in_bias

    def add_change_in_activation(self, change_in_activation):
        self.change_in_activation = change_in_activation

    def update(self):
        self.bias = self.bias - self.change_in_bias * self.learning_rate
