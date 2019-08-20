class Weight:
    def __init__(self, weight, from_neuron, learning_rate):
        self.weight = weight
        self.from_neuron = from_neuron
        self.to_neuron = None
        self.change = 0
        self.learning_rate = learning_rate

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
        self.weight = self.weight - self.change * self.learning_rate

    def get_learning_rate(self):
        return self.learning_rate
