import sys
import numpy as np
import unittest
from unittest.mock import patch
from Helpers import ActivationFunction
from NeuralNet import NeuralNet


@patch('builtins.print')
@unittest.skip
class TestForDifferentLogic(unittest.TestCase):
    def test_two_neurons_for_equal(self, mocked_print):
        x_test = np.array([[1], [0]])
        y_test = np.array([[1], [0]])
        net = NeuralNet(1, 2, 1)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nTwo neuron net predicting for EQUALS:\n')
        for test, expected in zip([[1], [0], [1], [0]], [1, 0, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {round(expected)}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_two_neurons_for_not(self, mocked_print):
        x_test = np.array([[1], [0]])
        y_test = np.array([[0], [1]])
        net = NeuralNet(1, 2, 1)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nTwo neuron net predicting for NOT:\n')
        for test, expected in zip([[1], [0], [1], [0]], [0, 1, 0, 1]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {round(expected)}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_three_neurons_for_equals(self, mocked_print):
        x_test = np.array([[1], [0]])
        y_test = np.array([[1], [0]])
        net = NeuralNet(1, 3, 1)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nThree neuron net predicting for EQUALS:\n')
        for test, expected in zip([[1], [0], [1], [0]], [1, 0, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {round(expected)}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_three_neurons_for_not(self, mocked_print):
        x_test = np.array([[1], [0]])
        y_test = np.array([[0], [1]])
        net = NeuralNet(1, 3, 1)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nThree neuron net predicting for NOT:\n')
        for test, expected in zip([[1], [0], [1], [0]], [0, 1, 0, 1]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {round(expected)}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_four_neurons_for_equals(self, mocked_print):
        x_test = np.array([[1], [0]])
        y_test = np.array([[1], [0]])
        net = NeuralNet(1, 3, 2)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nFour neuron net predicting for EQUALS:\n')
        for test, expected in zip([[1], [0], [1], [0]], [1, 0, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {round(expected)}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_four_neurons_for_not(self, mocked_print):
        x_test = np.array([[1], [0]])
        y_test = np.array([[0], [1]])
        net = NeuralNet(1, 3, 2)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nFour neuron net predicting for NOT:\n')
        for test, expected in zip([[1], [0], [1], [0]], [0, 1, 0, 1]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_three_neurons_for_and(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[0], [0], [0], [1]])
        net = NeuralNet(2, 2, 2)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nThree neuron net predicting for AND:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 0, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_five_neurons_for_and(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[0], [0], [0], [1]])
        net = NeuralNet(2, 3, 2)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nFive neuron net predicting for AND:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 0, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_three_neurons_for_or(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[1], [0], [1], [1]])
        net = NeuralNet(2, 2, 2)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nThree neuron net predicting for OR:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 1, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_five_neurons_for_or(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[1], [0], [1], [1]])
        net = NeuralNet(2, 3, 2)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nFive neuron net predicting for OR:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 1, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_seven_neurons_for_and(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[0], [0], [0], [1]])
        net = NeuralNet(2, 3, 4)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nSeven neuron net predicting for AND:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 0, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_seven_neurons_for_or(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[1], [0], [1], [1]])
        net = NeuralNet(2, 3, 4)
        net.train(x_test, y_test)
        sys.stderr.write(f'\n\nSeven neuron net predicting for OR:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 1, 1, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_five_neurons_for_mod2(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[1], [0], [1], [0]])
        net = NeuralNet(2, 3, 2, activation_function=ActivationFunction.LOG_LOSS)
        net.train(x_test, y_test, 20000)
        sys.stderr.write(f'\n\nFive neuron net predicting for MOD 2:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 1, 0, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    def test_seven_neurons_for_mod2(self, mocked_print):
        x_test = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        y_test = np.array([[1], [0], [1], [0]])
        net = NeuralNet(2, 3, 4)
        net.train(x_test, y_test, 20000)
        sys.stderr.write(f'\n\nSeven neuron net predicting for MOD 2:\n')
        for test, expected in zip([[1, 0], [0, 1], [1, 1], [0, 0]], [1, 1, 0, 0]):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)

    @unittest.skip
    def test_three_inputs_three_outputs_and_three_hidden_for_mod2(self, mocked_print):
        x_test = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0]])
        y_test = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
        net = NeuralNet(3, 3, 3, 3, activation_function=ActivationFunction.LOG)
        sys.stderr.write(f'\n\n------------------TRAINING------------------')
        net.train(x_test, y_test, 100000)
        sys.stderr.write(f'\n\nNine neuron net (3 input, 3 hidden, 3 output) predicting for MOD 2:\n')
        test_input = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
        test_output = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]
        for test, expected in zip(test_input, test_output):
            test_result = net.predict(test)
            sys.stderr.write(f'Input: {test}, Expected: {expected}, Prediction: {test_result}\n')
            self.assertTrue(np.abs(test_result - expected) < 0.1)


@patch('builtins.print')
class TestForProperConfiguration(unittest.TestCase):
    def testForProperNumberOfTrainingDataInput(self, mocked_print):
        net = NeuralNet(2, 3, 2)
        with self.assertRaises(ValueError):
            net.check_for_valid_input(np.array([[1]]), np.array([[1]]))

    def testForProperNumberOfTrainingDataOutput(self, mocked_print):
        net = NeuralNet(2, 3, 2, 3)
        with self.assertRaises(ValueError):
            net.check_for_valid_input(np.array([[1, 2]]), np.array([[1]]))

    def testForProperInputType(self, mocked_print):
        net = NeuralNet(2, 3, 2)
        with self.assertRaises(TypeError):
            net.train([[1, 0]], np.array([[1]]))

    def testForProperNumberOfHiddenBiasesInInput(self, mocked_print):
        with self.assertRaises(ValueError):
            NeuralNet(2, 3, 2, 1, [1, 2, 3])

    def testForProperNumberOfOutputBiasesInInput(self, mocked_print):
        with self.assertRaises(ValueError):
            NeuralNet(2, 3, 2, 1, [1, 2], [1, 2])

    def testForProperNumberOfWeightsInInput(self, mocked_print):
        with self.assertRaises(ValueError):
            NeuralNet(2, 3, 2, 1, [1, 2], [1], [1])

    def testForProperActivationFunction(self, mocked_print):
        def fake_function():
            pass

        with self.assertRaises(TypeError):
            NeuralNet(2, 3, 2, 1, [1, 2], [1], [1, 2, 3, 4, 5, 6], fake_function)


class TestForMultipleHiddenLayers(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('builtins.print')
        self.mock_object = self.patcher.start()
        self.x_test = np.array([[1], [0]])
        self.y_test = np.array([[1], [0]])
        self.hidden_node_biases = [0.10399593413625441,
                                   0.9477095968224993]
        self.output_node_biases = [0.1683069001902736]
        self.weights = [0.08946817286422626,
                        0.10066428298027419,
                        0.8768683269443115]
        self.net = NeuralNet(1, 4, 1, 1, self.hidden_node_biases, self.output_node_biases, self.weights)

    def tearDown(self):
        self.patcher.stop()

    def test_weights(self):
        self.assertEqual(self.net.get_weights(), self.weights)

    def test_biases(self):
        self.assertEqual(self.net.get_biases(), [0.10399593413625441,
                                                 0.9477095968224993,
                                                 0.1683069001902736])

    def test_partial_derivatives_for_biases(self):
        self.net.train(self.x_test, self.y_test, 1)
        self.assertEqual(self.net.get_biases(), [0.10397901969529151,
                                                 0.9470426291307198,
                                                 0.16445586456466701])

    def test_partial_derivatives_for_weights(self):
        self.net.train(self.x_test, self.y_test, 1)
        self.assertEqual(self.net.get_weights(), [0.08948139076478426,
                                                  0.10032526109304217,
                                                  0.8740529208576437])


class TestForMultipleOutputs(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('builtins.print')
        self.mock_object = self.patcher.start()
        self.x_test = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
        self.y_test = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        self.hidden_node_biases = [2, 2]
        self.output_node_biases = [3, 3]
        self.weights = [5, 5, 5, 5, 5, 5, 5, 5]
        self.net = NeuralNet(2, 3, 2, 2,
                             self.hidden_node_biases,
                             self.output_node_biases,
                             self.weights,
                             activation_function=ActivationFunction.LOG)

    def tearDown(self):
        self.patcher.stop()

    def test_weights(self):
        self.assertEqual(self.net.get_weights(), self.weights)

    def test_biases(self):
        self.assertEqual(self.net.get_biases(), [2, 2, 3, 3])

    def test_partial_derivatives_for_biases(self):
        self.net.train(self.x_test, self.y_test, 1)
        self.assertEqual(self.net.get_biases(), [2.000000000000002,
                                                 2.000000000000002,
                                                 3.0000000257168224,
                                                 2.99999997428318])

    def test_partial_derivatives_for_weights(self):
        self.net.train(self.x_test, self.y_test, 1)
        self.assertEqual(self.net.get_weights(), [5.0,
                                                  5.0,
                                                  5.0,
                                                  5.0,
                                                  5.000000021300251,
                                                  4.999999978699751,
                                                  5.000000021300251,
                                                  4.999999978699751])


if __name__ == '__main__':
    unittest.main()
