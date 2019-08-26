import unittest

from TwoLetterNeuralNet import TwoLetterNeuralNet


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = TwoLetterNeuralNet(number_of_epochs=1)

    def test_for_net(self):
        result = self.net.generate_words(3, 'a', 't')
        self.assertEqual(result, 'atewu')


if __name__ == '__main__':
    unittest.main()
