import sys
import unittest

from TwoLetterNeuralNet import TwoLetterNeuralNet


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = TwoLetterNeuralNet(number_of_epochs=10000,
                                      save_model=True)
        self.net.read_model_from_json('TwoLetterNeuralNet10000Trials.json')
        self.net.train()
        self.result = self.net.generate_words(998, 'a', 't')

    def test_for_net(self):
        out_file = open('1000_characters_10000_epochs.txt', 'w')
        out_file.write(self.result)
        out_file.close()
        print(self.result)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
