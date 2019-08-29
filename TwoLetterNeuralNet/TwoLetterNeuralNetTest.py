import unittest
from TwoLetterNeuralNet.TwoLetterNeuralNet import TwoLetterNeuralNet


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = TwoLetterNeuralNet(input_file='aladdin.txt',
                                      number_of_epochs=10000,
                                      save_model=True)
        self.net.train()
        self.result = self.net.generate_words(998, 'a', 't')

    def test_for_net(self):
        out_file = open(f'1000_charactersALADDIN.txt', 'w')
        out_file.write(self.result)
        out_file.close()
        print(self.result)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
