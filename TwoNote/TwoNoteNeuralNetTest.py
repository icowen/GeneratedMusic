import unittest
from TwoNote.TwoNoteNeuralNet import TwoNoteNeuralNet


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = TwoNoteNeuralNet(number_of_epochs=1)
        self.net.train()

    def test_set_up(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
