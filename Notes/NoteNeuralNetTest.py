import unittest
from playsound import playsound
from Notes.NoteNeuralNet import TwoNoteNeuralNet


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = TwoNoteNeuralNet(number_of_epochs=10000,
                                    number_of_previous_notes=5,
                                    save_model=True)
        self.net.train()

    def test_set_up(self):
        # self.net.predict()
        self.assertTrue(True)

    def test_play_music(self):
        playsound('generatedMusic.wav')
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
