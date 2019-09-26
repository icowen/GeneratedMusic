import unittest
from playsound import playsound
from Notes.NoteNeuralNet import NoteNeuralNet


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.net = NoteNeuralNet(number_of_epochs=1,
                                 number_of_previous_notes=5,
                                 save_model=True)
        self.net.train()

    def test_set_up(self):
        # self.net.predict()
        self.assertTrue(True)

    # def test_play_music(self):
    #     playsound('generatedMusic.wav')
    #     self.assertTrue(True)


class NetWithLengthAndVolume(unittest.TestCase):
    def setUp(self):
        self.net = NoteNeuralNet(number_of_epochs=1,
                                 number_of_previous_notes=5,
                                 save_model=True,
                                 input_file_with_notes_as_arr='flute_notes_with_volume_and_length.csv',
                                 wav_file_name='MusicWithLengthAndVolume.wav',
                                 model_save_h5='NetWithLengthAndVolume.h5',
                                 weights_save_h5='WeightsWithLengthAndVolume.h5',
                                 model_save_json='NetWithLengthAndVolume.json'
                                 )
        self.net.train()

    def test_set_up(self):
        # self.net.predict()
        self.assertTrue(True)

    # def test_play_music(self):
    #     playsound('generatedMusic.wav')
    #     self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
