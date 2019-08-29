import numpy as np
import unittest
from TwoNote.NoteParser import NoteParser


class MyTestCase(unittest.TestCase):
    def setUp(self):
        fake_input = 'C:/Users/ian_c/GeneratedMusic\MusicFiles\\3rdman.mid\n' \
                     '[\'G3, \'G#3\', \'A3\']\n' \
                     'C:/Users/ian_c/GeneratedMusic\MusicFiles\\4saalleg.mid\n' \
                     '[\'F4\', \'C5\', \'F4\']'
        self.parser = NoteParser(fake_input)

    def test_get_first_note_in_array_form(self):
        expected = np.zeros((128,), dtype=int)
        expected[55] = 1
        actual = self.parser.converted_songs[0][0]
        np.testing.assert_array_equal(actual, expected)

    def test_keras_first_two_notes(self):
        first_note = np.zeros((128,), dtype=int)
        first_note[55] = 1
        second_note = np.zeros((128,), dtype=int)
        second_note[56] = 1
        expected = np.concatenate([first_note, second_note])
        actual = self.parser.first_two_notes[0][0]
        np.testing.assert_array_equal(actual, expected)

    def test_keras_next_notes(self):
        next_note = np.zeros((128,), dtype=int)
        next_note[57] = 1
        actual = self.parser.next_notes[0][0]
        np.testing.assert_array_equal(actual, next_note)

    def test_convert_array_into_note(self):
        next_note = np.zeros((128,), dtype=int)
        next_note[57] = 1
        converted = self.parser.convert_array_into_note(next_note)
        self.assertEqual(converted, 'A3')


if __name__ == '__main__':
    unittest.main()
