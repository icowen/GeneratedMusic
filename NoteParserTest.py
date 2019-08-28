import numpy as np
import unittest
from NoteParser import NoteParser


class MyTestCase(unittest.TestCase):
    def setUp(self):
        fake_input = 'C:/Users/ian_c/GeneratedMusic\MusicFiles\\3rdman.mid\n' \
                     '[\'G3, \'G#3\', \'A3\']\n' \
                     'C:/Users/ian_c/GeneratedMusic\MusicFiles\\4saalleg.mid\n' \
                     '[\'F4\', \'C5\', \'F4\']'
        self.parser = NoteParser(fake_input)

    def test_get_first_note(self):
        self.assertEqual(self.parser.songs[0][:2], 'G3')

    def test_get_first_note_in_array_form(self):
        expected = np.zeros((128,), dtype=int)
        expected[55] = 1
        actual = self.parser.converted_songs[0][0]
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
