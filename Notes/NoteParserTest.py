import numpy as np
import unittest
from Notes.NoteParser import NoteParser


class MyTestCase(unittest.TestCase):
    def setUp(self):
        fake_input = 'C:/Users/ian_c/GeneratedMusic\MusicFiles\\3rdman.mid\n' \
                     '[\'G3, \'G#3\', \'A3\', \'A3\', \'A3\', \'A3\']\n' \
                     'C:/Users/ian_c/GeneratedMusic\MusicFiles\\4saalleg.mid\n' \
                     '[\'F4\', \'C5\', \'F4\']'
        self.parser_two_notes = NoteParser(fake_input)
        self.parser_five_notes = NoteParser(fake_input, 5)

    def test_get_first_note_in_array_form(self):
        expected = np.zeros((128,), dtype=int)
        expected[55] = 1
        actual = self.parser_two_notes.converted_songs[0][0]
        np.testing.assert_array_equal(actual, expected)

    def test_should_get_5_notes_for_input(self):
        length_of_first_notes = len(self.parser_five_notes.first_notes[0][0])
        length_of_5_notes = 5 * 128
        self.assertEqual(length_of_first_notes, length_of_5_notes)

    def test_keras_first_two_notes(self):
        first_note = np.zeros((128,), dtype=int)
        first_note[55] = 1
        second_note = np.zeros((128,), dtype=int)
        second_note[56] = 1
        expected = np.concatenate([first_note, second_note])
        actual = self.parser_two_notes.first_notes[0][0]
        np.testing.assert_array_equal(actual, expected)

    def test_keras_next_notes(self):
        next_note = np.zeros((128,), dtype=int)
        next_note[57] = 1
        actual = self.parser_two_notes.next_notes[0][0]
        np.testing.assert_array_equal(actual, next_note)

    def test_convert_array_into_note(self):
        next_note = np.zeros((128,), dtype=int)
        next_note[57] = 1
        converted = self.parser_two_notes.convert_array_into_note(next_note)
        self.assertEqual(converted, 'A3')


class TestNotesWithVolumeAndLength(unittest.TestCase):
    def setUp(self):
        fake_input = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,0.30000000000000004\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,91,0.10000000000000009-0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,0.30000000000000004\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,91,0.10000000000000009'
        self.parser_two_notes = NoteParser(fake_input, note_type=['volume', 'length'])
        self.parser_five_notes = NoteParser(fake_input, 5)

    def test_get_first_note_in_array_form(self):
        expected = np.zeros((130,), dtype=float)
        expected[55] = 1
        expected[128] = 86
        expected[129] = 0.30000000000000004
        actual = self.parser_two_notes.converted_songs[0][0]
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
