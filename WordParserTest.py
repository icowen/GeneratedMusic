import numpy as np
import unittest
from WordParser import WordConverter


class TestForMultipleOutputs(unittest.TestCase):
    def setUp(self):
        text_file = ['hello, Ca\'t', '\ncat']
        self.converter = WordConverter(text_file)
        self.input = self.converter.get_input()
        self.output = self.converter.get_output()

    def test_first_input(self):
        space = np.zeros((27,), dtype=int)
        space[26] = 1
        h = np.zeros((27,), dtype=int)
        h[7] = 1
        expected = np.concatenate([space, h])
        actual = self.input[0]
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f'Actual: {actual}\n'
                                              f'Expected: {expected}')

    def test_first_output(self):
        e = np.zeros((27,), dtype=int)
        e[4] = 1
        expected = np.concatenate([e])
        actual = self.output[0]
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f'Actual: {actual}\n'
                                              f'Expected: {expected}')

    def test_last_input(self):
        a = np.zeros((27,), dtype=int)
        a[0] = 1
        t = np.zeros((27,), dtype=int)
        t[19] = 1
        expected = np.concatenate([a, t])
        actual = self.input[len(self.output) - 1]
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f'Actual: {actual}\n'
                                              f'Expected: {expected}')

    def test_last_output(self):
        space = np.zeros((27,), dtype=int)
        space[26] = 1
        expected = np.concatenate([space])
        actual = self.output[len(self.output) - 1]
        np.testing.assert_array_equal(actual, expected)

    def test_convert_char_to_number(self):
        actual = self.converter.get_index_for_char_array('a')
        expected = 0
        self.assertEqual(actual, expected)

    def test_letter_dict(self):
        actual = self.converter.letter_dict(0)
        expected = 'a'
        self.assertEqual(actual, expected)

    def test_letter_dict_space(self):
        actual = self.converter.letter_dict(26)
        expected = ' '
        self.assertEqual(actual, expected)

    @unittest.skip
    def test_short_story(self):
        word_file = open('shortStory.txt', 'r')
        words = []
        for w in word_file.readlines():
            words.append(w)
        word_file.close()
        converter = WordConverter(words)
        x = converter.get_input()
        y = converter.get_output()
        for i in range(20):
            print(f'input: {converter.convert_index_to_ascii(x[i][:27])}'
                  f'{converter.convert_index_to_ascii(x[i][27:])} '
                  f'output: {converter.convert_index_to_ascii(y[i])}')


if __name__ == '__main__':
    unittest.main()
