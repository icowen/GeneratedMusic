import numpy as np
import unittest
from WordConverter import WordConverter


class TestForMultipleOutputs(unittest.TestCase):
    def setUp(self):
        word_list = ["hello", "cat"]
        self.converter = WordConverter(word_list)
        self.input, self.output = self.converter.get_converted_words()

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
        actual = self.converter.convert_char_to_number('a')
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

    def test_get_indicies(self):
        actual = self.converter.convert_index_to_letter(self.output)[0]
        expected = 'e'
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
