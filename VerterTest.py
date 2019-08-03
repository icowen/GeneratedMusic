import numpy as np
import unittest
from WordConverter import WordConverter


class TestForMultipleOutputs(unittest.TestCase):
    def setUp(self):
        word_list = ["hello", "cat"]
        self.converter = WordConverter(word_list)
        self.input, self.output = self.converter.get_converted_words()

    def test_first_input(self):
        space = np.zeros((26,), dtype=int)
        h = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected = np.concatenate([space, h])
        actual = self.input[0]
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f'Actual: {actual}\n'
                                              f'Expected: {expected}')

    def test_first_output(self):
        e = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected = np.concatenate([e])
        actual = self.output[0]
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f'Actual: {actual}\n'
                                              f'Expected: {expected}')

    def test_last_input(self):
        a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        expected = np.concatenate([a, t])
        actual = self.input[len(self.output) - 1]
        np.testing.assert_array_equal(actual, expected,
                                      err_msg=f'Actual: {actual}\n'
                                              f'Expected: {expected}')

    def test_last_output(self):
        space = np.zeros((26,), dtype=int)
        expected = np.concatenate([space])
        actual = self.output[len(self.output) - 1]
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
