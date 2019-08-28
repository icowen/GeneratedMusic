import numpy as np

from Helpers import clean_word_list, convert_char


class WordConverter:
    def __init__(self, word_list):
        self.space = np.zeros((27,), dtype=int)
        self.space[26] = 1
        self.word_list = clean_word_list(word_list)
        self.input = []
        self.output = []
        self.convert_words()

    def convert_words(self):
        first_two_letters = []
        next_letter = []
        for word in self.word_list:
            for i in range(-1, len(word) - 1):
                if i == -1:
                    first_two_letters.append(np.concatenate([self.space, convert_char(word[0])]))
                    if len(word) > 1:
                        next_letter.append(convert_char(word[1]))
                    else:
                        next_letter.append(self.space)
                elif i == len(word) - 2:
                    first_two_letters.append(np.concatenate([convert_char(word[i]),
                                                             convert_char(word[i + 1])]))
                    next_letter.append(self.space)
                else:
                    first_two_letters.append(
                        np.concatenate([convert_char(word[i]),
                                        convert_char(word[i + 1])]))
                    next_letter.append(convert_char(word[i + 2]))

        self.input = first_two_letters
        self.output = next_letter

    def get_output(self):
        return self.output

    def get_input(self):
        return self.input
