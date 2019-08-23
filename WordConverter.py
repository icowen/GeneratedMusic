from string import ascii_lowercase
import numpy as np
import re


class WordConverter:
    def __init__(self, word_list):
        self.space = np.zeros((27,), dtype=int)
        self.space[26] = 1
        self.word_list = self.clean_word_list(word_list)
        self.input = []
        self.output = []
        self.convert_words()

    def convert_words(self):
        first_two_letters = []
        next_letter = []
        for word in self.word_list:
            for i in range(-1, len(word) - 1):
                if i == -1:
                    first_two_letters.append(np.concatenate([self.space, self.convert_char(word[0])]))
                    if len(word) > 1:
                        next_letter.append(self.convert_char(word[1]))
                    else:
                        next_letter.append(self.space)
                elif i == len(word) - 2:
                    first_two_letters.append(np.concatenate([self.convert_char(word[i]),
                                                             self.convert_char(word[i + 1])]))
                    next_letter.append(self.space)
                else:
                    first_two_letters.append(
                        np.concatenate([self.convert_char(word[i]),
                                        self.convert_char(word[i + 1])]))
                    next_letter.append(self.convert_char(word[i + 2]))

        self.input = first_two_letters
        self.output = next_letter

    def clean_word_list(self, word_list):
        clean = []
        for line in word_list:
            for word in line.split():
                word = re.sub('[^a-zA_Z]', '', word.lower())
                word = re.sub('([\s])', ' ', word)
                clean.append(word)
        return clean

    def convert_char(self, char):
        c = np.zeros((27,), dtype=int)
        c[self.get_index_for_char_array(char)] = 1
        return c

    def get_index_for_char_array(self, char):
        return ord(char) - 97

    def convert_index_to_ascii(self, letter_list):
        index = np.argmax(letter_list)
        if index == 26:
            return '\' \''
        return chr(index + 97)

    def letter_dict(self, index):
        letter_dict = dict()
        for c in ascii_lowercase:
            letter_dict[self.get_index_for_char_array(c)] = c
        if index in letter_dict.keys():
            return letter_dict[index]
        else:
            return ' '

    def get_output(self):
        return self.output

    def get_input(self):
        return self.input
