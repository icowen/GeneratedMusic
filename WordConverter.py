from string import ascii_lowercase
import numpy as np
import re


class WordConverter:
    def __init__(self, word_list):
        self.space = np.zeros((27,), dtype=int)
        self.space[26] = 1
        self.word_list = word_list
        self.input = self.convert_words()
        self.output = []

    def convert_words(self):
        first_two_letters = []
        next_letter = []
        for word in self.word_list:
            word = re.sub('[^a-zA_Z]', '', word.lower())
            for i in range(-1, len(word) - 1):
                if i == -1:
                    first_two_letters.append(np.concatenate([self.space, self.convert_char(word[0])]))
                    next_letter.append(self.convert_char(word[1]))
                elif i == len(word) - 2:
                    first_two_letters.append(np.concatenate([self.convert_char(word[i]),
                                                             self.convert_char(word[i + 1])]))
                    next_letter.append(self.space)
                else:
                    first_two_letters.append(
                        np.concatenate([self.convert_char(word[i]),
                                        self.convert_char(word[i + 1])]))
                    next_letter.append(self.convert_char(word[i + 2]))

        self.set_output(next_letter)

        return first_two_letters, next_letter

    def convert_char(self, char):
        c = np.zeros((27,), dtype=int)
        c[self.convert_char_to_number(char)] = 1
        return c

    def convert_char_to_number(self, char):
        return ord(char) - 97

    def convert_number_list_to_ascii(self, letter_list):
        index = np.argmax(letter_list)
        if index == 26:
            return '\' \''
        return chr(index + 97)

    def get_converted_words(self):
        return self.input

    def letter_dict(self, index):
        letter_dict = dict()
        for c in ascii_lowercase:
            letter_dict[self.convert_char_to_number(c)] = c
        if index in letter_dict.keys():
            return letter_dict[index]
        else:
            return ' '

    def convert_index_to_letter(self, list_of_letters_lists):
        letters = []
        for letter_list in list_of_letters_lists:
            letters.append(self.convert_number_list_to_ascii(letter_list.tolist()))
        return letters

    def set_output(self, next_letter):
        self.output = next_letter
