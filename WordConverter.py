import numpy as np
import re


class WordConverter:
    def __init__(self, word_list):
        self.space = np.zeros((26,), dtype=int)
        self.word_list = word_list
        self.converted_words = self.convert_words()

    def convert_words(self):
        converted_words = []
        for word in self.word_list:
            word = re.sub('[^a-zA_Z]', '', word.lower())
            for i in range(-1, len(word)):
                if i == -1:
                    converted_words.append(
                        np.concatenate([self.space, self.convert_char(word[0])])
                    )
                elif i == len(word) - 1:
                    converted_words.append(
                        np.concatenate([self.convert_char(word[i]), self.space])
                    )
                else:
                    converted_words.append(
                        np.concatenate([self.convert_char(word[i]), self.convert_char(word[i + 1])])
                    )

        return np.asarray(converted_words)

    def convert_char(self, char):
        c = np.zeros((26,), dtype=int)
        c[ord(char) - 97] = 1
        return c

    def get_converted_words(self):
        return self.converted_words


word_file = open('words.txt', 'r')
words = []
for w in word_file.readlines():
    words.append(w)
wc = WordConverter(words)
cw = wc.get_converted_words()
print(len(cw))

