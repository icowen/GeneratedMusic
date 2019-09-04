import random
import tensorflow as tf
import numpy as np
from Helpers import convert_char, convert_index_number_to_ascii_letter
from TwoLetterNeuralNet.WordParser import WordConverter
import sys

np.set_printoptions(threshold=sys.maxsize)


class TwoLetterNeuralNet:
    random.seed(1)
    np.random.seed(2)
    tf.set_random_seed(3)

    def __init__(self,
                 input_file='shortStory.txt',
                 number_of_hidden_nodes=50,
                 number_of_output_nodes=27,
                 number_of_epochs=10000,
                 batch_size=50,
                 save_model=False,
                 json_file_input=False):
        self.input_file = input_file
        self.x_train, self.y_train = self.get_training_data()
        self.model = tf.keras.models.Sequential()
        self.number_of_hidden_nodes = number_of_hidden_nodes
        self.number_of_output_nodes = number_of_output_nodes
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.save_model = save_model
        self.model_save_h5 = 'TwoLetterNeuralNet/Aladdin.h5'
        self.weights_save_h5 = 'TwoLetterNeuralNet/AladdinWeights.h5'
        self.model_save_json = 'TwoLetterNeuralNet/Aladdin.json'

        if json_file_input:
            self.read_model_from_json(json_file_input)
        else:
            self.set_up_model()

    def get_training_data(self):
        global converter
        word_file = open(self.input_file, 'r', encoding='utf8')
        words = []
        for w in word_file.readlines():
            print(w)
            words.append(w)
        word_file.close()
        converter = WordConverter(words)
        x_list = converter.get_input()
        y_list = converter.get_output()
        x_list = tf.keras.utils.normalize(x_list, axis=1)
        return np.asarray(x_list), np.asarray(y_list)

    def set_up_model(self):
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(self.number_of_hidden_nodes,
                                             activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(self.number_of_output_nodes,
                                             activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        self.model.fit(self.x_train,
                       self.y_train,
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size)
        if self.save_model:
            self.model.save(self.model_save_h5)
            self.save_model_to_json()

    def generate_next_letter(self, first_letter, second_letter):
        normalized_input = self.get_correct_input_shape(first_letter, second_letter)
        predictions = self.get_predictions(normalized_input)
        random_number = random.random()
        cutoff = 0
        for i in range(len(predictions)):
            letter = predictions[i]
            cutoff += letter
            if random_number <= cutoff:
                return convert_index_number_to_ascii_letter(i)

    def generate_words(self,
                       number_of_generated_letters,
                       first_letter,
                       second_letter):
        result = first_letter + second_letter
        for i in range(number_of_generated_letters):
            next_letter = self.generate_next_letter(first_letter, second_letter)
            result += next_letter
            first_letter = second_letter
            second_letter = ' ' if next_letter == '\' \'' else next_letter
        return result

    def get_predictions(self, normalized_input):
        return self.normalize_predictions(
            self.model.predict([normalized_input[0:]])
        )

    @staticmethod
    def normalize_predictions(predictions):
        total = sum(predictions[0])
        return list(map(lambda x: x / total, predictions[0]))

    @staticmethod
    def get_correct_input_shape(first_letter, second_letter):
        return np.asarray(tf.keras.utils.normalize(np.asarray(
            np.concatenate([
                convert_char(first_letter),
                convert_char(second_letter)
            ])
        )))

    def save_model_to_json(self):
        model_json = self.model.to_json()
        with open(self.model_save_json, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.weights_save_h5)
        print('Saved model to disk')

    def read_model_from_json(self, file_name):
        json_file = open(file_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(self.weights_save_h5)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def read_model(self):
        self.model = tf.keras.models.load_model(self.model_save_h5)
