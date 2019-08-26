import random
import tensorflow as tf
import numpy as np
from Helpers import convert_char, convert_index_number_to_ascii_letter
from WordConverter import WordConverter
import sys
from tensorflow.python.client import device_lib


np.set_printoptions(threshold=sys.maxsize)


# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
# print('--------------------Devices from session: ')
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(f'\n---------get_available_gpus: {get_available_gpus()}')

class TwoLetterNeuralNet:
    random.seed(1)
    np.random.seed(2)
    tf.set_random_seed(3)

    def __init__(self,
                 input_file='shortStory.txt',
                 number_of_hidden_nodes=50,
                 number_of_output_nodes=27,
                 number_of_epochs=10000,
                 batch_size=50):
        self.input_file = input_file
        self.x_train, self.y_train = self.get_training_data()
        self.model = tf.keras.models.Sequential()
        self.number_of_hidden_nodes = number_of_hidden_nodes
        self.number_of_output_nodes = number_of_output_nodes
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.set_up_model()
        self.train()

    def get_training_data(self):
        global converter
        word_file = open(self.input_file, 'r')
        words = []
        for w in word_file.readlines():
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
        # with tf.device('/device:GPU:0'):
        self.model.fit(self.x_train,
                       self.y_train,
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size)

        val_loss, val_acc = self.model.evaluate(self.x_train, self.y_train)
        print(f'val_loss: {val_loss}; accuracy: {val_acc}')

    def generate_next_letter(self, first_letter, second_letter):
        converted_chars_to_lists = np.asarray(
            np.concatenate([
                convert_char(first_letter),
                convert_char(second_letter)
            ])
        )
        normalized_input = np.asarray(tf.keras.utils.normalize(converted_chars_to_lists))
        predictions = self.model.predict([normalized_input[0:]])
        random_number = random.random()
        cutoff = 0
        total = sum(predictions[0])
        predictions = list(map(lambda x: x / total, predictions[0]))
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
            second_letter = next_letter

        return result
