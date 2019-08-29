import random
import tensorflow as tf
import numpy as np

from TwoNote import NoteConverter
from TwoNote.NoteParser import NoteParser


class TwoNoteNeuralNet:
    random.seed(1)
    np.random.seed(2)
    tf.set_random_seed(3)

    def __init__(self,
                 number_of_epochs=10000,
                 batch_size=50,
                 input_file='flute_notes.txt',
                 save_model=False):
        self.model = tf.keras.models.Sequential()
        self.input_file = input_file
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.x_train, self.y_train = self.get_training_data()
        self.save_model = save_model
        self.model_save_h5 = 'TwoNote/TwoNoteMusic.h5'
        self.weights_save_h5 = 'TwoNote/TwoNoteMusic.h5'
        self.model_save_json = 'TwoNote/TwoNoteMusic.json'
        self.set_up_model()

    def set_up_model(self):
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(100,
                                             activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(128,
                                             activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_training_data(self):
        note_file = open(self.input_file, 'r')
        parser = NoteParser(note_file.read())
        first_notes_by_song = parser.first_two_notes
        next_notes_by_song = parser.next_notes
        note_file.close()
        return first_notes_by_song, next_notes_by_song

    def train(self):
        for epoch in range(self.number_of_epochs):
            for x, y in zip(self.x_train[:10], self.y_train[:10]):
                self.model.fit(x,
                               y,
                               epochs=self.number_of_epochs,
                               batch_size=self.batch_size)
        if self.save_model:
            self.model.save(self.model_save_h5)
            self.save_model_to_json()

    def predict(self):
        for i in range(25):
            print(f'Predicted: '
                  f'{self.generate_next_note(self.x_train[0][i])}; '
                  f'Acutal: {NoteParser.convert_array_into_note(self.y_train[0][i])}')

    def generate_next_note(self, notes):
        normalized_input = self.get_correct_input_shape(notes)
        predictions = self.get_predictions(normalized_input)
        random_number = random.random()
        cutoff = 0
        for i in range(len(predictions)):
            note = predictions[i]
            cutoff += note
            if random_number <= cutoff:
                return NoteConverter.get_dict_with_number_as_key()[i]

    @staticmethod
    def get_correct_input_shape(notes):
        return np.asarray(tf.keras.utils.normalize(notes))

    def get_predictions(self, normalized_input):
        return self.normalize_predictions(
            self.model.predict([normalized_input[0:]])
        )

    @staticmethod
    def normalize_predictions(predictions):
        total = sum(predictions[0])
        return list(map(lambda x: x / total, predictions[0]))

    def save_model_to_json(self):
        model_json = self.model.to_json()
        with open(self.model_save_json, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.weights_save_h5)
        print('Saved model to disk')

