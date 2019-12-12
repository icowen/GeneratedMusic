import random
# import pysynth
import random
import sys

import numpy as np
# import pysynth
import pandas as pd
# from midi2audio import FluidSynth
# from playsound import playsound
import pretty_midi
import tensorflow as tf

from Notes import NoteConverter
from Notes.NoteParser import NoteParser

np.set_printoptions(threshold=sys.maxsize)


class NoteNeuralNet:
    random.seed(1)
    np.random.seed(2)
    tf.set_random_seed(3)

    def __init__(self,
                 number_of_epochs=10000,
                 batch_size=50,
                 input_file='flute_notes.txt',
                 input_file_with_notes_as_arr=None,
                 save_model=False,
                 number_of_previous_notes=2,
                 wav_file_name='generatedMusic.wav',
                 model_save_h5='NoteMusic.h5',
                 weights_save_h5='NoteMusicWeights.h5',
                 model_save_json='NoteMusic.json'
                 ):
        self.input_file_with_notes_as_arr = input_file_with_notes_as_arr
        self.model = tf.keras.models.Sequential()
        self.number_of_previous_notes = number_of_previous_notes
        self.input_file = input_file
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.x_train, self.y_train = self.get_training_data()
        self.save_model = save_model
        self.wav_file_name = wav_file_name
        self.model_save_h5 = model_save_h5
        self.weights_save_h5 = weights_save_h5
        self.model_save_json = model_save_json
        self.set_up_model()

    def set_up_model(self):
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(100,
                                             activation=tf.nn.sigmoid))
        if self.input_file_with_notes_as_arr:
            num_of_ouput_neurons = 130
        else:
            num_of_ouput_neurons = 128
        self.model.add(tf.keras.layers.Dense(num_of_ouput_neurons,
                                             activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_training_data(self):
        if self.input_file_with_notes_as_arr:
            note_file = open(self.input_file_with_notes_as_arr)
            parser = NoteParser(note_file.read(),
                                self.number_of_previous_notes
                                # ,['volume', 'length']
                                )
        else:
            note_file = open(self.input_file, 'r')
            parser = NoteParser(note_file.read(),
                                self.number_of_previous_notes)
        first_notes_by_song = parser.first_notes
        next_notes_by_song = parser.next_notes
        note_file.close()
        pd.DataFrame(first_notes_by_song[0], dtype=int).to_json('input_notes.json', orient='values')
        pd.DataFrame(next_notes_by_song[0], dtype=int).to_json('output_notes.json', orient='values')
        return first_notes_by_song, next_notes_by_song

    def train(self):
        self.model.fit(self.x_train[0],
                       self.y_train[0],
                       epochs=self.number_of_epochs,
                       batch_size=self.batch_size)
        # tfjs.converters.save_keras_model(self.model, "C:/Users/ian_c/music-website/src/JSModelConfiguration")
        # if self.save_model:
        #     self.model.save(self.model_save_h5)
        #     self.save_model_to_json()

    def predict(self):
        song_input = self.x_train[0]
        new_song = []
        for i in range(len(song_input)):
            predicted = self.generate_next_note(song_input[i]).lower()
            note_with_duration = (predicted, 4)
            new_song.append(note_with_duration)

        # pysynth.make_wav(new_song, fn=self.wav_file_name)
        # playsound(self.wav_file_name)

    def predict_with_length_and_volume(self):
        flute_chord = pretty_midi.PrettyMIDI()
        flute_program = pretty_midi.instrument_name_to_program('Flute')
        flute = pretty_midi.Instrument(program=flute_program)
        song_input = self.x_train[0]
        time = 0
        for i in range(len(song_input) - 1):
            note_name = self.generate_next_note(song_input[i]).lower()
            note_number = pretty_midi.note_name_to_number(note_name)
            note_length = song_input[i][-1]
            note_volume = int(song_input[i][-2])
            note = pretty_midi.Note(
                velocity=note_volume,
                pitch=note_number,
                start=time,
                end=note_length)
            time += note_length
            flute.notes.append(note)
        flute_chord.instruments.append(flute)
        flute_chord.write('temp.mid')

    def generate_next_note(self, notes):
        normalized_input = self.get_correct_input_shape(notes)
        predictions = self.get_predictions(normalized_input)
        random_number = random.random()
        cutoff = 0
        for i in range(len(predictions)):
            note = predictions[i]
            cutoff += note
            if random_number <= cutoff:
                if i < 21:
                    return NoteConverter.get_dict_with_number_as_key()[i + 24]
                if i > 108:
                    return NoteConverter.get_dict_with_number_as_key()[i - 24]
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
