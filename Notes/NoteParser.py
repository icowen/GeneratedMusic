import numpy as np
import re
from Notes import NoteConverter


class NoteParser:
    def __init__(self,
                 input_data,
                 number_of_initial_notes=2,
                 note_type=None):
        self.input_data = input_data
        if note_type and ('volume' in note_type):
            self.converted_songs = self.convert_to_notes_with_volume_by_song(input_data)
        else:
            self.converted_songs = self.convert_to_notes_by_song(input_data)
        self.first_notes, self.next_notes = \
            self.convert_into_keras_input_lists(number_of_initial_notes)

    def convert_to_notes_by_song(self, input_text_file):
        song_iterator = re.finditer('\[.*\]', input_text_file)
        cleaned_songs = []
        for song in song_iterator:
            clean = re.sub('[\[\'\]\s]', '', song.group())
            cleaned_songs.append(clean)
        return self.convert_alphabet_notes_into_array(cleaned_songs)

    def convert_alphabet_notes_into_array(self, songs):
        converted_songs = []
        for song in songs:
            converted_song = []
            for note in song.split(sep=','):
                converted_song.append(self.convert_note_to_array(note))
            converted_songs.append(converted_song)
        return np.asarray(converted_songs)

    @staticmethod
    def convert_note_to_array(note):
        index = NoteConverter.get_dict_with_letter_as_key().get(note)
        note_as_array = np.zeros((128,), dtype=int)
        note_as_array[index] = 1
        return np.asarray(note_as_array)

    def convert_into_keras_input_lists(self, number_of_initial_notes):
        first_notes = []
        next_notes = []
        for song in self.converted_songs:
            song_first_notes = []
            song_next_notes = []
            for i in range(len(song) - number_of_initial_notes):
                notes = []
                for j in range(number_of_initial_notes):
                    notes = np.concatenate([notes, song[i + j]])
                song_first_notes.append(np.asarray(notes))
                song_next_notes.append(song[i + number_of_initial_notes])
            first_notes.append(np.asarray(song_first_notes))
            next_notes.append(np.asarray(song_next_notes))
        return np.asarray(first_notes), np.asarray(next_notes)

    @staticmethod
    def convert_array_into_note(note):
        index = np.argmax(note)
        next_note = NoteConverter.get_dict_with_number_as_key().get(index)
        return next_note

    @staticmethod
    def convert_to_notes_with_volume_by_song(input_data):
        data = []
        for song in input_data.split('-'):
            song_arr = []
            for note in song.split('\n'):
                note_arr = []
                for i in note.split(','):
                    if i:
                        note_arr.append(float(i))
                song_arr.append(note_arr)
            data.append(song_arr)
        return np.asarray(data)
