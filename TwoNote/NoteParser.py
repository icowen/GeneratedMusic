import numpy as np
import re
import NoteConverter


class NoteParser:
    def __init__(self, input_data):
        self.input_data = input_data
        self.converted_songs = self.convert_to_notes_by_song(input_data)
        self.first_two_notes = []
        self.next_notes = []
        self.convert_into_keras_input_lists()

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
        return converted_songs

    @staticmethod
    def convert_note_to_array(note):
        index = NoteConverter.get_dict_with_letter_as_key().get(note)
        note_as_array = np.zeros((128,), dtype=int)
        note_as_array[index] = 1
        return note_as_array

    def convert_into_keras_input_lists(self):
        first_two_notes = []
        next_notes = []
        for song in self.converted_songs:
            song_first_notes = []
            song_next_notes = []
            for i in range(len(song) - 2):
                song_first_notes.append(
                    np.concatenate([
                        song[i],
                        song[i + 1]
                    ])
                )
                song_next_notes.append(song[i + 2])
            first_two_notes.append(song_first_notes)
            next_notes.append(song_next_notes)
        self.first_two_notes = first_two_notes
        self.next_notes = next_notes
