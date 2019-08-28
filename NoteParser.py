import numpy as np
import re
import NoteConverter


class NoteParser:
    def __init__(self, input_data):
        self.input_data = input_data
        self.converted_songs = self.convert_to_notes_by_song(input_data)

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
        note_as_array = np.zeros((128, ), dtype=int)
        note_as_array[index] = 1
        return note_as_array
