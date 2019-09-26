import csv
import pretty_midi
import glob
import Notes.NoteConverter as NoteConverter
import numpy as np


def convert_note_to_array(note_pitch):
    note_as_array = np.zeros((128,), dtype=int)
    note_as_array[note_pitch] = 1
    return np.asarray(note_as_array)


with open('flute_notes_with_volume_and_length.csv', 'w', newline='\n') as out_file:
    wr = csv.writer(out_file)
    noteConverter = NoteConverter.get_dict_with_number_as_key()
    for song in glob.glob("C:\\Users\\ian_c\\GeneratedMusic\\MusicFiles\\*.mid")[:3]:
        output_notes = []
        midi_pretty_format = pretty_midi.PrettyMIDI(song)
        for instrument in midi_pretty_format.instruments:
            for note in instrument.notes:
                note_array = list(convert_note_to_array(note.pitch))
                volume = note.velocity
                length = note.end - note.start
                note_array.extend([volume, length])
                wr.writerow(note_array)
        wr.writerow('-')
