note = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
octave = []
for i in range(-1, 10):
    octave.append(str(i))


def get_dict_with_number_as_key():
    note_dict = dict()
    for j in range(128):
        note_dict[j] = note[j % 12] + octave[j // 12]
    return note_dict


def get_dict_with_letter_as_key():
    note_dict = dict()
    for j in range(128):
        note_dict[note[j % 12] + octave[j // 12]] = j
    return note_dict
