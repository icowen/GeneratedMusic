def get_dict():
    note_dict = dict()
    note = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = []
    for i in range(-1, 10):
        octave.append(str(i))
    for i in range(128):
        note_dict[i] = note[i % 12] + octave[i // 12]
    return note_dict
