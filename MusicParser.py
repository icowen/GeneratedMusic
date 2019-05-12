from mido import MidiFile


# File used to look at midi files
def main():
    notes = []
    mid = MidiFile('MusicFiles/flute pan1.mid')
    for msg in mid:
        if not msg.is_meta:
            if msg.type == "note_on":
                notes.append(msg.note)
        print(msg)
    print(notes)


main()
