from mido import MidiFile
import NoteConverter


# File used to look at midi files
def main():
    notes = []
    channel = 0
    mid = MidiFile('MusicFiles/bwv10351.mid')

    for msg in mid:
        if not msg.is_meta:
            if msg.type == "note_on" and not msg.velocity == 0 and msg.channel == channel:
                notes.append(msg.note)

    note_dict = NoteConverter.get_dict()
    note_letters = []
    for note in notes:
        note_letters.append(note_dict.get(note))
    print('Notes on channel', channel)
    print(note_letters)


main()
