from mido import MidiFile


def main():
    notes = []
    mid = MidiFile('bwv10354.mid')
    for msg in mid:
        print(msg)
    print(notes)


main()
