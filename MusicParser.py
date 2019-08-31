import binascii
import re
from mido import MidiFile
from Notes import NoteConverter
from os.path import dirname, join
import glob


# File used to look at midi files
def main():
    # get_notes_for_all_midis()
    get_notes_for_single_midi("MusicFiles/folies.mid")
    # get_binary_for_single_midi("MusicFiles/folies.mid")


def get_notes_for_all_midis():
    outfile = open("flute_notes.txt", "w")
    note_dict = NoteConverter.get_dict_with_number_as_key()
    file_names = glob.glob(join(dirname(__file__), "MusicFiles/", "*.mid"))
    for file in file_names:
        mid = MidiFile(file)
        notes = []
        for msg in mid:
            if not msg.is_meta:
                if msg.type == "note_on" and not msg.velocity == 0 and msg.channel == 0:
                    notes.append(msg.note)
        note_letters = []
        for note in notes:
            note_letters.append(note_dict.get(note))
        outfile.write(file + "\n")
        outfile.write(str(note_letters) + "\n")


def get_notes_for_single_midi(file_name):
    note_dict = NoteConverter.get_dict_with_number_as_key()
    notes = []
    note_letters = []
    mid = MidiFile(file_name)
    for msg in mid:
        if not msg.is_meta:
            if msg.type == "note_on" and not msg.velocity == 0 and msg.channel == 0:
                notes.append(msg.note)
    for note in notes:
        note_letters.append(note_dict.get(note))
    print(file_name)
    print(note_letters)


def get_binary_for_single_midi(file_name):
    with open(file_name, "rb") as text_file:
        binary = '0'     # Every midi file starts with an extra zero
        for line in text_file:
            binary += bin(int(binascii.hexlify(line), 16))[2:].zfill(8)

        #         header_index = binary.find('01001101010101000110100001100100')  # MThd
        #         track_index = binary.find('01001101010101000111001001101011')   # MTrk
        #         if header_index >= 0:
        #             print(MThd(binary[header_index:]).to_string())
        #         if track_index >= 0:
        #             print(MTrk(binary[track_index:]).to_string())
        print(re.sub(r'([01]{8})', r'\1 ', binary))
        print(re.findall(r'11111111\n[01]{8}', re.sub(r'([01]{8})', r'\1\n', binary)))
        print(len(binary) / 8)

# class MThd:
#     def __init__(self, binary):
#         self.binary = binary
#         self.length = int(binary[4*8:8*8], base=2)
#         self.format = int(binary[8*8:10*8], base=2)
#         self.tracks = int(binary[10*8:12*8], base=2)
#         self.division = int(binary[12*8:14*8], base=2)
#         print('Header:', re.sub(r'((0|1){8})', r'\1 ', binary[:8*8 + self.length*8]))
#
#     def to_string(self):
#         return str.format('Length: %(l)s, Format: %(f)s, Tracks: %(t)s, Division: %(d)s' %
#                           {'l': self.length, 'f': self.format, 't': self.tracks, 'd': self.division})
#
#
# class MTrk:
#     def __init__(self, binary):
#         self.binary = binary
#         self.length = int(binary[4*8:8*8], base=2)
#         print('Track:', re.sub(r'((0|1){8})', r'\1 ', binary[:8 * 8 + self.length * 8]))
#         self.event_list = get_event_list(binary[8*8:])
#
#     def to_string(self):
#         return str.format('Length: %(l)s, Event List: %(e)s' %
#                           {'l': self.length, 'e': self.event_list})
#
#
# class Event:
#     def __init__(self, binary):
#         self.delta_time, event_index = get_delta_time(binary)
#         self.event, self.length, self.text = get_event(binary[event_index:])
#         self.end = 16 + event_index + self.length
#
#     def to_string(self):
#         return str.format('Delta time: %(d)s, Event: %(e)s, Length: %(l)s, Text: %(t)s' %
#                           {'d': self.delta_time, 'e': self.event, 'l': self.length, 't': self.text})
#
#
# def get_delta_time(binary):
#     i = 0
#     delta_time = 0
#     if binary[i] == 1:
#         while binary[i] == 1 and i < len(binary):
#             delta_time += int(binary[i:i+8], base=2)
#             i += 8
#     delta_time = int(binary[i:i + 8])
#     return delta_time, i+8
#
#
# def get_length(binary):
#     i = 0
#     length = 0
#     if binary[i] == 1:
#         while binary[i] == 1 and i < len(binary):
#             length += int(binary[i:i + 8], base=2)
#             i += 8
#     length = int(binary[i:i + 8])
#     return length
#
#
# def get_event(binary):
#     event_type = str(hex(int(binary[:2*8], base=2))[2:])
#     length = get_length(binary[2*8:])
#     print("length:", length)
#     if length > 0:
#         text = str(hex(int(binary[3*8:4*8 + length], base=2))[2:])
#     else:
#         text = 'No text'
#     event = ''
#     print(event_type)
#     if event_type == 'ff03':
#         event = 'text_event'
#     if event_type == 'ff01':
#         event = 'track_name'
#     return event, length, text
#
#
# def get_event_list(binary):
#     i = 0
#     while i + 32 < len(binary):
#         event = Event(binary[i:])
#         print(event.to_string())
#         i = event.end


main()
