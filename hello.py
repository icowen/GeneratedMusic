import pysynth
from playsound import playsound

test = (('f#1', 4), ('e', 4), ('g', 4), ('c5', 1))
pysynth.make_wav(test, fn="test.wav")

playsound('test.wav')
