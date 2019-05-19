<h1>Generated Music</h1>
This project reads flute solo files from various composers and predicts the next note of a sequence based on the previous five notes.

<h2>MidiParser.py</h2>
This file takes in a midi file and displays the notes in chronological order.

To use this file, open the terminal (select <b>View | Tool Windows | Terminal</b>) then:
1. Run <code>pip install mido</code>.

<h2>MidiScraper.py</h2>
This file downloaded all of the .mid files in the <code>GeneratedMusic/MusicFiles</code> directory. 

To use this file, open the terminal (select <b>View | Tool Windows | Terminal</b>) then:
1. Run <code>pip install selenium</code>.
2. Download <code>ChromeDriver 74.0.3729.6</code> (or whatever version of chrome you use) from <a href>http://chromedriver.chromium.org/downloads</a>.
3. In environment variables on your computer, make sure the <code>path</code> variable is linked to where you just downloaded and saved <code>chromedriver.exe</code>.
4. In MidiScraper, put the path to your chromedriver in <code>driver = webdriver.Chrome("your path to chromedriver")</code>.

You can use this file as a template to get new midi files as well.

<h2> How to Install TensorFlow</h1>

1. Run <code>pip install --upgrade pip</code>.
2. Run <code>pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl</code>.
3. Run <code>pip install keras</code>.