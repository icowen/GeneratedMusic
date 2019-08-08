<h1>Generated Music</h1>
This project reads flute solo files from various composers and predicts the next note of a sequence based on the previous five notes.

<h2>NeuralNetTest.py</h2>
This is the file that you would want to change in order to configure a neural net. In the class
<code>TestLETTERS</code>, run the <code>test_run_net</code> method. You can change the size of the net,
the learning rate, activation function, input file, number of epochs, and anything else
from this one method. You may need to configure this class by:

1. Navigate to <i>Run | Edit Configurations</i> in the top left.
2. Click the <code>+ | PythonTests | Nosetest</code> in the top left of the window.
    2. If that is not an option, run <code>pip install nose</code> in the terminal and then restart WebStorm.
3. Then add a name for the configuration
4. Choose <code>Script Path</code> and click the folder on the right
5. Choose <code>NeuralNetTest.py</code>.
6. Click save and exit the window.
7. In the top right next to the play button, choose the name of your configuration from the dropdown.
8. You can now run the test by click the play button, or going to <i>Run | Run {name of your configuration}</i> or pressing <code>Shift-F10</code>

If other tests are still running, add <code>@unittest.skip</code> above the class name.

<h2> How to Install TensorFlow</h1>

1. Run <code>pip install --upgrade pip</code>.
2. Run <code>pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl</code>.
3. Run <code>pip install keras</code>.