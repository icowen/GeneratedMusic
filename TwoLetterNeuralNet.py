import tensorflow as tf
import numpy as np
from WordConverter import WordConverter
import sys
from tensorflow.python.client import device_lib


np.set_printoptions(threshold=sys.maxsize)


# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
#
#
# print('--------------------Devices from session: ')
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(f'\n---------get_available_gpus: {get_available_gpus()}')


def get_training_data():
    global converter
    word_file = open('words.txt', 'r')
    words = []
    for w in word_file.readlines():
        words.append(w)
    word_file.close()
    converter = WordConverter(words)
    x_list, y_list = converter.get_input()
    return np.asarray(x_list), np.asarray(y_list)


x_train, y_train = get_training_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(27, activation=tf.nn.sigmoid))

# adam = tf.keras.optimizers.Adam(lr=.1)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# with tf.device('/device:GPU:0'):
model.fit(x_train, np.asarray(y_train), epochs=1000, batch_size=50)


val_loss, val_acc = model.evaluate(x_train, y_train)
print(f'val_loss: {val_loss}; accuracy: {val_acc}')

x_test = x_train[:50]
y_test = y_train[:50]

predictions = model.predict([x_test])

for i in range(len(predictions)):
    first_letter = converter.convert_index_to_ascii(x_test[i][:27].tolist())
    second_letter = converter.convert_index_to_ascii(x_test[i][27:].tolist())
    input_letters = f'{first_letter}{second_letter}'
    predicted_index = predictions[np.argmax(predictions[i])]
    predicted_letter = converter.convert_index_to_ascii(predicted_index)
    actual = converter.convert_index_to_ascii(y_test[i])
    print(f'Input: {input_letters} predicted_letter: {predicted_letter} acutal: {actual})')
    print(predictions[i], '\n')
