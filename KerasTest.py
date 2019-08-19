import tensorflow as tf
import numpy as np
import sys
from WordConverter import WordConverter

# x_train = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0], [2, 3], [2, 4], [2, 2], [3, 3]])
# y_train = np.asarray([1, 0, 1, 0, 1, 0, 0, 0])

x_test = np.asarray([[1, 0], [2, 4], [4, 1], [3, 0], [1, 2], [4, 4]])
# y_test = np.asarray([1, 0, 1, 0, 1, 0])

word_file = open('words.txt', 'r')
words = []
for w in word_file.readlines():
    words.append(w)
word_file.close()
sys.stderr.write(f'Input of {len(words)} words.')
converter = WordConverter(words)

x_list, y_list = converter.get_converted_words()
x_train = np.asarray(x_list)
y_train = np.asarray(y_list)
print(f'test: {x_test}')
print(f'train: {x_train}')


# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, np.asarray(y_train), epochs=3)
#
# val_loss, val_acc = model.evaluate(x_train, y_train)
# print("val_loss: {0}; accuracy: {1}".format(val_loss, val_acc))
#
# predictions = model.predict([x_test])
#
# for i in range(len(predictions)):
#     print(np.argmax(predictions[i]))
