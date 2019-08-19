import tensorflow as tf
import numpy as np
from WordConverter import WordConverter


def set_up_converter():
    word_file = open('words.txt', 'r')
    words = []
    for w in word_file.readlines():
        words.append(w)
    word_file.close()
    return WordConverter(words)


converter = set_up_converter()
x_list, y_list = converter.get_converted_words()
y_list = converter.convert_index_to_letter(y_list)
x_train = np.asarray(x_list)
y_train = np.asarray(y_list)

x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, np.asarray(y_train), epochs=10000)

#
# val_loss, val_acc = model.evaluate(x_train, y_train)
# print("val_loss: {0}; accuracy: {1}".format(val_loss, val_acc))
#
# predictions = model.predict([x_test])
#
# for i in range(len(predictions)):
#     print(np.argmax(predictions[i]))


