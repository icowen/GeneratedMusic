import tensorflow as tf
import numpy as np
from WordConverter import WordConverter


def get_training_data():
    word_file = open('words.txt', 'r')
    words = []
    for w in word_file.readlines():
        words.append(w)
    word_file.close()
    converter = WordConverter(words)
    x_list, y_list = converter.get_converted_words()
    y_list = converter.convert_index_to_letter(y_list)
    return np.asarray(x_list), np.asarray(y_list)


x_train, y_train = get_training_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, np.asarray(y_train), epochs=1000)

#
# val_loss, val_acc = model.evaluate(x_train, y_train)
# print("val_loss: {0}; accuracy: {1}".format(val_loss, val_acc))
#
# predictions = model.predict([x_test])
#
# for i in range(len(predictions)):
#     print(np.argmax(predictions[i]))
