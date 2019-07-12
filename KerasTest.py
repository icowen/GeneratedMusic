import tensorflow as tf
import numpy as np

x_train = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0], [2, 3], [2, 4], [2, 2], [3, 3]])
y_train = np.asarray([1, 0, 1, 0, 1, 0, 0, 0])

x_test = np.asarray([[1, 0], [2, 4], [4, 1], [3, 0], [1, 2], [4, 4]])
y_test = np.asarray([1, 0, 1, 0, 1, 0])

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, np.asarray(y_train), epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("val_loss: {0}; accuracy: {1}".format(val_loss, val_acc))

predictions = model.predict([x_test])

for i in range(len(predictions)):
    print(np.argmax(predictions[i]))
