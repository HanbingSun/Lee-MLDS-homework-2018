import tensorflow.compat.v1 as tf
import numpy as np
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 128

dataset = np.load('.\\mnist.npz')

LEN = len(dataset['x_train'])
N_BATCH = math.ceil(LEN / BATCH_SIZE)
EPOCHS = 100


def data_process(dataset):
    train_data = dataset['x_train']
    train_label = dataset['y_train']
    test_data = dataset['x_test']
    test_label = dataset['y_test']
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset


def build_model(switch):
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28))]
    )
    if switch == 'deep':
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    elif switch == 'shadow':
        model.add(tf.keras.layers.Dense(units=8, activation='relu'))
        model.add(tf.keras.layers.Dense(units=117, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    else:
        model.add(tf.keras.layers.Dense(units=10))
        model.add(tf.keras.layers.Dense(units=20, activation='relu'))
        model.add(tf.keras.layers.Dense(units=13, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy']
                  )
    return model


class History(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def draw(x, y, label):
    y_2 = []
    for i in range(EPOCHS):
        y_2.append(np.mean(y[i * N_BATCH:(i + 1) * N_BATCH]))
    if label == 'deep':
        plt.plot(x, y_2, color='r')
    elif label == 'middle':
        plt.plot(x, y_2, color='b')
    else:
        plt.plot(x, y_2, color='g')
    plt.legend(['deep','middle','shadow'])


model_deep = build_model('deep')
model_shadow = build_model('shadow')
model_middle = build_model('middle')

train_dataset, test_dataset = data_process(dataset)

history_deep = History()
model_deep.fit(train_dataset, epochs=EPOCHS, callbacks=[history_deep])
history_middle = History()
model_middle.fit(train_dataset, epochs=EPOCHS, callbacks=[history_middle])
history_shadow = History()
model_shadow.fit(train_dataset, epochs=EPOCHS, callbacks=[history_shadow])
x = range(1, EPOCHS + 1)
y = history_deep.losses
draw(x, y,'deep')
y = history_middle.losses
draw(x, y,'middle')
y = history_shadow.losses
draw(x, y,'shadow')
plt.show()
