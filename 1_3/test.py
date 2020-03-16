import tensorflow.compat.v1 as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Layer

DATA_PATH = r'E:\THU\postgraduate2\tensorflow\dataset\mnist\datasets\mnist.npz'
EPOCHS = 10
BATCH_SIZE1 = 68
BATCH_SIZE2 = 100


def process_data(FILE_PATH, BATCH_SIZE):
    data = np.load(FILE_PATH)
    train_data = data['x_train']
    train_label = data['y_train']
    test_data = data['x_test']
    test_label = data['y_test']
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    train_label_copy = train_label

    # np.random.shuffle(train_label)

    random_train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    random_train_dataset = random_train_dataset.shuffle(1000).batch(BATCH_SIZE)
    return train_dataset, test_dataset, train_data, train_label_copy, test_data,test_label,random_train_dataset


def build_model(params=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=110))
    model.add(tf.keras.layers.Dense(units=136))
    for i in range(params - 1):
        model.add(tf.keras.layers.Dense(units=10, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model


class History(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))


train_dataset, test_dataset, train_data, train_label,test_data,test_label, random_train_dataset = process_data(DATA_PATH, BATCH_SIZE1)


# correct_history=model.fit(train_dataset,epochs=EPOCHS，validation_data=test_dataset,callbacks=[history])
# record_history(correct_history,'.\\correct_history.txt')
# model.save('.\\correct_model.h5')

def record_dictionary(data, FILE_PATH):
    with open(FILE_PATH, 'w') as f:
        for key, value in data.items():
            f.write(str(key) + '\t')
            for j in range(len(value)):
                f.write(str(value[j]) + '\t')
            f.write('\n')


def read_record_dictionary(FILE_PATH):
    with open(FILE_PATH) as f:
        params = f.readlines()
        data_dict = {}
        for i in range(len(params)):
            temp = params[i]
            temp = temp.split("\t")
            data_dict[temp[0]] = []
            j = 1
            print(len(temp))
            for j in range(len(temp) - 2):
                print(temp[j + 1])
                data_dict[temp[0]].append(float(temp[j + 1]))
        print(data_dict)
        return data_dict


def subtask1():
    def train():
        history = History()
        model = build_model()
        random_history = model.fit(random_train_dataset,
                                   epochs=EPOCHS,
                                   validation_data=test_dataset,
                                   callbacks=[history])
        record_dictionary(random_history.history, '.\\random_history.txt')
        model.save('.\\random_model.h5')

    def plot_image(x, train_y, test_y):
        train_line = plt.plot(x, train_y)
        test_line = plt.plot(x, test_y)
        plt.legend(handles=[train_line, test_line], labels=['r:train_data', 'y:test_data'])

    train()
    FILE_PATH = '.\\random_history.txt'
    history_record = read_record_dictionary(FILE_PATH)
    x = range(EPOCHS)
    plot_image(x, history_record['loss'], history_record['val_loss'])
    plt.show()


def subtask2():
    models = []
    test_res = []
    train_res = []
    number_models = 50
    history = History()

    def train_models():
        for i in range(number_models):
            models.append(build_model(params=i + 1))
            models[i].fit(train_dataset,
                          epochs=EPOCHS,
                          callbacks=[history])
            test_res.append(models[i].evaluate(test_dataset))
            train_res.append([history.losses[EPOCHS - 1], history.acc[EPOCHS - 1]])

    def record_loss_acc(FILE_PATH, test_res, train_res):
        with open(FILE_PATH, 'w') as f:
            for i in range(len(test_res)):
                f.write(str(test_res[i][0]) + '\t' + str(test_res[i][1]) + '\t' + str(train_res[i][0]) + '\t' + str(
                    train_res[i][1]))
                f.write('\n')

    def read_loss_acc(FILE_PATH):
        with open(FILE_PATH, 'r') as f:
            temp = f.readlines()
            for i in range(len(temp)):
                a_set_data = temp[i].split('\t')
                test_loss.append(float(a_set_data[0]))
                test_acc.append(float(a_set_data[1]))
                train_loss.append(float(a_set_data[2]))
                train_acc.append(float(a_set_data[3]))

    def plot_images():
        x = range(number_models)
        plt.subplot(1, 2, 1)
        plt.title('train&test losses with params')

        plt.plot(x, train_loss, 'r', label='train')
        plt.plot(x, test_loss, 'b', label='test')
        plt.subplot(1, 2, 2)
        plt.title('train&test acc with params')
        plt.plot(x, train_acc, 'y', label='train')
        plt.plot(x, test_acc, 'g', label='test')
        plt.show()

    FILE_PATH = '.\\params_change.txt'
    train_models()
    record_loss_acc(FILE_PATH, test_res, train_res)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    read_loss_acc(FILE_PATH)
    plot_images()


class MyLayer(Layer):
    weights, bias = None, None
    output_dim = None

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.gamma = self.weights
        self.beta = self.bias
        self.trainable_weights = [self.weights, self.bias]
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        out = K.dot(x, self.gamma) + self.beta
        print(self.gamma.shape)
        print(self.beta.shape)
        print(out.shape)

        # out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def subtask3():
    number_models = 5

    def train():
        history = History()
        train_dataset_larger_batch, test_dataset_larger_batch,_,_,_,_,_ = process_data(DATA_PATH, BATCH_SIZE2)
        model_larger_batch = build_model(params=1)
        model_smaller_batch = build_model(params=1)
        history_l = model_larger_batch.fit(train_dataset_larger_batch,
                                           epochs=EPOCHS,
                                           validation_data=test_dataset_larger_batch,
                                           callbacks=[history]
                                           )
        history_s = model_smaller_batch.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset
        )
        model_larger_batch.save('.\\model_larger_batch.h5')
        model_smaller_batch.save('.\\model_smaller_batch.h5')

    def fuck(y):
        y_ = np.zeros([len(y), 10])
        for i in range(len(y)):
            y_[i][y[i]] = 1

        return y_

    def softmax(x):
        sum_raw = np.sum(np.exp(x), axis=-1)
        x1 = np.ones(np.shape(x))
        for i in range(np.shape(x)[0]):
            x1[i] = np.exp(x[i]) / sum_raw[i]
        return x1

    def relu(data):
        new_data = (data + abs(data)) / 2 + 0.0001
        return new_data
    def pass_model(weights,data):
        y1 = np.reshape(data, (data.shape[0], 784))
        y2 = relu(np.dot(y1, weights[0]) + weights[1])
        y3 = relu(np.dot(y2, weights[2]) + weights[3])
        y4 = relu(np.dot(y3, weights[4]) + weights[5])
        return y4


    def calculate_crossentropy():
        model_larger_batch = tf.keras.models.load_model('.\\model_larger_batch.h5')
        model_smaller_batch = tf.keras.models.load_model('.\\model_smaller_batch.h5')
        weights_l = model_larger_batch.get_weights()
        weights_s = model_smaller_batch.get_weights()


        train_res=[]
        test_res=[]
        alphas=np.arange(-20,10,1)
        train_y = fuck(train_label)
        test_y=fuck(test_label)

        for i in range(len(alphas)):
            alpha=alphas[i]
            print(alpha)
            weights_m=[]

            for i in range(len(weights_l)):
                weights_m.append(alpha * weights_s[i] + (1 - alpha) * weights_l[i])

            # my_model = keras.models.Sequential()
            # my_model.add(keras.layers.Flatten(input_shape=(28, 28)))
            # real_model=build_model(params=1)
            # my_layer = MyLayer()
            # my_layer.output_dim=110
            # my_layer.weights = weights_m[0]
            # my_layer.bias = weights_m[1]
            # my_layer.build((None, 784))
            # # my_model.add(my_layer)
            #
            #
            # my_layer2 = MyLayer()
            # my_layer2.output_dim=136
            # my_layer2.weights = weights_m[2]
            # my_layer2.bias = weights_m[3]
            # my_layer2.build((None,110))
            # my_model.add(my_layer)
            # my_model.add(my_layer2)
            #
            # my_model.summary()




            train_y_pred=pass_model(weights_m,train_data)
            test_y_pred=pass_model(weights_m,test_data)
            train_y_pred=softmax(train_y_pred)
            test_y_pred=softmax(test_y_pred)
            E1 = -np.mean(train_y * np.log(train_y_pred), -1)
            E2 = -np.mean(test_y * np.log(test_y_pred), -1)

            train_res.append(np.mean(E1))
            test_res.append(np.mean(E2))
            print('E1',E1)
            print('E2',E2)
        return train_res,test_res,alphas
    def plot_cross_entropy(x,train_y,test_y):
        plt.plot(x,train_y)
        plt.plot(x,test_y)
        plt.show()
    # train()
    train_y,test_y,x=calculate_crossentropy()
    plot_cross_entropy(x,train_y,test_y)


subtask3()
