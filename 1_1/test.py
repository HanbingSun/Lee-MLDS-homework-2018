import tensorflow.compat.v1 as tf
import numpy as np
import math
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

tf.disable_eager_execution()

BATCH_SIZE = 128

dataset = np.load('.\\mnist.npz')

LEN = len(dataset['x_train'])
N_BATCH = math.ceil(LEN / BATCH_SIZE)
EPOCHS = 100
SAMPLE_PERIOD = 3
SAVE_PATH = r'E:\THU\postgraduate2\deep_learning\Lee_tuturioals\hw\hw1\1_1'


def data_process(dataset):
    train_data = dataset['x_train']
    train_label = dataset['y_train']
    test_data = dataset['x_test']
    test_label = dataset['y_test']
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, test_dataset, train_data, train_label, test_data, test_label


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
    elif switch == 'middle':
        model.add(tf.keras.layers.Dense(units=10))
        model.add(tf.keras.layers.Dense(units=20, activation='relu'))
        model.add(tf.keras.layers.Dense(units=13, activation='relu'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    elif switch == 'easy':
        model.add(tf.keras.layers.Dense(units=20, activation='relu'))
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

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


model_deep = build_model('deep')
model_shadow = build_model('shadow')
model_middle = build_model('middle')
model_easy=build_model('easy')

train_dataset, test_dataset, train_inputs, train_outputs, test_inputs, test_outputs = data_process(dataset)


def hw1():
    def draw1(x, y, label):

        if label == 'deep':
            plt.plot(x, y, color='r')
        elif label == 'middle':
            plt.plot(x, y, color='b')
        else:
            plt.plot(x, y, color='g')
        plt.legend(['deep', 'middle', 'shadow'])

    history_deep = History()
    model_deep.fit(train_dataset, epochs=EPOCHS, callbacks=[history_deep])
    history_middle = History()
    model_middle.fit(train_dataset, epochs=EPOCHS, callbacks=[history_middle])
    history_shadow = History()
    model_shadow.fit(train_dataset, epochs=EPOCHS, callbacks=[history_shadow])
    x = range(1, EPOCHS + 1)
    y = history_deep.losses
    draw1(x, y, 'deep')
    y = history_middle.losses
    draw1(x, y, 'middle')
    y = history_shadow.losses
    draw1(x, y, 'shadow')
    plt.show()


def hw2():
    TRAIN_TIMES = 8

    def train_model(k, isOverWritten=True):
        if isOverWritten:
            shutil.rmtree('.\\temp_model')
            os.mkdir('.\\temp_model')
            write_append_switch = 'w'
        else:
            write_append_switch = 'a'

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='.\\temp_model\\ep{epoch:03d}-loss{loss:.3f}.h5',
            save_best_only=True,
            monitor='loss',
            peroid=1
        )
        self_history = History()
        history = model_easy.fit(train_dataset,
                                   epochs=EPOCHS,
                                   callbacks=[checkpoint, self_history])
        model_easy.save('.\\model_of_hw1_{}.h5'.format(k))
        with open('losses.txt', write_append_switch) as f:
            f.truncate()
            for loss in self_history.losses:
                f.write(str(loss) + '\n')
        with open('history.txt', write_append_switch) as f:
            f.truncate()
            for keys, values in history.history.items():
                f.write(keys + '\t')
                for i in range(len(history.history[keys])):
                    f.write(str(history.history[keys][i]) + '\t')
                f.write('\n')

    def get_weights(FILE_PATH):
        weights = []
        files = os.listdir(FILE_PATH)
        for file in files:
            model = tf.keras.models.load_model(FILE_PATH + file)
            weights.append(model.get_weights())
        return weights

    def get_losses(FILE_PATH, sampling=True):
        float_losses = []
        with open(FILE_PATH) as f:
            losses = f.readlines()
        tag = SAMPLE_PERIOD
        temp = 0
        if sampling == True:
            for i in range(len(losses)):
                # losses[i]=losses[i].replace('\n', '')
                temp += float(losses[i])
                tag = tag - 1
                if tag == 0:
                    temp = temp / SAMPLE_PERIOD
                    float_losses.append(temp)
                    temp = 0
                    tag = SAMPLE_PERIOD
        else:
            for i in range(len(losses)):
                float_losses.append(float(losses[i]))
        return float_losses

    def process_weights(weights):
        flatten_vector = []
        for i in range(len(weights)):
            weight = weights[i]
            temp = np.zeros([1, 1], dtype='float32')
            for j in range(len(weight)):
                weight[j] = np.reshape(weight[j], (1, -1))
                temp = np.concatenate((temp, weight[j]), 1)
            temp = temp[0:, 1:]
            flatten_vector.append(temp)
        l = (flatten_vector[0].shape[1])
        data = np.zeros([1, l])
        for i in range(len(flatten_vector)):
            data = np.vstack((data, flatten_vector[i]))
        print(data.shape)
        pca = PCA(n_components=2)
        print(len(flatten_vector))
        new_data = pca.fit_transform(data)
        return new_data

    def scatter_plot(xy, z):
        x = xy[:, :1]
        y = xy[:, 1:]
        z = z[:x.shape[0]]
        ax = plt.subplot()
        ax.scatter(x, y)  # 绘制散点图，面积随机
        for i in range(len(z)):
            plt.annotate(round(z[i], 2), xy=(x[i], y[i]), xytext=(x[i] + 0.1, y[i] + 0.1))

    def hw1_2_1():
        for i in range(TRAIN_TIMES):
            train_model(i,isOverWritten=True)
            weights = get_weights('.\\temp_model\\')
            losses = get_losses('.\\losses.txt',sampling=True)
            pca_data = process_weights(weights)
            scatter_plot(pca_data, losses)
        plt.show()



    # def get_weight_grad(model, inputs, outputs):
    #     grads = (model.optimizer.get_gradients(loss=model.total_loss, params=model.trainable_weights))
    #     symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    #     f = tf.keras.backend.function(symb_inputs, grads)
    #     x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    #
    #     # print(sample_weight.shape)
    #     output_grad = f(x + y)
    #     return output_grad

    # def temp(outputs):
    #     new_out = np.zeros([len(outputs), 10])
    #     for i in range(len(outputs)):
    #         new_out[i, outputs[i]] = 1
    #     return new_out

    # 1-calculate-gradients-------------------------------------------------------------
    # x = tf.placeholder(tf.float32, shape=(None, 28,28))
    # y = model(x)
    # target=tf.placeholder(tf.float32,shape=(None,10))
    # loss=tf.keras.backend.sum(tf.keras.backend.square(target-y))
    # # grads=model.optimizer.get_gradients(loss,model.trainable_weights)
    # grads=tf.gradients(loss,model.trainable_weights)
    # sess=tf.InteractiveSession()
    # init=tf.global_variables_initializer()
    # sess.run(init)
    # sess.run(grads,{x:train_inputs,target:temp(train_outputs)})
    # 2-unused--------------------------------------------------------------------------
    # gradients = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    # input_tensors = [model.inputs[0],  # input data
    #                  model.outputs[0],  # labels
    #                  model.trainable_weights,
    #                  tf.keras.backend.learning_phase(),  # train or test mode
    #                  ]
    # get_gradients = tf.keras.backend.function(inputs=input_tensors, outputs=gradients)
    # input = [
    #     train_inputs,  # X
    #     train_outputs,  # y
    #     model.get_weights(),
    #     1           # learning phase in TRAIN mode
    # ]
    # print (model.trainable_weights, get_gradients(input))
    # 3-a-problem-exists-why-the-result-doesn`t-match-predict---------------------------

    # x = tf.placeholder(shape=(None, 28, 28),dtype=tf.float32)
    # y = model(x)
    # gradients=model.optimizer.get_gradients(model.total_loss,model.trainable_weights)
    # sess=tf.InteractiveSession()
    # init=tf.global_variables_initializer()
    # sess.run(init)
    # print(sess.run(y[:5],feed_dict={x:test_inputs}))
    # print(test_outputs[:5])

    def get_gradients():
        FILE_PATH = '.\\temp_model\\'
        with open('.\\gradients.txt', 'w') as f:
            for file in os.listdir(FILE_PATH):
                model = tf.keras.models.load_model(FILE_PATH + file)
                x = tf.placeholder(tf.float32, (None, 28, 28))
                y = model(x)
                label = train_outputs
                loss = tf.keras.losses.SparseCategoricalCrossentropy()(label, y)
                grads = model.optimizer.get_gradients(loss, model.trainable_weights)

                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                # print(sess.run(grads,feed_dict={x:train_inputs}))
                res = sess.run(grads, feed_dict={x: train_inputs})
                all_grad = 0
                for i in range(len(res)):
                    temp = np.sum(res[i] * res[i])
                    all_grad += temp
                # print(all_grad)
                all_grad = math.sqrt(all_grad)
                f.write(str(all_grad) + '\n')
    def plot_gradients_loss(gradients,epochs):
        plt.plot(epochs,gradients)
        # plt.show()
    def hw1_2_2():
        get_gradients()
        all_grads=[]
        with open('.\\gradients.txt','r') as f:
            temp=f.readlines()
            print(temp)
            print(len(temp))
            for i in range(len(temp)):
                all_grads.append(float(temp[i]))
        epochs=range(len(all_grads))
        plot_gradients_loss(all_grads,epochs)


    hw1_2_1()
    # hw1_2_2()
    plt.show()


hw2()
