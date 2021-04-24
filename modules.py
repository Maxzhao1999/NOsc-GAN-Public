
####### DEFINE ALL FUNCTIONS ######
import uproot
from PIL import Image
from os import listdir
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.constraints import Constraint
from tqdm import tqdm
import datetime
import os
import cProfile
import pstats
import random
import time
import timeit
import pathlib
import shutil
import re
import io
import glob
import PIL
import PIL.Image
# %%


def get_dataset(saving=False, plotting=False):
    """
    Loads the histograms.root file and creates the numpy arrays th13s, dcps,
    bins and counts.

    When saving = True, the th13s and dcps arrays are saved
    When plotting = True, the individual hostograms for each th13 and dcp
    combination is plotted and saved

    INPUT:
    Boolans for saving and plotting
    OUTPUT:
    th13s - numpy array
    dcps - numpy array

    ARGUMENTS:
    bins - list of lists of the bin edges of the 1D histrograms
    counts - list of lists of the counts of the 1D histrograms
    th13s - list of the theta-13 oscillation parameters associated with the histograms
    dcps -  list of the delta-cp oscillation parameters associated with the histograms

    """

    bins = []
    counts = []
    th13s = []
    dcps = []

    histograms = uproot.open(
        'histograms.root')

    for i in histograms.keys():

        th13s.append(float(str(i)[3:-3].split('_')[0]))
        dcps.append(float(str(i)[3:-3].split('_')[1]))
        bins.append(histograms[i].edges)
        counts.append(histograms[i].values)

    if saving:
        np.save(
            f'bins_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy', bins)
        np.save(
            f'counts_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy', counts)

    bins = np.load(
        f'bins_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy')
    counts = np.load(
        f'counts_{min(th13s)}-{max(th13s)}-{len(np.unique(th13s))}_{min(dcps)}-{max(dcps)}-{len(np.unique(dcps))}.npy')

    if plotting:
        for i in range(5):
            hist_label = r'$\theta_{13}$: ' + \
                f' {th13s[i]}\n' + r'$\delta_{cp}$: ' + f'{dcps[i]:.2f}'
            plt.hist(bins[i][:-1], bins[i],
                     weights=counts[i], label=hist_label)
            plt.xlabel('Reconstructed Energy [GeV]', fontsize=15)
            plt.ylabel(r'$\nu_e$ appearance events', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(fontsize=12)
            plt.savefig(f'figures/hist_{th13s[i]}_{dcps[i]}.png')
            plt.show()

    return bins, np.float32(counts), th13s, dcps


def r_loss(y_true, y_pred):
    return tf.reduce_mean(abs(y_true - y_pred))


def define_regression_model1(counts, lr):
    """
    Defines a regression analysis model using the data samples (histograms) to learn
    how their shapes relate to th13 and dcp. Ultimately aims to find the th13 and dcp
    values for the given histogram.

    INPUT:
    counts
    lr - learning rate
    OUTPUT:
    regression model

    """

    regress_model_input = keras.Input(
        shape=(np.shape(counts)[1]))  # counts = (100,40)

    dropout = 0.5

    x = tf.keras.layers.Dense(500)(regress_model_input)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.Reshape((1, 100))(x)
    x = tf.keras.layers.Conv1D(
        filters=80, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(50)(x)
    #x = tf.keras.layers.Dense(20)(x)
    # x = tf.keras.layers.Reshape((1, 30))(x)
    # x = tf.keras.layers.Conv1D(
    #     filters=20, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(4)(x)
    regress_model_output = tf.keras.layers.Dense(
        2)(x)

    regress_model = keras.Model(
        regress_model_input, regress_model_output, name='regress_model')

    optimizer = Adam(lr=lr)

    # regress_model.compile(loss=r_loss,
    #                       optimizer=optimizer)
    regress_model.compile(loss='mean_absolute_error',
                          optimizer=optimizer, metrics=['mean_absolute_error'])
    # regress_model.compile(loss='MeanSquaredError', optimizer=optimizer, metrics=['MeanSquaredError'])

    return regress_model


def define_discriminator1(counts, regress_weights, layers_to_ignore, lr=0.001):
    '''
    Defines the discriminator of the GAN model, whose aim is to differentate
    between real and fake 1D histrograms accurately

    INPUT:
    counts - Real(1) & Fake(0) 1D arrays with a length of 40
    regress_weights - weights of layers of regression models
    layers_to_ignore - number of layers in the discriminator to not change weights
                       usually set to 2 (first and last)
    lr - learning rate

    OUTPUT:
    Yes/no decision from 0 to 1

    '''

    discriminator_model_input = keras.Input(
        shape=(np.shape(counts)[1]))

    dropout = 0.5

    x = tf.keras.layers.Dense(500)(discriminator_model_input)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.Reshape((1, 100))(x)
    x = tf.keras.layers.Conv1D(
        filters=80, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(50)(x)
    # x = tf.keras.layers.Dense(50)(discriminator_model_input)
    # x = tf.keras.layers.Dense(40)(x)
    # x = tf.keras.layers.Dense(30)(x)
    # x = tf.keras.layers.Dense(20)(x)
    # x = tf.keras.layers.Reshape((1, 30))(x)
    # x = tf.keras.layers.Conv1D(
    #     filters=20, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    discriminator_model_output = tf.keras.layers.Dense(
        1, activation='sigmoid')(x)  # yes/no decision layer

    discriminator_model = keras.Model(
        discriminator_model_input, discriminator_model_output, name='discriminator_model')
    print(discriminator_model.layers)
    # Use adam version of stochastic gradient descent with a learning rate of 0.01
    optimizer = Adam(lr=lr)

    discriminator_model.compile(loss='BinaryCrossentropy', optimizer=optimizer, metrics=[
        'BinaryAccuracy'])

    d_layer_num = len(discriminator_model.layers)
    reweight_layers = d_layer_num - layers_to_ignore

    weights_setter(discriminator_model, reweight_layers, regress_weights)

    return discriminator_model


def define_regression_model(counts, lr=0.0001):
    """
    Defines a regression analysis model using the data samples (histograms) to learn
    how their shapes relate to th13 and dcp. Ultimately aims to find the th13 and dcp
    values for the given histogram.

    INPUT:
    counts
    lr - learning rate
    OUTPUT:
    regression model

    """
    # weight initialization
    init = tf.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)

    regress_model_input = keras.Input(
        shape=(np.shape(counts)[1]))  # counts = (100,40)

    dropout = 0.5

    x = tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(
        alpha=0.01))(regress_model_input)
    x = tf.keras.layers.Reshape((1, 20))(x)
    x = tf.keras.layers.Conv1D(
        filters=15, kernel_size=5, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(
        filters=15, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    # x = tf.keras.layers.Dense(
    #     5, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    # x = tf.keras.layers.Dense(
    #     5, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x = tf.keras.layers.Flatten()(x)
    regress_model_output = tf.keras.layers.Dense(
        2)(x)

    regress_model = keras.Model(
        regress_model_input, regress_model_output, name='regress_model')

    # Use adam version of stochastic gradient descent with a learning rate of 0.01
    optimizer = Adam(lr=lr)

    regress_model.compile(loss=r_loss,
                          optimizer=optimizer)
    # regress_model.compile(loss='MeanSquaredError', optimizer=optimizer, metrics=['MeanSquaredError'])

    # model trained to minimise the Mean Squared Error

    return regress_model


def weights_setter(model, reweight_layers_num, new_weights):
    """
    Sets the trained weights from particular layers in the regression model to
    the layers in the discriminator model.

    INPUT:
    model - discriminator model
    reweight_layers_num - int, number of layers to reweight
    new_weights - list of layered weights from regression model

    """
    for i in range(reweight_layers_num):

        # add one to index to skip the first input layer
        model.layers[i + 2].set_weights(new_weights[i + 2])


def define_discriminator(x, regress_weights, layers_to_ignore, lr=0.000001):
    '''
    Defines the discriminator of the GAN model, whose aim is to differentate
    between real and fake 1D histrograms accurately

    INPUT:
    counts - Real(1) & Fake(0) 1D arrays with a length of 40
    regress_weights - weights of layers of regression models
    layers_to_ignore - number of layers in the discriminator to not change weights
                       usually set to 2 (first and last)
    lr - learning rate

    OUTPUT:
    Yes/no decision from 0 to 1

    '''
    # weight initialization
    init = tf.initializers.RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)

    # discriminator_model_input
    discriminator_model_input = keras.Input(
        shape=(np.shape(x)[1]))

    dropout = 0.5

    x = tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(
        alpha=0.01))(discriminator_model_input)
    x = tf.keras.layers.Reshape((1, 20))(x)
    x = tf.keras.layers.Conv1D(
        filters=15, kernel_size=5, strides=1, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(
        filters=15, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    # x = tf.keras.layers.Dense(
    #     5, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    # x = tf.keras.layers.Dense(
    #     5, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x = tf.keras.layers.Flatten()(x)
    discriminator_model_output = tf.keras.layers.Dense(
        1, activation='sigmoid')(x)  # yes/no decision layer

    discriminator_model = keras.Model(
        discriminator_model_input, discriminator_model_output, name='discriminator_model')

    # Use adam version of stochastic gradient descent with a learning rate of 0.01
    optimizer = Adam(lr=lr)

    discriminator_model.compile(loss='BinaryCrossentropy', optimizer=optimizer, metrics=[
        'BinaryAccuracy'])

    # binary_crossentropy is appropriate for binary classification - true or fake data
    # model trained to minimise the binary_crossentropy

    # print('w1_i', discriminator_model.layers[1].get_weights())
    # print('w2_i', discriminator_model.layers[2].get_weights())
    # print('w3_i', discriminator_model.layers[3].get_weights())
    # print('w4_i', discriminator_model.layers[4].get_weights())
    # print('w5_i', discriminator_model.layers[5].get_weights())

    d_layer_num = len(discriminator_model.layers)
    reweight_layers = d_layer_num - layers_to_ignore

    weights_setter(discriminator_model, reweight_layers, regress_weights)

    # print('w1_o', discriminator_model.layers[1].get_weights())
    # print('w2_o', discriminator_model.layers[2].get_weights())
    # print('w3_o', discriminator_model.layers[3].get_weights())
    # print('w4_o', discriminator_model.layers[4].get_weights())
    # print('w5_o', discriminator_model.layers[5].get_weights())

    return discriminator_model


def define_generator(th13s, dcps):
    """
    Defines the generator model of the GAN model, which take the two oscillation
    parameters and learns how to produce realistic event prob histograms.

    INPUT:
    th13s - list of the theta-13 oscillation parameters associated with the histograms
    dcps -  list of the delta-cp oscillation parameters associated with the histograms
    OUTPUT:
    generator model
    """
    params = np.stack((th13s, dcps), axis=-1)

    generator_input = keras.Input(
        shape=(np.shape(params)[1]))  # params = (100,2)

    dropout = 0.5

    x = tf.keras.layers.Dense(10)(generator_input)
    x = tf.keras.layers.Dense(20)(x)
    x = tf.keras.layers.Dense(30)(x)
    # x = tf.keras.layers.Dense(100)(x)
    #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    #x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    #x = tf.keras.layers.Reshape((1, 100))(x)
    # x = tf.keras.layers.Dropout(dropout)(x)  # prevents overfitting
    # x = tf.keras.layers.Conv1DTranspose(filters=40, kernel_size=3, strides=2, padding='same')(
    #     x)
    #x = tf.keras.layers.Dropout(dropout)(x)
    # x = tf.keras.layers.Conv1DTranspose(filters=40, kernel_size=5, strides=1, padding='same')(
    #     x)
    x = tf.keras.layers.Dense(40)(x)
    #x = tf.keras.layers.Flatten()(x)

    generator_output = tf.keras.layers.Dense(40)(x)
    # x = tf.keras.layers.Reshape((40,))(
    #     x)  # generator_output shape = (None, 40)
    # x = tf.keras.layers.Dense(40)(x)
    # generator_output = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    generator_model = keras.Model(
        generator_input, generator_output, name='generator_model')

    # optimizer = Adam(lr=0.01)

    # generator_model.compile(loss='MeanSquaredError',
    # optimizer=optimizer, metrics=['MeanSquaredError'])

    return generator_model


def get_callbacks(file_time, results_loc, tensorboard_profiler=True, es_patience=200, delete_logs=True):
    """
    Defines the callbacks that may be created and then called when fitting models

    When tensorboard_profiler = True, the loss and val_loss are recorded during training
    and can use Tensorboard local host to view the plots
    When delete_logs = True, the /logs file will be deleted

    INPUT:
    es_patience - no of epochs to wait whilst the metric is no longer improving

    OUTPUT:
    tensorboard_callback - records training losses
    lr_reduce - reduced learning rate according to parameters
    earlystop_callback - stops the training according to parameters

    """
    if tensorboard_profiler:

        log_dir = results_loc + "/logs/fit/" + file_time
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, profile_batch='500,520')
        if delete_logs:
            shutil.rmtree('logs/')
    else:
        tensorboard_callback = tf.keras.callbacks.History()

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=50, verbose=1, mode='auto'
    )

    earlystop_callback = keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-3,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=es_patience,
        verbose=1,
    )

    return tensorboard_callback, lr_reduce, earlystop_callback


def preprocess(x, y, buffer_size=128, validation_split=0.2, batch_size=1):
    """
    Preprocesses the data sets x and y into Tensorboard datasets and splits the
    data into a training and validation set

    INPUT:
    x - data set x
    y - data set y
    buffer_size
    validation_split - the split ratio for validation set
    batch_size

    OUTPUT:
    training_set
    validation_set

    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(buffer_size=128)
    list(dataset.as_numpy_iterator())
    validation_set = dataset.take(
        round(int(dataset.cardinality()) * validation_split)).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    training_set = dataset.skip(
        round(int(dataset.cardinality()) * validation_split)).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return training_set, validation_set


def get_regress_model_weights(counts, th13s, dcps, results_loc, figure_loc, regress_checkpoint_loc, epochs=20, batch_size=32, lr=0.000001, plotting=False):
    """
    Creates the regression_model using 'define_regression_model', trains it,
    and then extracts the weights of the chosen layers

    INPUT:
    counts, th13s, dcps
    OUTPUT:
    weights - arrays of regression model weights for each layer

    """
    checkpoint_filepath = regress_checkpoint_loc + '/weights.{epoch}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_d_loss',
        save_freq='epoch',
        save_weights_only=True,
        save_best_only=False)

    regress_model = define_regression_model(counts, lr=lr)

    file_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(
        'regression_callbacks', results_loc, tensorboard_profiler=True, delete_logs=False, es_patience=50)

    x = counts
    y = np.stack((th13s, dcps), axis=-1)
    train, val = preprocess(x, y, batch_size=batch_size)

    regress_model.fit(train, validation_data=val, epochs=epochs,
                      verbose=0, callbacks=[tensorboard_callback, model_checkpoint_callback])

    if plotting:

        filenames = sorted(glob.glob(regress_checkpoint_loc + '/*'),
                           key=lambda x: int(x.split('.')[1]))

        print(filenames)
        print(epochs)

        for epoch in range(epochs):

            if epoch % 100 == 0:

                regress_model.load_weights(filenames[epoch])
                prediction = regress_model.predict(counts)
                plt.plot(th13s, dcps, 'x', color='red',
                         label='Real Params')
                plt.plot(prediction[:, 0], prediction[:, 1], 'x',
                         color='green', label='R Predicted Params')
                plt.xlabel('Th13')
                plt.ylabel('dcp')
                plt.savefig(figure_loc + f'/R_param_space{epoch}.png')
                plt.show()
                plt.close()

    weights = []

    for layer_i in range(len(regress_model.layers)):

        layer_weight = regress_model.layers[layer_i].get_weights()

        weights.append(layer_weight)

    return weights


def get_regress_model_weights1(counts, th13s, dcps, results_loc, figure_loc, regress_checkpoint_loc, epochs=20, batch_size=32, lr=0.000001, plotting=False):
    """
    Creates the regression_model using 'define_regression_model', trains it,
    and then extracts the weights of the chosen layers

    INPUT:
    counts, th13s, dcps
    OUTPUT:
    weights - arrays of regression model weights for each layer

    """
    checkpoint_filepath = regress_checkpoint_loc + '/weights.{epoch}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_d_loss',
        save_freq='epoch',
        save_weights_only=True,
        save_best_only=False)

    regress_model = define_regression_model1(counts, lr=lr)

    file_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(
        file_time, results_loc, tensorboard_profiler=False, delete_logs=False, es_patience=50)

    x = counts
    y = np.stack((th13s, dcps), axis=-1)
    train, val = preprocess(x, y, batch_size=batch_size)

    regress_model.fit(train, validation_data=val, epochs=epochs,
                      verbose=0, callbacks=[model_checkpoint_callback])

    if plotting:

        filenames = sorted(glob.glob(regress_checkpoint_loc + '/*'),
                           key=lambda x: int(x.split('.')[1]))

        print(filenames)
        print(epochs)

        for epoch in range(epochs):

            if epoch % 100 == 0:

                regress_model.load_weights(filenames[epoch])
                prediction = regress_model.predict(counts)
                plt.plot(th13s, dcps, 'x', color='red',
                         label='Real Params')
                plt.plot(prediction[:, 0], prediction[:, 1], 'x',
                         color='green', label='R Predicted Params')
                plt.xlabel('sin^2(2Th13)')
                plt.ylabel('sin(dcp)')
                plt.savefig(figure_loc + f'/R_param_space{epoch}.png')
                plt.show()
                plt.close()

    weights = []

    for layer_i in range(len(regress_model.layers)):

        layer_weight = regress_model.layers[layer_i].get_weights()

        weights.append(layer_weight)

    return weights


def crossentropy_loss(y_true, y_pred):
    return tf.reduce_mean(-y_true * y_pred)


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * -y_pred + (1 - y_true) * y_pred)

# clip model weights to a given hypercube


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# %%


class GAN(keras.Model):
    def __init__(self, discriminator, generator, batch_size, logging=None, train_ratio=5):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.batch_size = batch_size
        self.logging = logging
        self.train_ratio = train_ratio
        self.epochs = 0
        self.hist = []

    def compile(self, d_optimizer, g_optimizer, g_loss_fn, d_loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def set_noisy_label(self, label, p_flip):
        n_select = int(p_flip * label.shape[0])
        flip_ix = np.random.choice(
            [i for i in range(label.shape[0])], size=n_select)
        label[flip_ix] = 1 - label[flip_ix]

        return label

    @tf.function
    def train_step(self, real_images):

        real_images, params = real_images

        real_images = tf.concat(
            [abs(tf.squeeze(tf.random.poisson([1], real_images))), params], axis=1)

        real_images = tf.cast(real_images, dtype=tf.float32)

        #real_images = abs(real_images+np.random.normal([self.batch_size], 0, 10))
        # th13 = np.float32(np.random.uniform(
        #     low=0.056, high=0.137, size=(batch_size,)))
        # dcp = np.float32(np.random.uniform(
        #     low=-np.pi, high=np.pi, size=(batch_size,)))

        # Decode them to fake images
        generated_images = tf.concat([self.generator(params), params], axis=1)
        #print('gen images', generated_images)
        # Combine them with real images
        combined_images = tf.concat(
            [generated_images, real_images], axis=0)

        label_0 = np.zeros((self.batch_size, 1))
        label_1 = np.ones((self.batch_size, 1))

        # Suggestions of only doing positive class label smoothing
        # & keeping the smoothing [0.7,1.0]

        # smooth negative class (class=0) to [0,0.3]
        #label_0 = label_0 + (np.random.random(label_0.shape) * 0.3)
        # smooth positive class (class=1) to [0.7,1.2]
        label_1 = label_1 - 0.3 + (np.random.random(label_1.shape) * 0.5)

        # Randomly flip some labels
        # determine the percentage of labels to flip
        label_0 = self.set_noisy_label(label_0, 0.05)
        label_1 = self.set_noisy_label(label_1, 0.05)

        label_0 = tf.convert_to_tensor(label_0, dtype=np.float32)
        label_1 = tf.convert_to_tensor(label_1, dtype=np.float32)

        labels = tf.concat([label_0, label_1], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = tf.reshape(self.discriminator(
                combined_images, training=True), (self.batch_size * 2, 1))
            d_loss = self.d_loss_fn(labels, predictions)
        d_grads = tape.gradient(
            d_loss, self.discriminator.trainable_weights)

        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights))
        # print(self.discriminator.trainable_weights)

        # print(tf.print(self.discriminator.trainable_weights[0]))
        # print(tf.print(self.discriminator.trainable_weights[1]))
        #weights = self.discriminator.trainable_weights
        #[w.assign(tf.clip_by_value(w, -0.1, 0.1)) for w in weights]

        # print(tf.print(self.discriminator.trainable_weights[0]))

        # Assemble labels that say "all real images"
        misleading_labels = tf.random.uniform((self.batch_size, 1), 0.7, 1.0)

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                tf.concat([self.generator(params), params], axis=1), training=False)
            g_loss = self.g_loss_fn(misleading_labels, predictions)
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)

        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_weights))

        # if self.logging == 'g':
        #     print(g_grads)
        #     print('Ouput G Layer of shape (40,):', tf.print(g_grads[-1]))
        #     print('Input Layer of shape (2,5):', tf.print(g_grads[0]))
        # if self.logging == 'd':
        #     print(d_grads)
        #     print('Ouput D Layer of shape (1,):', tf.print(d_grads[-1]))
        #     print('Input Layer of shape (40,):', tf.print(d_grads[0]))
        #
        # if d_loss < g_loss * self.train_ratio:
        #     self.g_optimizer.apply_gradients(
        #         zip(g_grads, self.generator.trainable_weights))
        # else:
        #     self.d_optimizer.apply_gradients(
        #         zip(d_grads, self.discriminator.trainable_weights)
        #     )

        return {"g_loss": g_loss, "d_loss": d_loss}

    @tf.function
    def test_step(self, val_data):
        x, y = val_data
        # Compute predictions
        y_real = self.discriminator(tf.concat([x, y], axis=1), training=False)
        y_fake = self.discriminator(
            tf.concat([self.generator(y), y], axis=1), training=False)
        combined = tf.concat([y_fake, y_real], axis=0)

        label_0 = np.zeros((self.batch_size, 1))
        label_1 = np.ones((self.batch_size, 1))

        # smooth negative class (class=0) to [0,0.3]
        label_0 = label_0 + (np.random.random(label_0.shape) * 0.3)
        # smooth positive class (class=1) to [0.7,1.2]
        label_1 = label_1 - 0.3 + (np.random.random(label_1.shape) * 0.5)

        label_0 = tf.convert_to_tensor(label_0, dtype=np.float32)
        label_1 = tf.convert_to_tensor(label_1, dtype=np.float32)

        labels = tf.concat([label_0, label_1], axis=0)

        # Updates the metrics tracking the loss
        d_loss = self.d_loss_fn(labels, combined)

        return {"d_loss": d_loss}

    @tf.function
    def call(self, input):
        return self.generator(input)
