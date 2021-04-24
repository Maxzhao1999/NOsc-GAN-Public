#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:01:27 2020

@author: luanawilliams
"""

####### IMPORT REQUIRED PACKAGES ######

from modules import *
from Data_merger import *
from Analysis import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # what does this do?
%load_ext tensorboard

# %%
####### Get the required dataset #######

# print("Num GPUs Available: ", len(
#     tf.config.experimental.list_physical_devices('GPU')))
#
# bins, counts, th13s, dcps = merge_dataset(root_name=[
#                                           'histograms64000of102400', 'histograms38400of102400'], saving=True, plotting=False)
#
# norm_th13s = MinMaxScaler(feature_range=(-1, 1))
# norm_dcps = MinMaxScaler(feature_range=(-1, 1))
# #
# # normalising the dcps [0,1] to improve loss calculations and training
# th13s = np.array(th13s)
# th13s = th13s.reshape(len(th13s), 1)
# norm_th13s.fit(th13s)
# th13s = norm_th13s.transform(th13s)
# th13s = th13s.reshape(th13s.shape[0])
# th13s = th13s.tolist()
# #
# dcps = np.array(dcps)
# dcps = dcps.reshape(len(dcps), 1)
# norm_dcps.fit(dcps)
# dcps = norm_dcps.transform(dcps)
# dcps = dcps.reshape(dcps.shape[0])
# dcps = dcps.tolist()
#
# np.save('th13s_102400_normed', th13s)
# np.save('dcps_102400_normed', dcps)

# %%
##### Load the data set for 102400 samples (already normalised) ####
#
# bins = np.load('bins_0.0874-0.099969-320_-3.141593-3.121958-320.npy')
# bins = bins.tolist()
# counts = np.load('counts_0.0874-0.099969-320_-3.141593-3.121958-320.npy')
# counts = counts.tolist()
# th13s = np.load('th13s_102400_normed.npy')
# th13s = th13s.tolist()
# dcps = np.load('dcps_102400_normed.npy')
# dcps = dcps.tolist()

bins, counts, th13s, dcps = get_dataset(saving=True, plotting=False)
dcps = np.sin(dcps)

norm_th13s = MinMaxScaler(feature_range=(-1, 1))
norm_dcps = MinMaxScaler(feature_range=(-1, 1))
#
# normalising the dcps [0,1] to improve loss calculations and training
th13s = np.array(th13s)
th13s = th13s.reshape(len(th13s), 1)
norm_th13s.fit(th13s)
th13s = norm_th13s.transform(th13s)
th13s = th13s.reshape(th13s.shape[0])
th13s = th13s.tolist()
#
dcps = np.array(dcps)
dcps = dcps.reshape(len(dcps), 1)
norm_dcps.fit(dcps)
dcps = norm_dcps.transform(dcps)
dcps = dcps.reshape(dcps.shape[0])
dcps = dcps.tolist()

# %%
####### Define generator and prediction example w/o training  #######

generator_model = define_generator(th13s, dcps)
output = generator_model.predict(
    np.stack((th13s, dcps), axis=-1))[np.random.randint(0, len(th13s))]
plt.hist(bins[0][:-1], bins[0], weights=output)
plt.show()
'''
#test discriminator on a random histogram
t = output.reshape(1,40)
discrim_model.predict(t)
'''
# %%
####### Define D, G and GAN, train the GAN, make a prediction with trained generator #######

# Define checkpoints
shutil.rmtree('checkpoint/')
os.makedirs('checkpoint')
checkpoint_filepath = './checkpoint/weights.{epoch}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_d_loss',
    save_freq='epoch',
    save_weights_only=True,
    save_best_only=False)

batch_size = 32

# tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(
#     tensorboard_profiler=True, delete_logs=False)


# need to reduce this upper boundary
ratio = np.random.rand(counts.shape[0]) * 1.5 + 0.3

neg_counts = counts.copy()

for i in range(len(counts)):

    neg_index = np.random.randint(0, counts.shape[1], np.random.randint(1, 10))
    neg_counts[i][neg_index] = -np.random.random(1) * 20

# y_d = np.concatenate(
#     (np.ones(counts.shape[0]), np.zeros(counts.shape[0]), ratio * ratio, np.zeros(counts.shape[0]), np.around(1 / big_ratio, 1) * np.around(1 / big_ratio, 1)))

#y_d = np.concatenate((np.ones(counts.shape[0]), np.zeros(counts.shape[0]), ratio, np.zeros(counts.shape[0]), np.around(1 / big_ratio, 1)))

# x = np.concatenate((counts, np.random.rand(counts.shape)), axis=0)
x_0 = np.concatenate((counts, np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1))), axis=1)
x_1 = np.concatenate((abs(np.random.normal(counts.mean(), counts.std(
), counts.shape)), np.random.random((counts.shape[0], 2)) * 2 - 1), axis=1)
x_2 = np.concatenate((counts * ratio[:, np.newaxis], np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1)) * 2 - 1), axis=1)
x_3 = np.concatenate((neg_counts, np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1))), axis=1)
x_4 = np.concatenate((counts, np.random.random((counts.shape[0], 2))), axis=1)

x_d = np.concatenate((x_0, x_1, x_2, x_3, x_4), axis=0)

y_d = np.concatenate((np.ones(counts.shape[0]), np.zeros(
    counts.shape[0]), np.ones(counts.shape[0]) * (1 - abs(1 - ratio)), np.zeros(
        counts.shape[0]), np.zeros(counts.shape[0])))

# x_d = np.concatenate((abs(counts + np.random.normal(0, counts.std() * 0.2, counts.shape)),
#                       abs(np.random.normal(counts.mean(), counts.std(), counts.shape)), counts * ratio[:, np.newaxis], neg_counts, counts * big_ratio[:, np.newaxis]), axis=0)
#
train_d, val_d = preprocess(x_d, y_d, batch_size=batch_size)

shutil.rmtree('figures/')
os.makedirs('figures')

results_loc = 'figures'
figure_loc = 'figures'
regress_checkpoint_loc = 'regress_checkpoint'


regress_weights = get_regress_model_weights1(
    counts, th13s, dcps, results_loc, figure_loc, regress_checkpoint_loc, epochs=1000, batch_size=batch_size, lr=0.000001, plotting=True)

# %%

discriminator = define_discriminator1(x_0, regress_weights, 3, lr=0.001)
generator = define_generator(th13s, dcps)


def generator_loss(y_true, y_pred):
    return tf.reduce_mean(abs(y_true - y_pred))


def gen_samples_for_G_training(x, y, n_samples):

    # chose n_samples random images from the dataset
    ix = np.random.randint(0, x.shape[0], n_samples)

    x_g = x[ix]

    y_g = y[ix]

    return x_g, y_g


print('Discriminator Pre-Training:')

file_time = '50'
tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(file_time, results_loc,
                                                                    tensorboard_profiler=False, delete_logs=False)

discriminator.fit(train_d, validation_data=val_d, epochs=300,
                  verbose=2, callbacks=[lr_reduce, earlystop_callback])

print('discriminator prediction on fake image: ',
      max(discriminator.predict(x_1)))
print('discriminator prediction on real image: ',
      min(discriminator.predict(x_0)))

# %%
# tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(file_time,
#                                                                     tensorboard_profiler=True, delete_logs=False)

shutil.rmtree('checkpoint/')
os.makedirs('checkpoint')
checkpoint_filepath = './checkpoint/weights.{epoch}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_d_loss',
    save_freq='epoch',
    save_weights_only=True,
    save_best_only=False)

lr_d = 0.0001
lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    lr_d, decay_steps=10000, decay_rate=0.96, staircase=True
)

lr_g = 0.00001
lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(
    lr_g, decay_steps=10000, decay_rate=0.96, staircase=True
)

x = tf.cast(counts.copy(), tf.float32)

y = tf.cast(np.stack((th13s, dcps), axis=1), tf.float32)


train, val = preprocess(y, x, batch_size=32, validation_split=0.1)

generator.compile(loss=generator_loss,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001))

print('Generator Pre-Training:')

generator.fit(train, validation_data=val, epochs=20)

output = generator.predict(np.stack((th13s, dcps), axis=1))[
    np.random.randint(0, len(th13s))]

plt.hist(bins[0][:-1], bins[0], weights=output)
plt.show()

gan = GAN(discriminator=discriminator,
          generator=generator, batch_size=batch_size, logging=None, train_ratio=0.1)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_d),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_g),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
)
train_data, val_data = preprocess(
    x, y, batch_size=batch_size, validation_split=0.1)

t3 = time.time()

print('GAN Training:')

gan.fit(train_data, validation_data=val_data, epochs=1000,
        callbacks=[model_checkpoint_callback], verbose=0)

# %%

shutil.rmtree('figures/')
os.makedirs('figures')

filenames = sorted(glob.glob('checkpoint/*'),
                   key=lambda x: int(x.split('.')[1]))

dcp_fixed_list = np.arange(16, 1024, 32, dtype=np.int)

gan(np.stack((th13s, dcps), axis=1))

vals = np.arange(9, len(filenames), 10)
list = vals.tolist()
list.insert(0, 0)

D_real_pred = []
D_fake_pred = []

for epoch in list:

    gan.load_weights(filenames[epoch])
    test = generator.predict(np.stack((th13s, dcps), axis=-1))
    Full_Poisson_Error, pred_error, error_diff = PoissonError(counts, test)
    num = int(filenames[epoch].split('.')[1])

    DFP = discriminator.predict(tf.reshape(
        tf.concat([test, np.stack((th13s, dcps), axis=-1)], axis=-1), (len(test), 42)))
    DRP = discriminator.predict(tf.reshape(x_0, (len(x_0), 42)))

    D_fake_pred.append(DFP.mean())
    D_real_pred.append(DRP.mean())

    for index in sorted(random.sample(range(len(counts)), 2)):

        pred = test[index]

        plot_G_outputs(pred, index, error_diff, bins, counts,
                       th13s, dcps, Full_Poisson_Error, num, figure_loc)

        DFP = discriminator.predict(tf.reshape(
            tf.concat([pred, y[index]], axis=0), (1, 42)))
        DRP = discriminator.predict(tf.reshape(x_0[index], (1, 42)))

        print('discriminator prediction on fake image: ', DFP)
        print('discriminator prediction on real image: ', DRP)

plot_D_predictions(list, D_real_pred, D_fake_pred, figure_loc)
