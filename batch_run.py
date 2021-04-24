####### IMPORT REQUIRED PACKAGES ######

from modules import *
from Data_merger import *
from Analysis import *
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # what does this do?
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# %%
# tf.debugging.set_log_device_placement(True)
####### Get the required dataset #######
t1 = time.time()
epoch_num = int(sys.argv[1])
# epoch_num = 2
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

file_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

results_loc = "results/" + file_time

os.mkdir(results_loc)

figure_loc = results_loc + "/figures"
checkpoint_loc = results_loc + "/checkpoint"
regress_checkpoint_loc = results_loc + "/regress_checkpoint"

os.mkdir(figure_loc)
os.mkdir(checkpoint_loc)
os.mkdir(regress_checkpoint_loc)

bins = np.load('bins_0.0874-0.099969-320_-1.0-1.0-321.npy')
counts = np.load('counts_0.0874-0.099969-320_-1.0-1.0-321.npy')
th13s = np.load('th13s_sindcp_normed.npy')
th13s = th13s.tolist()
dcps = np.load('dcps_sindcp_normed.npy')
dcps = dcps.tolist()

# scaler = max(map(max, counts))
# counts = counts / scaler

t2 = time.time()
# %%
####### Define D, G and GAN, train the GAN, make a prediction with trained generator #######

# Define checkpoints
checkpoint_filepath = checkpoint_loc + '/weights.{epoch}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_d_loss',
    save_freq='epoch',
    save_weights_only=True,
    save_best_only=False)

batch_size = 256

ratio = np.random.rand(counts.shape[0]) * 1.5 + 0.3

neg_counts = counts.copy()

for i in range(len(counts)):

    neg_index = np.random.randint(0, counts.shape[1], np.random.randint(1, 10))
    neg_counts[i][neg_index] = -np.random.random(1) * 20


x_0 = np.concatenate((np.random.poisson(counts, counts.shape), np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1))), axis=1)
x_1 = np.concatenate((abs(np.random.normal(counts.mean(), counts.std(
), counts.shape)), np.random.random((counts.shape[0], 2)) * 2 - 1), axis=1)
x_2 = np.concatenate((np.random.poisson(counts, counts.shape) * ratio[:, np.newaxis], np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1)) * 2 - 1), axis=1)
x_3 = np.concatenate((neg_counts, np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1))), axis=1)
x_4 = np.concatenate((counts, np.random.random((counts.shape[0], 2))), axis=1)

# x_d = np.concatenate((x_0, x_1, x_2, x_3, x_4), axis=0)
#
# y_d = np.concatenate((np.ones(counts.shape[0]), np.zeros(
#     counts.shape[0]), np.ones(counts.shape[0]) * (1 - abs(1 - ratio)), np.zeros(
#         counts.shape[0]), np.zeros(counts.shape[0])))

x_d = np.concatenate((x_0, x_1, x_2), axis=0)
#x_d = np.concatenate((x_0, x_1), axis=0)

y_d = np.concatenate((np.ones(counts.shape[0]), np.zeros(
    counts.shape[0]), np.ones(counts.shape[0]) * (1 - abs(1 - ratio))))

# y_d = np.concatenate((np.ones(counts.shape[0]), np.zeros(
#     counts.shape[0])))

train_d, val_d = preprocess(x_d, y_d, batch_size=batch_size)

# %%

# R_filename = f'results/20210210-104040/regress_checkpoint/weights.10000.hdf5'
#
# regress_model = define_regression_model1(counts, lr=0.000001)
#
# regress_model.load_weights(R_filename)
#
# prediction = regress_model.predict(counts)
# plt.plot(th13s, dcps, 'x', color='red',
#          label='Real Params')
# plt.plot(prediction[:, 0], prediction[:, 1], 'x',
#          color='green', label='R Predicted Params')
# plt.xlabel('sin^2(2Th13)')
# plt.ylabel('sin(dcp)')
# plt.savefig(figure_loc + '/R_param_space10000.png')
# plt.show()
# plt.close()
#
# regress_weights = []
#
# for layer_i in range(len(regress_model.layers)):
#
#     layer_weight = regress_model.layers[layer_i].get_weights()
#
#     regress_weights.append(layer_weight)

regress_weights = get_regress_model_weights1(
    counts, th13s, dcps, results_loc, figure_loc, regress_checkpoint_loc, epochs=10000, batch_size=batch_size, lr=0.000001, plotting=True)

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


generator.compile(loss=generator_loss,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001))

print('Generator Pre-Training:')

x = counts.copy()

y = np.stack((th13s, dcps), axis=1)

for i in range(500):

    x_g, y_g = gen_samples_for_G_training(x, y, 1000)

    generator.train_on_batch(y_g, x_g)

output = generator.predict(np.stack((th13s, dcps), axis=1))[
    np.random.randint(0, len(th13s))]

plt.hist(bins[0][:-1], bins[0], weights=output)
plt.savefig(figure_loc + '/pre-trained_G.png')
plt.close()


print('Discriminator Pre-Training:')


tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(file_time, results_loc,
                                                                    tensorboard_profiler=True, delete_logs=False)

discriminator.fit(train_d, validation_data=val_d, epochs=1000,
                  callbacks=[tensorboard_callback, lr_reduce, earlystop_callback], verbose=2)

print('max discriminator prediction on fake image: ',
      max(discriminator.predict(x_1)))
print('min discriminator prediction on real image: ',
      min(discriminator.predict(x_0)))


tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks('GAN_tensorboardcallback', results_loc,
                                                                    tensorboard_profiler=True, delete_logs=False)

lr_d = 0.0001
lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    lr_d, decay_steps=10000, decay_rate=0.96, staircase=True
)  # is this actually being applied?

lr_g = 0.00001
lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(
    lr_g, decay_steps=10000, decay_rate=0.96, staircase=True
)


gan = GAN(discriminator=discriminator,
          generator=generator, batch_size=batch_size, logging=None, train_ratio=0.1)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_d),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_g),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
)

t3 = time.time()

x = tf.cast(counts.copy(), tf.float32)

y = tf.cast(np.stack((th13s, dcps), axis=1), tf.float32)

train_data, val_data = preprocess(
    x, y, batch_size=batch_size, validation_split=0.1)

print('GAN Training:')

gan.fit(train_data, validation_data=val_data, epochs=epoch_num,
        callbacks=[tensorboard_callback, model_checkpoint_callback], verbose=0)

t4 = time.time()

# %%
filenames = sorted(glob.glob((checkpoint_loc + '/*')),
                   key=lambda x: int(x.split('.')[1]))

gan(np.stack((th13s, dcps), axis=1))

# vals = np.arange(9, len(filenames), 10)
# list = vals.tolist()
# list.insert(0, 0)

vals = np.arange(99, len(filenames), 100)
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

t5 = time.time()

print(generator.summary())
print(discriminator.summary())
print('loading numpy arrays: ', t2 - t1, ' seconds')
print('pretraining: ', t3 - t2, ' seconds')
print('GAN training: ', t4 - t3, ' seconds')
print('plotting: ', t5 - t4, ' seconds')
print('total time taken: ', t5 - t1, ' seconds')
