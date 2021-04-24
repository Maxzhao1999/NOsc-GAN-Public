####### IMPORT REQUIRED PACKAGES ######

from modules import *
from Data_merger import *
from Analysis import *

filename = 'results/20201227-102934/checkpoint/weights.1000.hdf5'
adder = 1000

epoch_num = int(sys.argv[1])

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

bins = np.load('bins_0.0874-0.099969-320_-3.141593-3.121958-320.npy')
counts = np.load('counts_0.0874-0.099969-320_-3.141593-3.121958-320.npy')
th13s = np.load('th13s_102400_normed.npy')
th13s = th13s.tolist()
dcps = np.load('dcps_102400_normed.npy')
dcps = dcps.tolist()


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

x_0 = np.concatenate((counts, np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1))), axis=1)

regress_weights = np.load('regress_weights_dummy.npy', allow_pickle=True)

discriminator = define_discriminator1(x_0, regress_weights, 3, lr=0.01)
generator = define_generator(th13s, dcps)


tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(file_time, results_loc,
                                                                    tensorboard_profiler=True, delete_logs=False)

lr_d = 0.0001
lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    lr_d, decay_steps=1000, decay_rate=0.96, staircase=True
)  # is this actually being applied?

lr_g = 0.00001
lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(
    lr_g, decay_steps=1000, decay_rate=0.96, staircase=True
)

gan = GAN(discriminator=discriminator,
          generator=generator, batch_size=batch_size, logging=None, train_ratio=0.1)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_d),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_g),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
)

x = tf.cast(counts.copy(), tf.float32)

y = tf.cast(np.stack((th13s, dcps), axis=1), tf.float32)

gan(np.stack((th13s, dcps), axis=1))
gan.load_weights(filename)

train_data, val_data = preprocess(
    x, y, batch_size=batch_size, validation_split=0.1)

print('GAN Training:')

gan.fit(train_data, validation_data=val_data, epochs=epoch_num, callbacks=[
        tensorboard_callback, model_checkpoint_callback], verbose=0)

# %%
filenames = sorted(glob.glob((checkpoint_loc + '/*')),
                   key=lambda x: int(x.split('.')[1]))

gan(np.stack((th13s, dcps), axis=1))

vals = np.arange(9, len(filenames), 10)
list = vals.tolist()
list.insert(0, 0)

D_real_pred = []
D_fake_pred = []

num_list = []

for epoch in list:

    gan.load_weights(filenames[epoch])
    test = generator.predict(np.stack((th13s, dcps), axis=-1))
    Full_Poisson_Error, pred_error, error_diff = PoissonError(counts, test)
    num = int(filenames[epoch].split('.')[1]) + adder
    num_list.append(num)

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

plot_D_predictions(num_list, D_real_pred, D_fake_pred, figure_loc)
