from modules import *
from Data_merger import *
from Analysis import *
# %%
# file_time = '20210117-160650'
file_time = '20210208-164718'

checkpoint_loc = f'results/{file_time}/checkpoint'

filenames = sorted(glob.glob((checkpoint_loc + '/*')),
                   key=lambda x: int(x.split('.')[1]))

bins_o = np.load('bins_0.0874-0.099969-320_-1.0-1.0-321.npy')
counts_o = np.load('counts_0.0874-0.099969-320_-1.0-1.0-321.npy')
th13s_o = np.load('th13s_sindcp_normed.npy')
th13s_o = th13s_o.tolist()
dcps_o = np.load('dcps_sindcp_normed.npy')
dcps_o = dcps_o.tolist()

bins = np.load('bins_0.08742-0.099988-320_-0.996875-0.996875-320.npy')
counts = np.load('counts_0.08742-0.099988-320_-0.996875-0.996875-320.npy')
th13s = np.load('th13s_between_normed.npy')
len(th13s)
th13s = th13s.tolist()
dcps = np.load('dcps_between_normed.npy')
len(dcps)
dcps = dcps.tolist()

batch_size = 256

x_0 = np.concatenate((counts_o, np.reshape(
    th13s_o, (len(th13s_o), 1)), np.reshape(dcps_o, (len(dcps_o), 1))), axis=1)

R_filename = 'results/20210202-134702/regress_checkpoint/weights.10000.hdf5'
#R_filename = 'results/20210117-160650/regress_checkpoint/weights.10000.hdf5'
#R_filename = 'regress_checkpoint/weights.1.hdf5'

regress_model = define_regression_model1(counts_o, lr=0.000001)

regress_model.load_weights(R_filename)

regress_weights = []

for layer_i in range(len(regress_model.layers)):

    layer_weight = regress_model.layers[layer_i].get_weights()

    regress_weights.append(layer_weight)


discriminator = define_discriminator1(x_0, regress_weights, 3, lr=0.01)
generator = define_generator(th13s_o, dcps_o)


lr_d = 0.0001
lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    lr_d, decay_steps=10000, decay_rate=0.96, staircase=True
)

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


gan(np.stack((th13s_o, dcps_o), axis=1))

vals = np.arange(9, len(filenames), 10)
list = vals.tolist()
list.insert(0, 0)

y = tf.cast(np.stack((th13s_o, dcps_o), axis=1), tf.float32)

# %%

x_axis = np.linspace(-1, 1, 320)

y_axis = np.linspace(-1, 1, 320)

X, Y = np.meshgrid(x_axis, y_axis)

figure_loc = f"results/{file_time}/figures/error_spectrograms_between"
os.mkdir(figure_loc)
#figure_loc = 'figures'


# for epoch in list:

gan.load_weights(filenames[-1])
test = generator.predict(np.stack((th13s, dcps), axis=1))

#num = int(filenames[epoch].split('.')[1])
num = 1000

Full_Poisson_Error, pred_error, error_diff = PoissonError(
    counts, test)

pred_error_means = []

for i in range(len(pred_error)):

    pred_error_means.append(sum(pred_error[i]))

pred_error_means_split = []

for i in range(0, len(pred_error_means), 320):

    chunk = pred_error_means[i:i + 320]

    pred_error_means_split.append(chunk)

im = np.array(pred_error_means_split)

error_spectrogram_plotter(X, Y, im, figure_loc, num)
