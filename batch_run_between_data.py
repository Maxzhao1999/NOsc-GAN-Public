####### IMPORT REQUIRED PACKAGES ######

from modules import *
from Data_merger import *
from Analysis import *

file_time = '20210117-160650'

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

figure_loc = f"results/{file_time}/figures/test_on_between_dataset"
os.mkdir(figure_loc)

bins_o = np.load('bins_0.0874-0.099969-320_-1.0-1.0-321.npy')
counts_o = np.load('counts_0.0874-0.099969-320_-1.0-1.0-321.npy')
th13s_o = np.load('th13s_sindcp_normed.npy')
th13s_o = th13s_o.tolist()
dcps_o = np.load('dcps_sindcp_normed.npy')
dcps_o = dcps_o.tolist()

bins = np.load('bins_0.08742-0.099988-320_-0.996875-0.996875-320.npy')
counts = np.load('counts_0.08742-0.099988-320_-0.996875-0.996875-320.npy')
th13s = np.load('th13s_between_normed.npy')
th13s = th13s.tolist()
dcps = np.load('dcps_between_normed.npy')
dcps = dcps.tolist()


# %%
####### Define D, G and GAN, train the GAN, make a prediction with trained generator #######

batch_size = 256

x_0 = np.concatenate((counts_o, np.reshape(
    th13s_o, (len(th13s_o), 1)), np.reshape(dcps_o, (len(dcps_o), 1))), axis=1)

R_filename = 'results/20210117-160650/regress_checkpoint/weights.10000.hdf5'

regress_model = define_regression_model1(counts_o, lr=0.000001)

regress_model.load_weights(R_filename)

prediction = regress_model.predict(counts_o)
plt.plot(th13s_o, dcps_o, 'x', color='red',
         label='Real Params')
plt.plot(prediction[:, 0], prediction[:, 1], 'x',
         color='green', label='R Predicted Params')
plt.xlabel('sin^2(2Th13)')
plt.ylabel('sin(dcp)')
plt.savefig(figure_loc + '/R_param_space10000.png')
plt.show()
plt.close()

regress_weights = []

for layer_i in range(len(regress_model.layers)):

    layer_weight = regress_model.layers[layer_i].get_weights()

    regress_weights.append(layer_weight)

discriminator = define_discriminator1(x_0, regress_weights, 3, lr=0.01)
generator = define_generator(th13s_o, dcps_o)

lr_d = 0.0001
lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    lr_d, decay_steps=1000, decay_rate=0.96, staircase=True  # changed on B
)  # is this actually being applied?

lr_g = 0.00001
lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(
    lr_g, decay_steps=1000, decay_rate=0.96, staircase=True  # changed on B
)

gan = GAN(discriminator=discriminator,
          generator=generator, batch_size=batch_size, logging=None, train_ratio=0.1)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_d),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_g),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
)

x = tf.cast(counts_o.copy(), tf.float32)

y = tf.cast(np.stack((th13s_o, dcps_o), axis=1), tf.float32)

# gan(np.stack((th13s, dcps), axis=1))
# gan.load_weights(filename)
#
# train_data, val_data = preprocess(
#     x, y, batch_size=batch_size, validation_split=0.1)
#
# print('GAN Training:')
#
# gan.fit(train_data, validation_data=val_data, epochs=epoch_num, callbacks=[
#         tensorboard_callback, model_checkpoint_callback], verbose=0)

# %%

checkpoint_loc = f'results/{file_time}/checkpoint'

filenames = sorted(glob.glob((checkpoint_loc + '/*')),
                   key=lambda x: int(x.split('.')[1]))

gan(np.stack((th13s_o, dcps_o), axis=1))

gan.load_weights(filenames[-1])
test = generator.predict(np.stack((th13s, dcps), axis=-1))
Full_Poisson_Error, pred_error, error_diff = PoissonError(counts, test)
num = 1000

list = np.arange(0, len(counts), 100)

for index in list:

    pred = test[index]

    plot_G_outputs(pred, index, error_diff, bins, counts,
                   th13s, dcps, Full_Poisson_Error, num, figure_loc)
