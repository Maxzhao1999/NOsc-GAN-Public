from modules import *
from Data_merger import *
from Analysis import *
# %%
file_time = '20210105-111448'

checkpoint_loc = f'results/{file_time}/checkpoint'
#checkpoint_loc = 'checkpoint'

filenames = sorted(glob.glob((checkpoint_loc + '/*')),
                   key=lambda x: int(x.split('.')[1]))

bins = np.load('bins_sindcp.npy')
counts = np.load('counts_sindcp.npy')
th13s = np.load('th13s_102400_sindcp_normed.npy')
th13s = th13s.tolist()
dcps = np.load('dcps_102400_sindcp_normed.npy')
dcps = dcps.tolist()

batch_size = 256

x_0 = np.concatenate((counts, np.reshape(
    th13s, (len(th13s), 1)), np.reshape(dcps, (len(dcps), 1))), axis=1)


R_filename = 'results/20210105-111448/regress_checkpoint/weights.10000.hdf5'
#R_filename = 'regress_checkpoint/weights.1.hdf5'

regress_model = define_regression_model1(counts, lr=0.000001)

regress_model.load_weights(R_filename)

regress_weights = []

for layer_i in range(len(regress_model.layers)):

    layer_weight = regress_model.layers[layer_i].get_weights()

    regress_weights.append(layer_weight)


discriminator = define_discriminator1(x_0, regress_weights, 3, lr=0.01)
generator = define_generator(th13s, dcps)


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


gan(np.stack((th13s, dcps), axis=1))

vals = np.arange(9, len(filenames), 10)
list = vals.tolist()
list.insert(0, 0)

y = tf.cast(np.stack((th13s, dcps), axis=1), tf.float32)

all_th13s_index = np.arange(
    0, 102400, 321, dtype=np.int)  # with dcps all = -1.0
all_dcps_index = np.arange(0, 321, 1, dtype=np.int)  # with th13s all = -1.0

# all_th13s_index = np.arange(
#     0, 1024, 32, dtype=np.int)  # with dcps all = -1.0
# all_dcps_index = np.arange(0, 32, 1, dtype=np.int)  # with th13s all = -1.0

error_fig_loc = f"results/{file_time}/figures"
#error_fig_loc = "figures"

full_multi_pred_error_list_all_th13s = []
full_all_th13s = []

for index in all_th13s_index:

    multi_pred_error_list = []

    for epoch in list:

        gan.load_weights(filenames[epoch])
        test = generator.predict(np.stack((th13s, dcps), axis=1))

        Full_Poisson_Error, pred_error, error_diff = PoissonError(
            counts, test)

        multi_pred_error_list.append(pred_error[index].mean())

    full_all_th13s.append(th13s[index])

    full_multi_pred_error_list_all_th13s.append(multi_pred_error_list)


multi_plot_errorgraph_all_th13s(
    list, full_multi_pred_error_list_all_th13s, error_fig_loc, full_all_th13s)


full_multi_pred_error_list_all_dcps = []
full_all_dcps = []

for index in all_th13s_index:

    multi_pred_error_list = []

    for epoch in list:

        gan.load_weights(filenames[epoch])
        test = generator.predict(np.stack((th13s, dcps), axis=1))

        Full_Poisson_Error, pred_error, error_diff = PoissonError(
            counts, test)

        multi_pred_error_list.append(pred_error[index].mean())

    full_all_dcps.append(dcps[index])

    full_multi_pred_error_list_all_dcps.append(multi_pred_error_list)


multi_plot_errorgraph_all_dcps(
    list, full_multi_pred_error_list_all_dcps, error_fig_loc, full_all_dcps)
