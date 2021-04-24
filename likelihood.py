####### IMPORT REQUIRED PACKAGES ######

from modules import *
from Data_merger import *
from Analysis import *
import scipy as sp
import pandas as pd
import logging
tf.get_logger().setLevel(logging.ERROR)

#file_time = '20210118-2234'
# file_time = '20210119-091527'

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

#figure_loc = f"results/{file_time}/figures/test_on_between_dataset"
figure_loc = 'likelihood_figures'
#filename = f"results/{file_time}/checkpoint/weights.1000.hdf5"
#filename = 'downloaded_data_for_test_running/weights_20210208-164718.10000.hdf5'
filename = 'downloaded_data_for_test_running/20210218-weights.10000.hdf5'

if not os.path.exists(figure_loc):
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

# bins = np.load('bins_0.075976-0.105991-320_-0.996875-0.996875-320.npy')
# counts = np.load('counts_0.075976-0.105991-320_-0.996875-0.996875-320.npy')
# th13s = np.load('th13s_larger_normed.npy')
# th13s = th13s.tolist()
# dcps = np.load('dcps_larger_normed.npy')
# dcps = dcps.tolist()


# %%
####### Define D, G and GAN, train the GAN, make a prediction with trained generator #######

batch_size = 256

x_0 = np.concatenate((counts_o, np.reshape(
    th13s_o, (len(th13s_o), 1)), np.reshape(dcps_o, (len(dcps_o), 1))), axis=1)

# R_filename = f'results/{file_time}/regress_checkpoint/weights.10000.hdf5'
# R_filename = f'results/20210117-160650/regress_checkpoint/weights.10000.hdf5'
# R_filename = 'downloaded_data_for_test_running/R_weights_20210202-134702.10000.hdf5'
R_filename = 'downloaded_data_for_test_running/20210218-Rweights.10000.hdf5'

regress_model = define_regression_model1(counts_o, lr=0.000001)

regress_model.load_weights(R_filename)

prediction = regress_model.predict(counts_o)
diff_th13 = th13s_o - prediction[:, 0]
diff_dcp = dcps_o - prediction[:, 1]

plt.plot(th13s_o, dcps_o, 'x', color='red',
         label='Real Params')
plt.plot(prediction[:, 0], prediction[:, 1], 'x',
         color='green', label='R Predicted Params')
plt.xlabel('sin^2(2Th13)')
plt.ylabel('sin(dcp)')
plt.savefig(figure_loc + '/R_param_space10000.png')
plt.show()
plt.close()

z = np.stack((th13s_o, dcps_o), axis=1)

diff = (z[:, 0] - prediction[:, 0])**2 + (z[:, 1] - prediction[:, 1])**2


df = pd.DataFrame.from_dict(np.array([dcps_o, th13s_o, diff]).T)
df.columns = ['sin(dcp)', 'sin^2(2th13)', '-2lnL']
df['-2lnL'] = pd.to_numeric(df['-2lnL'])
pivotted = df.pivot('sin^2(2th13)', 'sin(dcp)', '-2lnL')

fig = plt.scatter(df['sin(dcp)'], df['sin^2(2th13)'], c=df['-2lnL'],
                  cmap='inferno').get_figure()
plt.colorbar()

regress_weights = []

for layer_i in range(len(regress_model.layers)):

    layer_weight = regress_model.layers[layer_i].get_weights()

    regress_weights.append(layer_weight)

discriminator = define_discriminator1(x_0, regress_weights, 3, lr=0.01)
generator = define_generator(th13s_o, dcps_o)
generator.summary()


lr_d = 0.0001
lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
    lr_d, decay_steps=10000, decay_rate=0.96, staircase=True  # changed on B
)  # is this actually being applied?

lr_g = 0.00001
lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(
    lr_g, decay_steps=10000, decay_rate=0.96, staircase=True  # changed on B
)

# %%
gan = GAN(discriminator=discriminator,
          generator=generator, batch_size=batch_size, logging=None, train_ratio=0.1)

gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_d),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_g),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
)

x = tf.cast(counts, tf.float32)

y = tf.cast(np.stack((th13s, dcps), axis=1), tf.float32)

gan(np.stack((th13s_o, dcps_o), axis=1))
gan.load_weights(filename)

test = generator.predict(y)
# %%
filenames = 'downloaded_data_for_test_running/20210218-weights.10000.hdf5'
# filenames = sorted(glob.glob((checkpoint_loc + '/*')),
#                    key=lambda x: int(x.split('.')[1]))

gan(np.stack((th13s, dcps), axis=1))

# vals = np.arange(9, len(filenames), 10)
# list = vals.tolist()
# list.insert(0, 0)

vals = np.arange(99, len(filenames), 100)
l = vals.tolist()
l.insert(0, 0)

D_real_pred = []
D_fake_pred = []

gan.load_weights(filenames)
test = generator.predict(np.stack((th13s, dcps), axis=-1))
Full_Poisson_Error, pred_error, error_diff = PoissonError(counts, test)
num = 10000

DFP = discriminator.predict(tf.reshape(
    tf.concat([test, np.stack((th13s, dcps), axis=-1)], axis=-1), (len(test), 42)))
DRP = discriminator.predict(tf.reshape(x_0, (len(x_0), 42)))

D_fake_pred.append(DFP.mean())
D_real_pred.append(DRP.mean())

for index in sorted(random.sample(range(len(counts)), 100)):

    pred = test[index]

    plot_G_outputs(pred, index, error_diff, bins, counts,
                   th13s, dcps, Full_Poisson_Error, num, figure_loc)

    DFP = discriminator.predict(tf.reshape(
        tf.concat([pred, y[index]], axis=0), (1, 42)))
    DRP = discriminator.predict(tf.reshape(x_0[index], (1, 42)))

    print('discriminator prediction on fake image: ', DFP)
    print('discriminator prediction on real image: ', DRP)

plot_D_predictions(l, D_real_pred, D_fake_pred, figure_loc)

# %%


def likelihood(n_i, n_obs):
    #n_i = np.array(n_i)
    #n_obs = np.array(n_obs)

    z = 2 * np.sum(n_i - n_obs + n_obs * np.log(n_obs / n_i), axis=-1)

    return z


# select random true unseen data as event
#observed_index = random.choice(range(len(bins)))
observed_index = int(len(dcps) / 2 + 160)

# observed_index
n_observed_0 = test[observed_index]

PoissonError = np.sqrt(n_observed_0)
half_PoissonError = PoissonError / 2

sampled_mins_th13s = []
sampled_mins_dcps = []

# x = np.load('sin(dcps)_larger_unnormalised.npy')
x = np.load('dcps_between_normed.npy')
x = x.tolist()
# y = np.load('sin2(2th13s)_larger_unnormalised.npy')
y = np.load('th13s_between_normed.npy')
y = y.tolist()

for j in range(10):

    n_observed = np.random.poisson(n_observed_0, size=n_observed_0.shape)

    z = []
    for i in test:
        try:
            n_i = np.rint(i)
            z.append(likelihood(n_i, n_observed))
        except:
            print('error on ', i)

    a = test - counts

    df = pd.DataFrame.from_dict(np.array([x, y, z]).T)
    df.columns = ['sin(dcp)', 'sin^2(2th13)', '-2lnL']
    df['-2lnL'] = pd.to_numeric(df['-2lnL'])
    pivotted = df.pivot('sin^2(2th13)', 'sin(dcp)', '-2lnL')
    min_val = df.idxmin()[2]

    sampled_mins_th13s.append(y[min_val])
    sampled_mins_dcps.append(x[min_val])


n_observed = np.rint(test[observed_index])

z = []
for i in test:
    try:
        n_i = np.rint(i)
        z.append(likelihood(n_i, n_observed))
    except:
        print('error on ', i)

a = test - counts

df = pd.DataFrame.from_dict(np.array([x, y, z]).T)
df.columns = ['sin(dcp)', 'sin^2(2th13)', '-2lnL']
df['-2lnL'] = pd.to_numeric(df['-2lnL'])
pivotted = df.pivot('sin^2(2th13)', 'sin(dcp)', '-2lnL')

# cm=plt.pcolor(pivotted)
# plt.colorbar(cm)
# plt.show()
#plt.figure(figsize=(12, 9))
# %%
fig = plt.scatter(df['sin(dcp)'], df['sin^2(2th13)'], c=df['-2lnL'],
                  cmap='viridis').get_figure()
plt.colorbar()
fig.savefig(figure_loc + '/LikelihoodSurface_Plot.png')

plt.tricontour(df['sin(dcp)'], df['sin^2(2th13)'], df['-2lnL'],
               levels=[2.30], linewidths=0.5, colors='r')
plt.plot(np.array(x[observed_index]), np.array(y[observed_index]), 'ro')
#plt.plot(np.array(sampled_mins_dcps), np.array(sampled_mins_th13s), '.')
plt.ylim(min(y), max(y))
plt.xlim(min(x), max(x))
plt.xticks(np.linspace(-1, 1, 7),
           labels=np.round(np.linspace(-3.14, 3.14, 7), 2))
plt.yticks(np.linspace(-1, 1, 7),
           labels=list(map(lambda x: '{:.4f}'.format(x), np.linspace(0.0874, 0.1, 7))))
plt.xlabel(f'$\delta_{{CP}}$', fontsize=12)
plt.ylabel(f'$sin^2(2\\theta_{{13}})$', fontsize=12)
plt.tight_layout()
plt.savefig(figure_loc + '/LikelihoodSurface_Plot_withContours.png',
            transparent=True, dpi=300)
plt.show()

print('Selected sin^2(2th13):', th13s[observed_index])
print('Selected sin(dcp):', dcps[observed_index])

# %%

input = np.stack((th13s, dcps), axis=-1)
output = np.random.poisson(counts, size=counts.shape).astype(
    float)  # generator.predict(input)
output[0]
test[0]
print('input tensor: \n', input)
print('output tensor: \n', output)

weights = []

for layer_i in range(len(generator.layers)):

    layer_weight = generator.layers[layer_i].get_weights()
    weights.append(layer_weight)

output_r = output.copy()

for i in list(reversed(range(len(weights))))[:-1]:

    output_r -= weights[i][1]
    weights_inv = np.linalg.pinv(weights[i][0])
    output_r = np.matmul(output_r, weights_inv)

print(input)
print(output_r)

print('subtraction \n', input - output_r)
print(output_r)
plt.plot(output_r[:, 0], output_r[:, 1])
plt.plot(input[:, 0], input[:, 1])
plt.show()

# %% Monte Carlo Sampling to find Contours

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10,
    decay_rate=0.96,
    staircase=True)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


def Diff_likelihood(grad, n_i, n_observed):
    dL = 2 * np.sum(tf.cast(grad, 'float32') - (n_observed / n_i)
                    * tf.cast(grad, 'float32'), axis=-1)
    L = likelihood(n_i, n_observed)
    dx = 4 * dL * (L - 2.3)**3

    return dx


observed_index = int(len(dcps) / 2 + 160)
accepted = []

n_observed = test[observed_index]

z = []
niters = 10
trace = []
grad_trace = []


param_val = np.random.uniform(-1, 1, size=(niters, 2))
trace.append(param_val)

var = tf.Variable(param_val)

marker = np.ones((niters, 2))

iter = 0

while sum(np.sum(marker, axis=1) / 2) >= 0.1 * len(marker) and iter < 1000:
    # for i in range(50):
    iter += 1

    with tf.GradientTape() as tape:
        # Forward pass
        tape.watch(var)
        n_i = generator(var)

    # Calculate gradients with respect to every trainable variable
    grad = tape.batch_jacobian(n_i, var)

    dG_dTh = grad.numpy()

    dG_dTh13 = dG_dTh[:, :, 0]
    #dG_dTh13 = [i[0] for i in grad.numpy()[0]]
    dG_ddcp = dG_dTh[:, :, 1]

    grad = tf.stack([dG_dTh[:, :, 0], dG_dTh[:, :, 1]])

    dz = Diff_likelihood(grad, n_i, n_observed)
    # NEED TO ADD CONDITION FOR WHICH DIRECTION TO TRAVEL IN

    Z = (likelihood(n_i, n_observed) - 2.3)**4

    for i in range(len(Z)):
        if Z[i] < 0.00000001:
            # print(iter,i)
            #print('Z[i]', Z[i])
            marker[i] = 0

    # param_val_new[:,0] = param_val[:,0] - (decayed_learning_rate(i,0.000005) * dz_dth13)
    epoch = opt.apply_gradients(zip([np.stack(dz, axis=-1) * marker], [var]))
    if epoch % 10 == 0:
        print('Epoch: ', epoch.numpy())
        print('Learning rate: ', opt._decayed_lr('float32').numpy())
        print('Number of converged points: ', len(
            marker) - sum(np.sum(marker, axis=1) / 2))
    # param_val = param_val_new.numpy()
    trace.append(var.numpy())
    grad_trace.append(dz)

# %%

# x = np.load('sin(dcps)_larger_unnormalised.npy')
x = np.load('dcps_between_normed.npy')
x = x.tolist()
# y = np.load('sin2(2th13s)_larger_unnormalised.npy')
y = np.load('th13s_between_normed.npy')
y = y.tolist()

z = np.load('likelihood_surface_z_for_midpoint.npy')
# z = []
# for i in test:
#     try:
#         n_i_ = np.rint(i)
#         z.append(likelihood(n_i_, n_observed))
#     except:
#         print('error on ', i)
#
# a = test - counts

df = pd.DataFrame.from_dict(np.array([x, y, z]).T)
df.columns = ['sin(dcp)', 'sin^2(2th13)', '-2lnL']
df['-2lnL'] = pd.to_numeric(df['-2lnL'])
pivotted = df.pivot('sin^2(2th13)', 'sin(dcp)', '-2lnL')
# np.shape(trace)
# plt.plot(np.array(x), np.array(y))
# plt.plot(np.array(x[observed_index]), np.array(y[observed_index]), 'ro')
#plt.plot(np.array(accepted)[:, 1], np.array(accepted)[:, 0], '.')
plt.plot(np.array(trace)[:, :, 1], np.array(trace)[:, :, 0], 'k.')
plt.plot(np.array(trace)[-1, :, 1], np.array(trace)[-1, :, 0], 'rx')
fig = plt.scatter(df['sin(dcp)'], df['sin^2(2th13)'], c=df['-2lnL'],
                  cmap='viridis').get_figure()
fig.savefig(figure_loc + '/LikelihoodSurface_Plot.png')

plt.tricontour(df['sin(dcp)'], df['sin^2(2th13)'], df['-2lnL'],
               levels=[2.30], linewidths=0.5, colors='r')
plt.plot(np.array(x[observed_index]), np.array(y[observed_index]), 'r.')


# plt.plot(np.array(sampled_mins_dcps), np.array(sampled_mins_th13s), '.')
plt.ylim(min(y), max(y))
plt.xlim(min(x), max(x))
# plt.xticks(np.linspace(-1,1,7),labels=np.round(np.linspace(-3.14,3.14,7),2))
# plt.yticks(np.linspace(-1,1,7),labels=list(map(lambda x: '{:.4f}'.format(x),np.linspace(0.0874,0.1,7))))
plt.xlabel(f'$\delta_{{CP}}$', fontsize=12)
plt.ylabel(f'$sin^2(2\\theta_{{13}})$', fontsize=12)
plt.tight_layout()
plt.savefig(figure_loc + '/LikelihoodSurface_Plot_withContours.png',
            transparent=True, dpi=300)
plt.show()

plt.close()

plt.plot(abs(np.array(grad_trace))[:, 0, :])
plt.show()

# dG_dTh = grad.numpy()[0][0][0][0].flatten()
# dG_dTH_split[0] = []
#
# for i in range(40):
#     insert = []
#     insert.append(dG_dTh[2*i])
#     insert.append(dG_dTh[(2*i)+1])
#     dG_dTH_split.append(insert)
# %%

'''

x = tf.constant([np.stack((th13s, dcps), axis=1)[1000]])

with tf.GradientTape() as tape:
    # Forward pass
    tape.watch(x)
    y = generator(x)

# Calculate gradients with respect to every trainable variable

grad = tape.jacobian(y, x)
grad = grad.numpy().reshape(40,2)

'''
# %%
############ TOY MODEL TO TEST #######################

'''
def toy_dense(th13s, dcps):

    params = np.stack((th13s, dcps), axis=-1)

    generator_input = keras.Input(
        shape=(np.shape(params)[1]))  # params = (100,2)

    dropout = 0.5
    init = tf.keras.initializers.RandomUniform(minval=-1, maxval=2, seed=None)

    x = tf.keras.layers.Dense(
        40, kernel_initializer=init, bias_initializer=init)(generator_input)

    # x = tf.keras.layers.Dense(
    #     20, kernel_initializer=init, bias_initializer=init)(x)
    #
    # x = tf.keras.layers.Dense(
    #     30, kernel_initializer=init, bias_initializer=init)(x)

    x = tf.keras.layers.Dense(
        40, kernel_initializer=init, bias_initializer=init)(x)

    generator_output = tf.keras.layers.Dense(
        40, kernel_initializer=init, bias_initializer=init)(x)

    generator_model = keras.Model(
        generator_input, generator_output, name='generator_model')

    optimizer = Adam(lr=0.01)

    generator_model.compile(loss='MeanSquaredError',
                            optimizer=optimizer, metrics=['MeanSquaredError'])

    return generator_model


toy_dense = toy_dense(th13s, dcps)

toy_dense.fit(x=np.stack((th13s, dcps), axis=-1), y=counts, epochs=100)
'''

# %%
'''
input = np.stack((th13s, dcps), axis=-1)
output = toy_dense(input).numpy().tolist()
print('input tensor: \n', input)
print('output tensor: \n', output)

weights_toy = []

for layer_i in range(len(toy_dense.layers)):

    layer_weight = toy_dense.layers[layer_i].get_weights()
    weights_toy.append(layer_weight)

output_r = output.copy()

weights_toy[1][0].shape

for i in list(reversed(range(len(weights_toy))))[:-1]:

    output_r -= weights_toy[i][1]
    output_r = np.matmul(output_r, np.linalg.pinv(weights_toy[i][0]))


plt.plot(input[:, 0], input[:, 1])
plt.plot(output_r[:, 0], output_r[:, 1])
'''
# %%
'''
counts[0]
input = np.stack((th13s, dcps), axis=-1)
output = toy_dense(input).numpy()
output[0]
print('input tensor: \n', input)
print('output tensor: \n', output)

weights_toy = []

for layer_i in range(len(toy_dense.layers)):

    layer_weight = toy_dense.layers[layer_i].get_weights()
    weights_toy.append(layer_weight)

output_r = output.copy()

weights_toy[1][0].shape

for i in list(reversed(range(len(weights_toy))))[:-1]:

    output_r -= weights_toy[i][1]
    inv_weight = np.linalg.pinv(weights_toy[i][0])
    output_r = np.matmul(output_r, inv_weight)

print(input)
print(output_r)

plt.plot(input[:, 0], input[:, 1])
plt.plot(output_r[:, 0], output_r[:, 1])

a = (input - output_r)
#plt.plot(abs(a[:,0])+abs(a[:,1]), ',')
# weights
# weights_toy

'''
