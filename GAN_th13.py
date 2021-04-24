#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:01:27 2020

@author: luanawilliams
"""

####### IMPORT REQUIRED PACKAGES ######

from modules_th13 import *
%load_ext autoreload
%autoreload 2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # what does this do?
# %load_ext tensorboard

# %%
####### Get the required dataset #######

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

##### Load the data set for 102400 samples (already normalised) ####
# bins = np.load('bins_0.0874-0.099969-320_-3.141593-3.121958-320.npy')
# bins = bins.tolist()
# counts = np.load('counts_0.0874-0.099969-320_-3.141593-3.121958-320.npy')
# counts = np.float32(counts)
# th13s = np.load('th13s_102400_normed.npy')
# th13s = th13s.tolist()
# dcps = np.load('dcps_102400_normed.npy')
# dcps = dcps.tolist()

bins, counts, th13s, dcps = get_dataset(saving=True, plotting=False)

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

x = counts
y = th13s
batch_size = 32
# np.float32(np.random.normal(1, counts.std()*0.001, (5,1)))
train_data, val_data = preprocess(x, y, batch_size=batch_size)

y = np.concatenate((np.ones(counts.shape[0]), np.zeros(counts.shape[0])))

# x = np.concatenate((counts, np.random.rand(counts.shape)), axis=0)
x = np.concatenate((counts+abs(counts+np.random.normal(0, counts.std()*0.2, counts.shape)), np.random.normal(counts.mean(), counts.std(), counts.shape)), axis=0)

train, val = preprocess(x, y, batch_size=32, validation_split=0.1)

regress_weights = get_regress_model_weights(
    counts, th13s, epochs=1, lr=0.00001)
discriminator = define_discriminator(counts, regress_weights, 2, lr=0.0001)
#pred1 = generator.predict(np.stack((th13s, dcps), axis=-1))[np.random.randint(0, len(th13s))]

# a=abs(counts+np.random.normal(0, counts.std()*0.2, counts.shape))

# plt.hist(bins[0][:-1], bins[0], weights=generator.predict(np.array([[0.5, 0.5], [0.5, 0.5]]))[1].reshape(-1))
# plt.hist(bins[0][:-1], bins[0], weights=a[0])

discriminator.fit(train, validation_data=val, epochs=20, verbose=2)


generator = define_generator(th13s)

def generator_loss(y_true, y_pred):
    return tf.reduce_mean(abs(y_true-y_pred))

x = th13s
y = counts
train, val = preprocess(x, y, batch_size=32, validation_split=0.1)
generator.compile(loss=generator_loss,
                      optimizer=keras.optimizers.Adam(learning_rate=0.0001))

generator.fit(train, validation_data=val, epochs=20)

output = generator.predict(th13s)[np.random.randint(0, len(th13s))]
plt.hist(bins[0][:-1], bins[0], weights=output)
plt.show()


tensorboard_callback, lr_reduce, earlystop_callback = get_callbacks(
    tensorboard_profiler=True, delete_logs=False)

gan = GAN(discriminator=discriminator,
          generator=generator, batch_size=batch_size, logging=None, train_ratio=0.1) # bigger train_ratio more generator training
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(train_data, validation_data=val_data, epochs=1000, callbacks=[model_checkpoint_callback], verbose=2)



# %%

filenames = sorted(glob.glob('checkpoint/*'),
                   key=lambda x: int(x.split('.')[1]))
list = np.arange(0, 1024, 32, dtype=np.int)

gan(np.array(th13s))

shutil.rmtree('figures/')
os.makedirs('figures')

for epoch in range(0,len(filenames),20):
# for epoch in range(1):
# epoch = 199
    gan.load_weights(filenames[epoch])
    test = generator.predict(th13s)
    num = int(filenames[epoch].split('.')[1])
    for index in sorted(random.sample(range(len(counts)), 3)):
        # for index in list:
        #index = np.random.randint(0, len(th13s))
        print('th13:', th13s[index])

        print('epoch:', num)
        pred = test[index]

        #plt.hist(bins[0][:-1], bins[0], weights=generator.predict(np.array([[0.5, 0.5], [0.5, 0.5]]))[1].reshape(-1))
        rand_no = np.random.randint(0, len(counts[0]))
        #plt.hist(bins[0][:-1], bins[0], weights=counts[rand_no], label='Real')

        # plot all histograms from each 100 epochs
        # for hist in gan.hist:
        plt.close()
        plt.hist(bins[0][:-1], bins[0], weights=counts[index], label='real')
        plt.hist(bins[0][:-1], bins[0], weights=pred,
                 label='fake', histtype='step')
        plt.xlabel('Reconstructed Energy [GeV]', fontsize=15)
        plt.ylabel(r'$\nu_e$ appearance events', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=12)
        # plt.savefig(
        #     f'figures/generated_hist.{epoch}-{th13s[index]:{10}.{2}}-{dcps[index]:{10}.{2}}.png')
        plt.show()
        print('discriminator prediction on fake image: ',
              discriminator.predict(tf.reshape(pred, (1, 40))))
        print('discriminator prediction on real image: ',
              discriminator.predict(tf.reshape(counts[index], (1, 40))))
