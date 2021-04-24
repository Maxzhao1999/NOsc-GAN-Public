from modules import *
from Data_merger import *


bins = np.load('bins_0.0874-0.099969-320_-3.121958-3.121958-319.npy')
counts = np.load('counts_0.0874-0.099969-320_-3.121958-3.121958-319.npy')
th13s = np.load('th13s_102080_normed.npy')
th13s = th13s.tolist()
dcps = np.load('dcps_102080_normed.npy')
dcps = dcps.tolist()

batch_size = 256

file_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

results_loc = "results/" + file_time

os.mkdir(results_loc)

figure_loc = results_loc + "/figures"
regress_checkpoint_loc = results_loc + "/regress_checkpoint"

os.mkdir(figure_loc)
os.mkdir(regress_checkpoint_loc)

regress_weights = get_regress_model_weights1(
    counts, th13s, dcps, results_loc, figure_loc, regress_checkpoint_loc, epochs=10000, batch_size=batch_size, lr=0.00000001, plotting=True)
