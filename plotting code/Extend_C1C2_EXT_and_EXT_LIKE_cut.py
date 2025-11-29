import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from scipy.stats import beta 

# -- Setup paramiters -----------------------------------------------------------------------------------------------------------

sample_size = 10000

C1_EXT_LIKE_cut = 33
C2_EXT_LIKE_cut = 15
EXT_cut = 5

# target number of sources in sample
target_count = 623
target_range = 10

# Confidence binning
Conf_bin_edges = np.asarray([0.05*i for i in range(19)])

EXT_bin_count = 200
EXT_LIKE_bin_count = 200

# -- Data import and setup ------------------------------------------------------------------------------------------------------

# -- North data

# Confidence
Conf_Training_Results = pd.read_csv("../Data_runs/Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv")

# Paramiters
North_Data = pd.read_csv("../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_Pointings_Cleaned_03.csv")

# Combine data with confidence values
North_Data = pd.merge(North_Data, Conf_Training_Results, left_on = "Index", right_on = "id")

# Reduce data to relevant columns
North_Data = North_Data[["EXT", "EXT_LIKE", "Conf mean", "C1C2"]]

# Split into C1, C2 and non-C1C2
North_Data_C1   = np.asarray(North_Data.loc[North_Data["C1C2"] == 1][["Conf mean", "EXT", "EXT_LIKE"]])
North_Data_C2   = np.asarray(North_Data.loc[North_Data["C1C2"] == 2][["Conf mean", "EXT", "EXT_LIKE"]])
North_Data_else = North_Data.loc[North_Data["C1C2"] == 0][["Conf mean", "EXT", "EXT_LIKE"]]

# Only need sources with EXT grater than 5 arcseconds
North_Data_else = np.asarray(North_Data_else.loc[North_Data_else["EXT"] > 5][["Conf mean", "EXT", "EXT_LIKE"]])

print(f"len(North_Data_else) = {len(North_Data_else)}")

# -- Lotto data --
bin_else_lotto_sizes          = np.asarray([107, 136, 176, 57, 25, 15, 9, 3, 8, 6, 3, 2, 4, 2, 2, 2, 0, 0])
bin_else_lotto_cluster_counts = np.asarray([ 12,  39,  41, 25,  7,  6, 4, 2, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0])

bin_C1_lotto_sizes          = np.asarray([0, 3, 8, 6, 6, 6, 6, 7, 5, 3, 8, 11, 10, 21, 11, 9, 9, 11, 0])
bin_C1_lotto_cluster_counts = np.asarray([0, 0, 2, 3, 5, 4, 6, 7, 5, 3, 8, 10, 10, 20, 11, 9, 9, 10, 0])

bin_C2_lotto_sizes          = np.asarray([0, 20, 28, 13, 6, 6, 6, 7, 2, 1, 7, 1, 5, 0, 1, 2, 0, 0])
bin_C2_lotto_cluster_counts = np.asarray([0,  8, 13,  8, 5, 5, 4, 6, 2, 1, 6, 1, 3, 0, 0, 2, 0, 0])

C1_lotto_size          = np.sum(bin_C1_lotto_sizes)
C1_lotto_cluster_count = np.sum(bin_C1_lotto_cluster_counts)

C2_lotto_size          = np.sum(bin_C2_lotto_sizes)
C2_lotto_cluster_count = np.sum(bin_C2_lotto_cluster_counts)

print(f"C1_lotto_size: {C1_lotto_size}")
print(f"C1_lotto_cluster_count: {C1_lotto_cluster_count}")
print(f"C2_lotto_size: {C2_lotto_size}")
print(f"C2_lotto_cluster_count: {C2_lotto_cluster_count}")

# -- Find cut on EXT_LIKE to achive target sample count ----------------------------------------------------------------

# Create bin edges
EXT_range = np.log10([np.min(North_Data['EXT']), np.max(North_Data['EXT'])])
EXT_LIKE_range = np.log10([max(np.min(North_Data['EXT_LIKE']),10**-4), np.max(North_Data['EXT_LIKE'])])

EXT_bins = 10**(np.linspace(EXT_range[0], EXT_range[1], EXT_bin_count))
EXT_LIKE_bins = 10**(np.linspace(EXT_LIKE_range[0], EXT_LIKE_range[1], EXT_LIKE_bin_count))
Conf_mean_bins = [0.05*i for i in range(0,21)]

# Bin all data
all_data_count, edges = np.histogramdd(North_Data[['EXT', 'EXT_LIKE']].values, bins = [EXT_bins, EXT_LIKE_bins])

all_data_individual_bin_count = cp.deepcopy(all_data_count)

plt.imshow(np.log10(all_data_count))
plt.show()

# Convert to number above cut
# Make EXT_LIKE inverse cumulative
for i in range(EXT_LIKE_bin_count-3,-1,-1):
    all_data_count[:,i] += all_data_count[:,i+1]

# Make EXT inverse cumulative
for i in range(EXT_bin_count-3, -1, -1):
    all_data_count[i,:] += all_data_count[i+1,:]

plt.imshow(np.log10(all_data_count))
plt.show()

# Find bins where the count is within target_range 0f target_count
targets = np.where((all_data_count > target_count-target_range) * (all_data_count < target_count + target_range))

print(targets)

# We want to keep all of the C1 and C2 sources
# Remove bins where the EXT is larger than EXT_cut
new_targets = []
new_targets.append(targets[0][np.where(EXT_bins[targets[0]] < EXT_cut)[0]])
new_targets.append(targets[1][np.where(EXT_bins[targets[0]] < EXT_cut)[0]])
targets = cp.deepcopy(new_targets)

# Remove bins where the EXT_LIKE is larger than C2_EXT_LIKE_cut (this will include C1 sources)
new_targets = []
new_targets.append(targets[0][np.where(EXT_LIKE_bins[targets[1]] < C2_EXT_LIKE_cut)[0]])
new_targets.append(targets[1][np.where(EXT_LIKE_bins[targets[1]] < C2_EXT_LIKE_cut)[0]])
targets = cp.deepcopy(new_targets)

print(targets)

target_bin_count = len(targets[0])

target_bin_total_count = all_data_count[targets[0], targets[1]]
print(f"target bin source counts: {target_bin_total_count}")

plt.imshow(np.log10(all_data_individual_bin_count))
plt.scatter(targets[0],targets[1], c = 'black')
plt.show()

# -- Get source contents of each cut -----------------------------------------------------------------------------------
# Bin non-C1C2 data
bined_else_data, edges = np.histogramdd(North_Data_else,
                                        bins = [EXT_bins, EXT_LIKE_bins, Conf_bin_edges])

# Convert to number above cut
# Make EXT_LIKE inverse cumulative
for i in range(EXT_LIKE_bin_count-3,-1,-1):
    bined_else_data[:,i] += bined_else_data[:,i+1]

# Make EXT inverse cumulative
for i in range(1,EXT_bin_count-1):
    bined_else_data[i,:] += bined_else_data[i-1,:]

# Create array of the selected cuts data
selected_bin_else_data = bined_else_data[targets[0], targets[1], :]

# Calculate the number of non-C1C2 sources in the cut bin
selected_bin_else_count = np.sum(selected_bin_else_data, axis = 1)

# -- Monticarlo sample purity ------------------------------------------------------------------------------------------

# Generate base sample purities
C1_purity_samples = beta.rvs(1 + C1_lotto_cluster_count,
                             1 + C1_lotto_size - C1_lotto_cluster_count,
                             size = sample_size)
C2_purity_samples = beta.rvs(1 + C2_lotto_cluster_count,
                             1 + C2_lotto_size - C2_lotto_cluster_count,
                             size = sample_size)

# setup array to contain random purities for non-C1C2 sources in confidence each bin
else_purity_samples = np.ones([len(bin_else_lotto_sizes), sample_size])

# randomly sample beta distribution for each confidence bin
for i in range(len(bin_else_lotto_sizes)):
    else_purity_samples[i] = beta.rvs(1 + bin_else_lotto_cluster_counts[i],
                                      1 + bin_else_lotto_sizes[i] - bin_else_lotto_cluster_counts[i],
                                      size = sample_size)

# Calculate expected cluster content
C1_expected_cluster_count = len(North_Data_C1) * C1_purity_samples
C2_expected_cluster_count = len(North_Data_C2) * C2_purity_samples

non_C1C2_expected_cluster_counts_bined = []
non_C1C2_expected_cluster_counts = []
expected_cluster_counts = []
target_bin_purity_samples = []

# Loop over the different sample bins
for i in range(target_bin_count):
    # Array to hold the different cluster counts for each confidence bin
    non_C1C2_expected_cluster_counts_bined.append(np.zeros(else_purity_samples.shape))

    # Calulate the expected number of clusters for each confidence bin
    for j in range(sample_size):
        non_C1C2_expected_cluster_counts_bined[-1][:,j] = selected_bin_else_data[i] * else_purity_samples[:,j]

    # Calculate the expected number of non-C1C2 clusters in the sample
    non_C1C2_expected_cluster_counts.append(np.sum(non_C1C2_expected_cluster_counts_bined[-1], axis = 0))

    # Calculate the total number of expected clusters in the current target bin for each sample
    expected_cluster_counts.append(non_C1C2_expected_cluster_counts[-1]
                                   + C1_expected_cluster_count
                                   + C2_expected_cluster_count)

    # Calcualte the purity of the current target bin for all samples
    target_bin_purity_samples.append(expected_cluster_counts[-1]/target_bin_total_count[i])

# -- Results -----------------------------------------------------------------------------------------------------------

medians = []
plus_error = []
minus_error = []

print("targets", targets)

print("EXT cut, EXT_LIKE cut, source count, purity, +error, -error")
for i in range(target_bin_count):
    purity = np.percentile(target_bin_purity_samples[i], [15.9, 50, 84.1])

    medians.append(purity[1])
    plus_error.append(purity[2] - medians[-1])
    minus_error.append(purity[0] - medians[-1])

    print(f"{EXT_bins[targets[0][i]]:.4e}, {EXT_LIKE_bins[targets[1][i]]:.4e}, {target_bin_total_count[i]}, {medians[-1]:.4f}, {plus_error[-1]:.4f}, {minus_error[-1]:.4f}")

optimal_cut_id = np.argmax(medians)

print("\nOptimal cut choice:")
print(f"EXT: {EXT_bins[targets[0][optimal_cut_id]]:.4e}")
print(f"EXT_LIKE: {EXT_LIKE_bins[targets[1][optimal_cut_id]]:.4e}")
print(f"Sample source count: {int(target_bin_total_count[optimal_cut_id])}")
print(f"Sample purity: {medians[optimal_cut_id]:.4f} + {plus_error[optimal_cut_id]:.4} - {minus_error[optimal_cut_id]:4f}")