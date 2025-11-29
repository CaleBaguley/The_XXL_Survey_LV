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
Target_Count = 623

# Confidence binning
Conf_bin_edges = np.asarray([0.05*i for i in range(18)])

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

# -- Find cut on EXT_LIKE to achive target sample count -------------------------------------------------------------------------

North_Data_else = North_Data_else[North_Data_else[:,2].argsort()][::-1]

Non_C1C2_count = Target_Count - (len(North_Data_C1) + len(North_Data_C2))

print(Non_C1C2_count, len(North_Data_C1), len(North_Data_C2), Non_C1C2_count + len(North_Data_C1) + len(North_Data_C2) )

EXT_LIKE_cut = North_Data_else[Non_C1C2_count,2]

#Keep N highest EXT_LIKE non-C1C2 objects
North_Data_else = North_Data_else[:Non_C1C2_count]

print(len(North_Data_else))

confidence_counts = np.histogram(North_Data_else[:,0], bins = Conf_bin_edges)[0]

print(f"EXT_LIKE_cut = {EXT_LIKE_cut}")
print(f"confidence_counts = {confidence_counts}")
print(f"len(North_Data_C1) = {len(North_Data_C1)}")
print(f"len(North_Data_C2) = {len(North_Data_C2)}")

# -- Generate random purity samples --------------------------------------------------------------------------------------------

# Sample random purity for C1 and C2 sources
C1_purity_samples = beta.rvs(1 + C1_lotto_cluster_count, 1 + C1_lotto_size - C1_lotto_cluster_count, size = sample_size)
C2_purity_samples = beta.rvs(1 + C2_lotto_cluster_count, 1 + C2_lotto_size - C2_lotto_cluster_count, size = sample_size)

# setup array to contain random purities for non-C1C2 sources in confidence each bin
else_purity_samples = np.ones([len(bin_else_lotto_sizes), sample_size])

# randomly sample beta distribution for each confidence bin
for i in range(len(bin_else_lotto_sizes)):
    else_purity_samples[i] = beta.rvs(1 + bin_else_lotto_cluster_counts[i], 1 + bin_else_lotto_sizes[i] - bin_else_lotto_cluster_counts[i], size = sample_size)

# -- Calculate combined purity samples -----------------------------------------------------------------------------------------

cluster_count = np.einsum("ij,i->j", else_purity_samples, confidence_counts)

cluster_count += len(North_Data_C1)*C1_purity_samples
cluster_count += len(North_Data_C2)*C2_purity_samples

purity_values = cluster_count/Target_Count

print("mean", np.mean(purity_values))
print("Purity 16th, 50th and 84th percentile:", np.percentile(purity_values, [15.9, 50, 84.1]))

plt.hist(purity_values, bins = np.linspace(0,1, 100))
plt.show()







