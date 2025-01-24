import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from scipy.stats import beta 
from scipy.optimize import curve_fit

# -- Setup paramiters -----------------------------------------------------------------------------------------------------------

sample_size = 1000

C1_EXT_LIKE_cut = 33
C2_EXT_LIKE_cut = 15
EXT_cut = 5

# target sample purity
Target_Count = 623
Target_Count_Error = 10

# -- Set bin values --

# Confidence binning
Conf_bin_edges = np.asarray([0.05*i for i in range(18)])
Conf_bin_midle = (Conf_bin_edges[:-1] + Conf_bin_edges[1:])/2.

# EXT logarithmic binning
EXT_bin_edges = 10**np.linspace(-4,3,8*7+1)
EXT_bin_midle = (EXT_bin_edges[:-1] + EXT_bin_edges[1:])/2.

# EXT_LIKE logarithmic binning
EXT_LIKE_bin_edges = 10**np.linspace(-4,5,8*10+1)
EXT_LIKE_bin_midle = (EXT_LIKE_bin_edges[:-1] + EXT_LIKE_bin_edges[1:])/2.

# -- Data import and setup ------------------------------------------------------------------------------------------------------

# -- North data

# Confidence
Conf_Training_Results = pd.read_csv("../Data_runs/Sample simdata/BG_Rate/North_cleaned/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv")

# Paramiters
North_Data = pd.read_csv("../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_Pointings_Cleaned_03.csv")

# Combine data with confidence values
North_Data = pd.merge(North_Data, Conf_Training_Results, left_on = "Index", right_on = "id")

# Reduce data to relevant columns
North_Data = North_Data[["EXT", "EXT_LIKE", "Conf mean", "C1C2"]]

# Split into C1, C2 and non-C1C2
North_Data_C1   = np.asarray(North_Data.loc[North_Data["C1C2"] == 1][["Conf mean", "EXT", "EXT_LIKE"]])
North_Data_C2   = np.asarray(North_Data.loc[North_Data["C1C2"] == 2][["Conf mean", "EXT", "EXT_LIKE"]])
North_Data_else = np.asarray(North_Data.loc[North_Data["C1C2"] == 0][["Conf mean", "EXT", "EXT_LIKE"]])

# -- Lotto data --
bin_else_lotto_sizes          = np.asarray([105, 106, 23, 27, 24, 19, 4, 13, 8, 6, 5, 5, 3, 2, 3, 0, 1])
bin_else_lotto_cluster_counts = np.asarray([  6,   8,  5,  5,  3,  6, 0,  3, 2, 1, 1, 1, 0, 1, 0, 0, 0])

C1_lotto_size          = 121
C1_lotto_cluster_count = 109

C2_lotto_size          = 20
C2_lotto_cluster_count = 13

# -- Bin Data ------------------------------------------------------------------------------------------------------------------

C1_bin_counts  , edges = np.histogramdd(North_Data_C1  , bins = [Conf_bin_edges, EXT_bin_edges, EXT_LIKE_bin_edges])
C2_bin_counts  , edges = np.histogramdd(North_Data_C2  , bins = [Conf_bin_edges, EXT_bin_edges, EXT_LIKE_bin_edges])
else_bin_counts, edges = np.histogramdd(North_Data_else, bins = [Conf_bin_edges, EXT_bin_edges, EXT_LIKE_bin_edges])

# -- Generate random purity samples --------------------------------------------------------------------------------------------

# Sample random purity for C1 and C2 sources
C1_purity_samples = beta.rvs(1 + C1_lotto_cluster_count, 1 + C1_lotto_size - C1_lotto_cluster_count, size = sample_size)
C2_purity_samples = beta.rvs(1 + C2_lotto_cluster_count, 1 + C2_lotto_size - C2_lotto_cluster_count, size = sample_size)

# setup array to contain random purities for non-C1C2 sources in confidence each bin
else_purity_samples = np.ones([len(bin_else_lotto_sizes), sample_size])

# randomly sample beta distribution for each confidence bin
for i in range(len(bin_else_lotto_sizes)):
    else_purity_samples[i] = beta.rvs(1 + bin_else_lotto_cluster_counts[i], 1 + bin_else_lotto_sizes[i] - bin_else_lotto_cluster_counts[i], size = sample_size)

# -- Calculate sampled cluster count in each bin ----------------------------------------------------------------------------

# We should end up with two results arrays with shape [len(conf_bins), len(EXT_bins), len(EXT_LIKE_bins), sample_size]
# One for all data and one for only the non-C1C2 sources.

# Calcualte cluster counts for each purity sample for C1 and C2 objects
# This is calculated by taking the bined source count distribution and then for each sample multiplying by the sampled purity.
# The result is an array of size [sample_size, len(conf_bins), len(EXT_bins), len(EXT_LIKE_bins)] conatining a distribution of
# clusters for each purity sample.
C1_cluster_counts = np.einsum("ijk,l->lijk", C1_bin_counts, C1_purity_samples)
C2_cluster_counts = np.einsum("ijk,l->lijk", C2_bin_counts, C2_purity_samples)

# Calcualte cluster counts for each purity sample for non-C1C2 objects
# This is calculated by taking the bined source count distribution and then for each sample multiplying by the sampled purity
# in each conidence bin. The result is an array of size [sample_size, len(conf_bins_midle), len(EXT_bins_midle), len(EXT_LIKE_bins_midle)]
# conatining a distribution of clusters for each purity sample.
else_cluster_counts = np.einsum("ijk,il->lijk", else_bin_counts, else_purity_samples)

# Combine data to use all class samples
All_bin_counts = C1_bin_counts + C2_bin_counts + else_bin_counts
All_cluster_counts = C1_cluster_counts + C2_cluster_counts + else_cluster_counts

# -- Get array of EXT EXT_LIKE bins containing at least one source ----------------------------------------------------------

All_Contains_no_source  = np.einsum("ijk -> jk", All_bin_counts)
else_Contains_no_source = np.einsum("ijk -> jk", else_bin_counts)

All_Contains_no_source  = All_Contains_no_source  <= 0
else_Contains_no_source = else_Contains_no_source <= 0

# -- Make inverse cumulative --------------------------------------------------------------------------------------------------

# Setup for source counts
All_bin_counts_cumulative  = cp.deepcopy(All_bin_counts)
else_bin_counts_cumulative = cp.deepcopy(else_bin_counts)

# Setup for cluster counts
All_cluster_counts_cumulative  = cp.deepcopy(All_cluster_counts)
else_cluster_counts_cumulative = cp.deepcopy(else_cluster_counts)

# iterate backwards over conf bins
for i in range(2,len(Conf_bin_midle)+1):
    
    # Calculate for source counts
    All_bin_counts_cumulative [-i,:,:] += All_bin_counts_cumulative [1-i,:,:] 
    else_bin_counts_cumulative[-i,:,:] += else_bin_counts_cumulative[1-i,:,:]
    
    # Calculate for cluster counts
    All_cluster_counts_cumulative [:,-i,:,:] += All_cluster_counts_cumulative [:,1-i,:,:]
    else_cluster_counts_cumulative[:,-i,:,:] += else_cluster_counts_cumulative[:,1-i,:,:]
    
# iterate backwards over EXT bins
for i in range(2,len(EXT_bin_midle)+1):
    
    # Calculate for source counts
    All_bin_counts_cumulative [:,-i,:] += All_bin_counts_cumulative [:,1-i,:] 
    else_bin_counts_cumulative[:,-i,:] += else_bin_counts_cumulative[:,1-i,:]
    
    # Calculate for cluster counts
    All_cluster_counts_cumulative [:,:,-i,:] += All_cluster_counts_cumulative [:,:,1-i,:]
    else_cluster_counts_cumulative[:,:,-i,:] += else_cluster_counts_cumulative[:,:,1-i,:]
    
# iterate backwards over EXT bins
for i in range(2,len(EXT_LIKE_bin_midle)+1):
    
    # Calculate for source counts
    All_bin_counts_cumulative [:,:,-i] += All_bin_counts_cumulative [:,:,1-i] 
    else_bin_counts_cumulative[:,:,-i] += else_bin_counts_cumulative[:,:,1-i]
    
    # Calculate for cluster counts
    All_cluster_counts_cumulative [:,:,:,-i] += All_cluster_counts_cumulative [:,:,:,1-i]
    else_cluster_counts_cumulative[:,:,:,-i] += else_cluster_counts_cumulative[:,:,:,1-i]
    
# -- Convert to EXT > cut OR EXT_LIKE > cut ----------------------------------------------------------------------------------
# The counts above work if we requier sources to pass both cuts, however we want sorces that pass atleast one instead of requiering both.

# array of number above EXT_LIKE cut
# For all sources
All_bin_counts_selected  = np.einsum("ij,ijk->ijk", All_bin_counts_cumulative [:,:,0], np.ones(All_bin_counts_cumulative.shape))
else_bin_counts_selected = np.einsum("ij,ijk->ijk", else_bin_counts_cumulative[:,:,0], np.ones(else_bin_counts_cumulative.shape))

# For sampled cluster counts
All_clusters_selected_counts  = np.einsum("ijk,ijkl->ijkl", All_cluster_counts_cumulative [:,:,:,0], np.ones(All_cluster_counts_cumulative.shape))
else_clusters_selected_counts = np.einsum("ijk,ijkl->ijkl", else_cluster_counts_cumulative[:,:,:,0], np.ones(else_cluster_counts_cumulative.shape))

# add array of number above EXT cut
# For all sources
All_bin_counts_selected  += np.einsum("ik,ijk->ijk", All_bin_counts_cumulative [:,0,:], np.ones(All_bin_counts_cumulative.shape))
else_bin_counts_selected += np.einsum("ik,ijk->ijk", else_bin_counts_cumulative[:,0,:], np.ones(else_bin_counts_cumulative.shape))

# For sampled cluster counts
All_clusters_selected_counts  += np.einsum("ijl,ijkl->ijkl", All_cluster_counts_cumulative [:,:,0,:], np.ones(All_cluster_counts_cumulative.shape))
else_clusters_selected_counts += np.einsum("ijl,ijkl->ijkl", else_cluster_counts_cumulative[:,:,0,:], np.ones(else_cluster_counts_cumulative.shape))

# We have doube counted source that are above both cuts so subtract this count
# For all sources
All_bin_counts_selected  -= All_bin_counts_cumulative
else_bin_counts_selected -= else_bin_counts_cumulative

# For sampled cluster counts
All_clusters_selected_counts  -= All_cluster_counts_cumulative
else_clusters_selected_counts -= else_cluster_counts_cumulative

# -- Establish which bins in EXT EXT_LIKE space contain atleast one source ----------------------------------------------------
    
All_Contains_no_source_cumulative  = np.einsum("ijk -> jk", All_bin_counts_cumulative)
else_Contains_no_source_cumulative = np.einsum("ijk -> jk", else_bin_counts_cumulative)

All_Contains_no_source_cumulative  = All_Contains_no_source_cumulative  <= 0
else_Contains_no_source_cumulative = else_Contains_no_source_cumulative <= 0

# -- Calculate purity and compleatness of samples -----------------------------------------------------------------------------

# -- Purity --
All_purity  = np.einsum("ijkl,jkl->ijkl", All_cluster_counts , 1/All_bin_counts)
else_purity = np.einsum("ijkl,jkl->ijkl", else_cluster_counts, 1/else_bin_counts)

All_purity_cumulative_rand_samples  = np.einsum("ijkl,jkl->ijkl", All_cluster_counts_cumulative , 1/All_bin_counts_cumulative)
else_purity_cumulative = np.einsum("ijkl,jkl->ijkl", else_cluster_counts_cumulative, 1/else_bin_counts_cumulative)

All_purity_sample  = np.einsum("ijkl,jkl->ijkl", All_clusters_selected_counts , 1/All_bin_counts_selected)
else_purity_sample = np.einsum("ijkl,jkl->ijkl", else_clusters_selected_counts, 1/else_bin_counts_selected)

# -- Compleatness --
All_compleatness  = np.einsum("ijkl,i->ijkl", All_cluster_counts , 1/All_cluster_counts_cumulative[:,0,0,0])
else_compleatness = np.einsum("ijkl,i->ijkl", else_cluster_counts, 1/All_cluster_counts_cumulative[:,0,0,0])

All_compleatness_cumulative  = np.einsum("ijkl,i->ijkl", All_cluster_counts_cumulative , 1/All_cluster_counts_cumulative[:,0,0,0])
else_compleatness_cumulative = np.einsum("ijkl,i->ijkl", else_cluster_counts_cumulative, 1/All_cluster_counts_cumulative[:,0,0,0])

All_compleatness_sample  = np.einsum("ijkl,i->ijkl", All_clusters_selected_counts , 1/All_cluster_counts_cumulative[:,0,0,0])
else_compleatness_sample = np.einsum("ijkl,i->ijkl", else_clusters_selected_counts, 1/All_cluster_counts_cumulative[:,0,0,0])

# -- Calculate mean purity and compleatness -----------------------------------------------------------------------------------
# Summs over sample dimention and devides by number of samples

All_purity  = np.einsum("ijkl->jkl", All_purity ) / sample_size
else_purity = np.einsum("ijkl->jkl", else_purity) / sample_size

All_compleatness  = np.einsum("ijkl->jkl", All_compleatness ) / sample_size
else_compleatness = np.einsum("ijkl->jkl", else_compleatness) / sample_size

All_purity_cumulative  = np.einsum("ijkl->jkl", All_purity_cumulative_rand_samples ) / sample_size
else_purity_cumulative = np.einsum("ijkl->jkl", else_purity_cumulative) / sample_size

All_compleatness_cumulative  = np.einsum("ijkl->jkl", All_compleatness_cumulative ) / sample_size
else_compleatness_cumulative = np.einsum("ijkl->jkl", else_compleatness_cumulative) / sample_size

All_purity_sample  = np.einsum("ijkl->jkl", All_purity_sample ) / sample_size
else_purity_sample = np.einsum("ijkl->jkl", else_purity_sample) / sample_size

All_compleatness_sample  = np.einsum("ijkl->jkl", All_compleatness_sample ) / sample_size
else_compleatness_sample = np.einsum("ijkl->jkl", else_compleatness_sample) / sample_size

# Set nan values to 0
All_purity [np.isnan(All_purity)]  = 0
else_purity[np.isnan(else_purity)] = 0

All_compleatness [np.isnan(All_compleatness)]  = 0
else_compleatness[np.isnan(else_compleatness)] = 0

All_purity_cumulative [np.isnan(All_purity_cumulative)]  = 0
else_purity_cumulative[np.isnan(else_purity_cumulative)] = 0

All_compleatness_cumulative [np.isnan(All_compleatness_cumulative)]  = 0
else_compleatness_cumulative[np.isnan(else_compleatness_cumulative)] = 0

All_purity_sample [np.isnan(All_purity_sample)]  = 0
else_purity_sample[np.isnan(else_purity_sample)] = 0

All_compleatness_sample [np.isnan(All_compleatness_sample)]  = 0
else_compleatness_sample[np.isnan(else_compleatness_sample)] = 0

# -- calculate cuts needed to achive close to target count -------------------------------------------------------------------

Has_target_count = np.where((Target_Count - Target_Count_Error < All_bin_counts_cumulative[0]) & (All_bin_counts_cumulative[0] < Target_Count + Target_Count_Error))

# Get cumulative cuts
cumulative_EXT_cuts = EXT_bin_edges[Has_target_count[0]]
cumulative_EXT_LIKE_cuts = EXT_LIKE_bin_edges[Has_target_count[1]]

# Get properties of samples for cumulative cuts
cumulative_purities     = All_purity_cumulative      [0, Has_target_count[0], Has_target_count[1]]
cumulative_compleatness = All_compleatness_cumulative[0, Has_target_count[0], Has_target_count[1]]
cumulative_count        = All_bin_counts_cumulative  [0, Has_target_count[0], Has_target_count[1]]

print(" : EXT cut, EXT_LIKE cut, Purity, Compleatness, Count")

for i in range(len(Has_target_count[0])):
    print("{}: {}, {}, {}, {}, {}".format(i, cumulative_EXT_cuts[i], cumulative_EXT_LIKE_cuts[i], cumulative_purities[i], cumulative_compleatness[i], cumulative_count[i]))

max_purity = np.argmax(cumulative_purities)

print("\nhighest purity:\n EXT > {}\n EXT_LIKE > {}\n Purity = {}\n Compleatness = {}\n Count = {}\n".format(cumulative_EXT_cuts[max_purity], cumulative_EXT_LIKE_cuts[max_purity], cumulative_purities[max_purity], cumulative_compleatness[max_purity], cumulative_count[max_purity]))

print(Has_target_count)
print(max_purity)
print(Has_target_count[0][max_purity])
print(Has_target_count[1][max_purity])

minus_one_std_percentile = np.percentile(All_purity_cumulative_rand_samples[:, 0, Has_target_count[0][max_purity], Has_target_count[1][max_purity]], 15.9)
plus_one_std_percentile  = np.percentile(All_purity_cumulative_rand_samples[:, 0, Has_target_count[0][max_purity], Has_target_count[1][max_purity]], 84.1)

print(" {} +{} -{}".format(cumulative_purities[max_purity],  plus_one_std_percentile - cumulative_purities[max_purity], cumulative_purities[max_purity] - minus_one_std_percentile))

exit()

# -- Plot purity of sample for EXT, EXT_LIKE cuts for all objects -------------------------------------------------------------

# where there are no sources in the EXT EXT_LIKE bins (not considering confidence) set array entry to nan
All_purity[:, All_Contains_no_source] = np.nan
All_purity_cumulative[:, All_Contains_no_source_cumulative] = np.nan
#All_purity_sample    [:, All_Contains_no_source_cumulative] = np.nan

EXT_tick_values      = 10**np.linspace(-4,3,8)
EXT_LIKE_tick_values = 10**np.linspace(-4,5,10)

EXT_tick_positions      = (len(EXT_bin_edges)-2)      * (np.log10(EXT_tick_values)      - np.log10(EXT_bin_edges[0]))      / (np.log10(EXT_bin_edges[-1])      - np.log10(EXT_bin_edges[0]))
EXT_LIKE_tick_positions = (len(EXT_LIKE_bin_edges)-2) * (np.log10(EXT_LIKE_tick_values) - np.log10(EXT_LIKE_bin_edges[0])) / (np.log10(EXT_LIKE_bin_edges[-1]) - np.log10(EXT_LIKE_bin_edges[0]))

# Calculate the position of each source relative to the image
North_Data['EXT_image']      = (len(EXT_bin_edges)-2)      * (np.log10(North_Data['EXT'])      - np.log10(EXT_bin_edges[0]))      / (np.log10(EXT_bin_edges[-1])      - np.log10(EXT_bin_edges[0]))
North_Data['EXT_LIKE_image'] = (len(EXT_LIKE_bin_edges)-2) * (np.log10(North_Data['EXT_LIKE']) - np.log10(EXT_LIKE_bin_edges[0])) / (np.log10(EXT_LIKE_bin_edges[-1]) - np.log10(EXT_LIKE_bin_edges[0]))

# Calculate the position of the C1C2 cuts relative to the image
EXT_cut_image         = (len(EXT_bin_edges)-2)      * (np.log10(EXT_cut)         - np.log10(EXT_bin_edges[0]))      / (np.log10(EXT_bin_edges[-1])      - np.log10(EXT_bin_edges[0]))
C1_EXT_LIKE_cut_image = (len(EXT_LIKE_bin_edges)-2) * (np.log10(C1_EXT_LIKE_cut) - np.log10(EXT_LIKE_bin_edges[0])) / (np.log10(EXT_LIKE_bin_edges[-1]) - np.log10(EXT_LIKE_bin_edges[0]))
C2_EXT_LIKE_cut_image = (len(EXT_LIKE_bin_edges)-2) * (np.log10(C2_EXT_LIKE_cut) - np.log10(EXT_LIKE_bin_edges[0])) / (np.log10(EXT_LIKE_bin_edges[-1]) - np.log10(EXT_LIKE_bin_edges[0]))

# Calculate the position of the cumulative cuts with target purity
cumulative_EXT_cuts_image      = (len(EXT_bin_edges)-2)      * (np.log10(cumulative_EXT_cuts)      - np.log10(EXT_bin_edges[0]))      / (np.log10(EXT_bin_edges[-1])      - np.log10(EXT_bin_edges[0]))
cumulative_EXT_LIKE_cuts_image = (len(EXT_LIKE_bin_edges)-2) * (np.log10(cumulative_EXT_LIKE_cuts) - np.log10(EXT_LIKE_bin_edges[0])) / (np.log10(EXT_LIKE_bin_edges[-1]) - np.log10(EXT_LIKE_bin_edges[0]))


def Ploting_conf_bin(histogram, title):
    
    # plot purity of each bin
    for i in range(1,len(histogram)+1):
        plt.imshow(histogram[-i].T, origin = 'lower', vmin = 0, vmax = 1)
        
        tmp_selected_sources   = North_Data[          (Conf_bin_edges[-i-1] < North_Data['Conf mean']) * (North_Data['Conf mean'] < Conf_bin_edges[-i]) ]
        tmp_unselected_sources = North_Data[np.invert((Conf_bin_edges[-i-1] < North_Data['Conf mean']) * (North_Data['Conf mean'] < Conf_bin_edges[-i]))]
        plt.scatter( tmp_unselected_sources['EXT_image'], tmp_unselected_sources['EXT_LIKE_image'], s = 10, c = 'black', alpha = 0.2, edgecolors='none')
        plt.scatter( tmp_selected_sources['EXT_image']  , tmp_selected_sources['EXT_LIKE_image']  , s = 20, c = 'white', alpha = 0.4, edgecolors='none')
        
        # Plot EXT cut
        plt.plot([EXT_cut_image, EXT_cut_image],[0,len(EXT_LIKE_bin_edges)-2], c = 'grey', linestyle = '--')
        
        # Plot EXT_LIKE cuts
        plt.plot([0, len(EXT_bin_edges)-2],[C1_EXT_LIKE_cut_image, C1_EXT_LIKE_cut_image], c = 'grey', linestyle = '--')
        plt.plot([0, len(EXT_bin_edges)-2],[C2_EXT_LIKE_cut_image, C2_EXT_LIKE_cut_image], c = 'grey', linestyle = '--')
        
        # set ticks
        plt.xticks(EXT_tick_positions, EXT_tick_values)
        plt.yticks(EXT_LIKE_tick_positions, EXT_LIKE_tick_values)
        
        # set range
        plt.xlim(0,len(EXT_bin_edges)-2)
        plt.ylim(0,len(EXT_LIKE_bin_edges)-2)
        
        #set title
        plt.title(title.format(Conf_bin_edges[-i-1], Conf_bin_edges[-i]))
        
        # formating
        plt.gca().set_aspect('auto')
        plt.colorbar()
        plt.show()

def Ploting_conf_cut(histogram, title, target_purity_cuts = None):
    
    # plot purity of each bin
    for i in range(1,len(histogram)+1):
        plt.imshow(histogram[-i].T, origin = 'lower', vmin = 0, vmax = 1)
        
        tmp_selected_sources   = North_Data[Conf_bin_edges[-i-1] < North_Data['Conf mean']]
        tmp_unselected_sources = North_Data[Conf_bin_edges[-i-1] > North_Data['Conf mean']]
        plt.scatter( tmp_unselected_sources['EXT_image'], tmp_unselected_sources['EXT_LIKE_image'], s = 10, c = 'black', alpha = 0.2, edgecolors='none')
        plt.scatter( tmp_selected_sources['EXT_image']  , tmp_selected_sources['EXT_LIKE_image']  , s = 20, c = 'white', alpha = 0.4, edgecolors='none')
        
        # Plot EXT cut
        plt.plot([EXT_cut_image, EXT_cut_image],[0,len(EXT_LIKE_bin_edges)-2], c = 'grey', linestyle = '--')
        
        # Plot EXT_LIKE cuts
        plt.plot([0, len(EXT_bin_edges)-2],[C1_EXT_LIKE_cut_image, C1_EXT_LIKE_cut_image], c = 'grey', linestyle = '--')
        plt.plot([0, len(EXT_bin_edges)-2],[C2_EXT_LIKE_cut_image, C2_EXT_LIKE_cut_image], c = 'grey', linestyle = '--')
        
        # Plot cuts for targeted purity
        if(target_purity_cuts is not None):
            plt.scatter(target_purity_cuts[0], target_purity_cuts[1], c = 'red')
        
        # set ticks
        plt.xticks(EXT_tick_positions, EXT_tick_values)
        plt.yticks(EXT_LIKE_tick_positions, EXT_LIKE_tick_values)
        
        # set range
        plt.xlim(0,len(EXT_bin_edges)-2)
        plt.ylim(0,len(EXT_LIKE_bin_edges)-2)
        
        #set title
        plt.title(title.format(Conf_bin_edges[-i-1]))
        
        #set lables
        plt.xlabel("EXT")
        plt.ylabel("EXT_LIKE")
        
        # formating
        plt.gca().set_aspect('auto')
        plt.colorbar(label = "Purity")
        plt.show()

Ploting_conf_bin(All_purity, 'bin purity, conf bin: {:.2f} -> {:.2f}')
Ploting_conf_cut(All_purity_cumulative, 'Cumulative purity, confidence cut: {:.2f}', [cumulative_EXT_cuts_image, cumulative_EXT_LIKE_cuts_image])
Ploting_conf_cut(All_purity_sample, 'EXT or EXT_LIKE cut, confidence cut: {:.2f}')




















