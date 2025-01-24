import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy as cp
import scipy.stats as stats
from matplotlib import cm

# --- Data setup -------------------------------------------------------------------------------------------------

# load relevent data
Data = pd.read_csv("../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_Pointings_Cleaned_03.csv")
DataConf = pd.read_csv("../Data_runs/Sample simdata/BG_Rate/North_cleaned/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv")
Camira_matched = pd.read_csv("../../../../Data/PipelineV4.3/Camira_Matched_XXLn_Redused_Cols_Pointings_Cleaned_03.csv")

# Reduce DataConf to id, mean and std
DataConf = DataConf[['id', 'Conf mean', 'Conf std']]

# Match data sets
Data = Data.merge(DataConf, left_on='Index', right_on='id')
Camira_objects = Camira_matched.merge(DataConf, left_on='Index', right_on='id')

# order based on confidence
Data = Data.sort_values('Conf mean', ignore_index = 'True')

C1_objects = Data.loc[Data['C1C2'] == 1]
C2_objects = Data.loc[Data['C1C2'] == 2]
Non_C1C2_objects = Data.loc[Data['C1C2'] == 0]

Camera_C1_objects = Camira_objects.loc[Camira_matched['C1C2'] == 1]
Camera_C2_objects = Camira_objects.loc[Camira_matched['C1C2'] == 2]
Camera_Non_C1C2_objects = Camira_objects.loc[Camira_matched['C1C2'] == 0]

# ----- List creation ---------------------------------------------------------------------------------------------------------

lotto_subsets = []

# -- get all Non C1C2 objects with a confidence >= 0.3 --
#lotto_subsets.append(Non_C1C2_objects.loc[0.3 <= Non_C1C2_objects['Conf mean']])

# -- randomly sample from confidence bins --
bins = [[0.05, 0.10, 100],
        [0.00, 0.05, 100]]

# counts unused samples from previous bins that can be used for next bin in list
rollover = 0

for current_bin in bins:
    
    # select objects in sample range
    current_sample = Non_C1C2_objects.loc[current_bin[0] < Non_C1C2_objects['Conf mean']] # apply minimum cut
    current_sample = current_sample.loc[Non_C1C2_objects['Conf mean'] < current_bin[1]] # apply maximum cut
    
    # get number of objects that can be sampled
    sample_size = min(len(current_sample), current_bin[2] + rollover)
    
    # calcualte number of sample in hand for next bin
    rollover += current_bin[2] - sample_size
    
    current_sample = current_sample.sample(n = sample_size)
    
    
    
    lotto_subsets.append(current_sample)

# --- camera objects ---
num_camera_objects = 50
num_camera_objects = min(len(Camera_Non_C1C2_objects), num_camera_objects)

Camera_objects_lotto = Camera_Non_C1C2_objects.sample(n = num_camera_objects)

#lotto_subsets.append(Camera_objects_lotto)

print("Camera objects; ", len(Camera_objects_lotto))


# ----- Save lotto objects ----------------------------------------------------------------------------------------------------

# combine data sets and shuffle 
Combined_lotto_objects = pd.concat(lotto_subsets)
Combined_lotto_objects = Combined_lotto_objects.sample(frac = 1)

# create data sets to save
Combined_lotto_objects = Combined_lotto_objects[['Index', 'EXT_RA', 'EXT_DEC']]
Camera_objects_lotto   = Camera_objects_lotto  [['Index', 'EXT_RA', 'EXT_DEC']]

# save full lotto to file
Combined_lotto_objects.to_csv("V4.3_Lotto_run/Low_conf_lotto_Sources.csv", index = False)

# save subsets for later
#Camera_objects_lotto.to_csv("V4.3_Lotto_run/Data subsets/Camera_objects_lotto.csv", index = False)


















