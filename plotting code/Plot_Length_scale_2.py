import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy as cp
import scipy.stats as stats
from matplotlib import cm


source_file_address_N = "Sample_opposing_field/BG_Rate/North/Length_Scales.csv"
sources_file_address_N = "../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_Pointings_Cleaned_03.csv"
confidence_file_address_N = "Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv"


source_file_address_S = "Sample_opposing_field/BG_Rate/South/Length_Scales.csv"
sources_file_address_S = "../../../../Data/PipelineV4.3/XXLs_Reduced_Cols_03.csv"
confidence_file_address_S = "Sample_opposing_field/BG_Rate/South/10Iters_ARD43s_Training_Results.csv"

training_field = 'North'

# --- Data setup -------------------------------------------------------------------------------------------------

def Setup_Length_Scale(source_file_address, sources_file_address, confidence_file_address):
    Length_Scale = pd.read_csv(source_file_address)

    Data = pd.read_csv(sources_file_address)
    conf_values = pd.read_csv(confidence_file_address)['id']

    Data = pd.merge(Data, conf_values, left_on = 'Index', right_on = 'id')

    try:
        col_names = Length_Scale['Name']
        
    except:
        All_Cols = Data.keys()

        col_id = Length_Scale['Col id']
        
        col_names = All_Cols[col_id]
        
        Length_Scale['Name'] = col_names
        
        Length_Scale.to_csv(source_file_address)

    Length_Scale = Length_Scale[['Col id', 'Run 0', 'Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8', 'Run 9', 'Mean length scale', 'Std length scale', 'Name']]

    # Sort data smallest to largest
    Length_Scale.sort_values(['Mean length scale'], ascending=False, inplace=True)

    # Change index to new order
    Length_Scale.reset_index(drop = True, inplace = True)
    
    return Length_Scale, Data

Length_Scale_N, Data_N = Setup_Length_Scale(source_file_address_N, sources_file_address_N, confidence_file_address_N)
Length_Scale_S, Data_S = Setup_Length_Scale(source_file_address_S, sources_file_address_S, confidence_file_address_S)

# --- Length scales ----------------------------------------------------------------------------------------------

def plot_length_scales(Length_Scale, training_field, length_scale_col = 'Mean length scale', error_col = 'Std length scale', x_label = 'Length scale over normalised data', xlim = (0,5.5), size = (8,7)):
    TextSize = 16
    Marker_Size = 100

    plt.rcParams.update({'font.size': TextSize})
    fig, ax = plt.subplots()
    ax.barh(Length_Scale['Name'],
            np.zeros(len(Length_Scale)),
            left = Length_Scale[length_scale_col],
            xerr=Length_Scale[error_col],
            log = False, color = "black")
            
    ax.scatter(Length_Scale[length_scale_col], [i for i in range(len(Length_Scale))], s = Marker_Size, color = 'black')

    # Add values to each column
    for i, current in Length_Scale.iterrows():
        #print(i, current)
        ax.text(current[length_scale_col] + current[error_col] + 0.02, i - 0.25 , "{:.2f}".format(current[length_scale_col]), fontsize=TextSize)
       
    plt.title('Parameter length scales {} trained'.format(training_field), fontsize=30)
    plt.xlabel(x_label)
    plt.xlim(xlim[0], xlim[1])
    
    plt.gcf().set_size_inches(size[0], size[1])
    
    plt.show()
    
#plot_length_scales(Length_Scale_N, 'North')
#plot_length_scales(Length_Scale_S, 'South')

# --- Length scales scaled to data range ------------------------------------------------------------------------

def Scale_to_data_range(Length_Scale, Data):
    Rescaled_length_scales = cp.deepcopy(Length_Scale['Mean length scale'].to_numpy())
    Rescaled_length_scales_std = cp.deepcopy(Length_Scale['Std length scale'].to_numpy())

    Sub_Data = Data.iloc[:, Length_Scale['Col id']]

    Sub_Data = Sub_Data[(Sub_Data > 0).all(1)]

    Col_Range = (Sub_Data.max(0) - Sub_Data.min(0)).to_numpy()

    Rescaled_length_scales *= Col_Range
    Rescaled_length_scales_std *= Col_Range

    Length_Scale['min val'] = Sub_Data.min(0)
    Length_Scale['max val'] = Sub_Data.min(0)
    Length_Scale['col range'] = Col_Range
    Length_Scale['Rescaled mean length scale'] = Rescaled_length_scales
    Length_Scale['Rescaled std length scale'] = Rescaled_length_scales_std
    
    return Length_Scale

Length_Scale_N = Scale_to_data_range(Length_Scale_N, Data_N)
Length_Scale_S = Scale_to_data_range(Length_Scale_S, Data_S)

Length_Scale_N.to_csv(source_file_address_N)
Length_Scale_S.to_csv(source_file_address_S)

# --- Length scales scaled to STD of data ------------------------------------------------------------------------

def Scale_to_std_in_data(Length_Scale, Data):
    SD_based_length_scales = cp.deepcopy(Length_Scale['Rescaled mean length scale'].to_numpy())
    SD_based_length_scales_std = cp.deepcopy(Length_Scale['Rescaled std length scale'].to_numpy())

    Sub_Data = Data.iloc[:, Length_Scale['Col id']]

    Sub_Data = Sub_Data[(Sub_Data > 0).all(1)]

    Col_std = (Sub_Data.std(0)).to_numpy()

    SD_based_length_scales /= Col_std
    SD_based_length_scales_std /= Col_std

    Length_Scale['col std'] = Col_std
    Length_Scale['STD based mean length scale'] = SD_based_length_scales
    Length_Scale['STD based std length scale'] = SD_based_length_scales_std

    # -- calcualte error on col std by bootstraping --
    samples = 10000
    std_error = np.zeros(len(Col_std))

    for i in range(samples):
        # Get random sample from sources
        sample = Sub_Data.sample(frac = 1, replace = True)
        
        # calculate std of sample
        sample_col_std = (sample.std(0)).to_numpy()
        
        # add error
        std_error += np.square(Col_std - sample_col_std)

    # Normalise and square root
    std_error = np.sqrt(std_error/samples)

    # Add to data frame
    Length_Scale['col std error'] = std_error
    
    # Re-sort data smallest to largest STD based length scale
    Length_Scale.sort_values(['STD based mean length scale'], ascending=False, inplace=True)

    # Change index to new order
    Length_Scale.reset_index(drop = True, inplace = True)
    
    return Length_Scale

Length_Scale_N = Scale_to_std_in_data(Length_Scale_N, Data_N)
Length_Scale_S = Scale_to_std_in_data(Length_Scale_S, Data_S)

Length_Scale_N.to_csv(source_file_address_N)
Length_Scale_S.to_csv(source_file_address_S)

# -- Plot length scale results ------------------------------------------------------------------------------------

#plot_length_scales(Length_Scale_N, 'North', length_scale_col = 'STD based mean length scale', error_col = 'STD based std length scale', x_label = 'Length scale over normalised data', xlim = (0,150), size = (8,7))
#plot_length_scales(Length_Scale_S, 'South', length_scale_col = 'STD based mean length scale', error_col = 'STD based std length scale', x_label = 'Length scale over normalised data', xlim = (0,150), size = (8,7))

# -- plot both sharing the x-axis --



TextSize = 14
Marker_Size = 50

plt.rcParams.update({'font.size': TextSize})

fig, (ax_N, ax_S) = plt.subplots(2,1, sharex = True)
fig.subplots_adjust(hspace=0)

# plot bars
ax_N.barh(Length_Scale_N['Name'],
          np.zeros(len(Length_Scale_N)),
          left = Length_Scale_N['STD based mean length scale'],
          xerr = np.sqrt(np.square(Length_Scale_N['STD based std length scale']) + np.square(Length_Scale_N['col std error'])),
          log = False, color = "black")
ax_S.barh(Length_Scale_S['Name'],
          np.zeros(len(Length_Scale_S)),
          left = Length_Scale_S['STD based mean length scale'],
          xerr = np.sqrt(np.square(Length_Scale_S['STD based std length scale']) + np.square(Length_Scale_S['col std error'])),
          log = False, color = "black")

# add points
ax_N.scatter(Length_Scale_N['STD based mean length scale'], [i for i in range(len(Length_Scale_N))], s = Marker_Size, color = 'black')
ax_S.scatter(Length_Scale_S['STD based mean length scale'], [i for i in range(len(Length_Scale_S))], s = Marker_Size, color = 'black')

#ax.scatter(Length_Scale['col range']/Length_Scale['col std'], [i for i in range(len(Length_Scale))], s = Marker_Size, color = 'grey', marker = '|')

# Add values to each column for North
for i, current in Length_Scale_N.iterrows():
    ax_N.text(current['STD based mean length scale'] + current['STD based std length scale'] + 1, i - 0.4 , "{:.0f}".format(current['STD based mean length scale']), fontsize=TextSize)

# Add values to each column for South
for i, current in Length_Scale_S.iterrows():
    ax_S.text(current['STD based mean length scale'] + current['STD based std length scale'] + 1, i - 0.4 , "{:.0f}".format(current['STD based mean length scale']), fontsize=TextSize)
   
# lable x-axis
plt.xlabel('Length scale over\nnormalised data', size = 'x-large')
plt.xlim(0,125)

# lable plots
ax_N.text(60, len(Length_Scale_N) - 2.2, '  GP trained on\nNorth catalogue', fontsize = 16)
ax_S.text(60, len(Length_Scale_S) - 2.2, '  GP trained on\nSouth catalogue', fontsize = 16)

# set plot size
plt.gcf().set_size_inches(7,15)

# Adjust padding
plt.subplots_adjust(left=0.4, right=0.9, top=0.95, bottom=0.10)

plt.show()


















