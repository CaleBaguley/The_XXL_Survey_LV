import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FixedLocator
import copy as cp
import scipy.stats as stats
from matplotlib import cm

length_mean_col = 'STD based mean length scale'

# --- Data setup -------------------------------------------------------------------------------------------------

feild = 'South'

if(feild == 'North'):
    # North data
    Data = pd.read_csv("../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_03.csv")
    DataConf = pd.read_csv("Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv")
    #DataConf = pd.read_csv("/home/jb14389/Documents/XMM/GPy/Results/V4.3 Data/Data_runs/Sample_opposing_field/BG_Rate/South/10Iters_ARD43s_North_Sample_Results.csv")
    
    output_file = "Corner_plot_N.png"
    
    # manualy set ranges
    ranges = [[9e-5, 18470], # EXT_LIKE
          [6e-4, 1850 ], # EXT
          [2e-8, 5    ], # EXT_RATE_PN
          [2e-8, 1    ], # PNT_RATE_PN
          [2e-8, 0.5  ], # EXT_RATE_MOS
          [2e-8, 0.5  ], # EXT_RATE_MOS
          # north
          [0.9e-7, 5e-5 ]] # EXT_BG_RATE_PN
          
elif(feild == 'South'):
          
    # South data
    Data = pd.read_csv("../../../../Data/PipelineV4.3/XXLs_Reduced_Cols_03.csv")
    DataConf = pd.read_csv("Sample_opposing_field/BG_Rate/South/10Iters_ARD43s_Training_Results.csv")
    #DataConf = pd.read_csv("/home/jb14389/Documents/XMM/GPy/Results/V4.3 Data/Data_runs/Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_South_Sample_Results.csv")
    
    output_file = "Corner_plot_S.png"
    
    # manualy set ranges
    ranges = [[9e-5, 18470], # EXT_LIKE
          [6e-4, 1850 ], # EXT
          [2e-8, 5    ], # EXT_RATE_PN
          [2e-8, 1    ], # PNT_RATE_PN
          [2e-8, 0.5  ], # EXT_RATE_MOS
          [2e-8, 0.5  ], # EXT_RATE_MOS
          # south
          [0.9e-6, 2e-4 ]] # EXT_BG_RATE_MOS
          
else:
    print(f'ERROR: invalid field {field}.')
    exit()

# Colls to plot in order
cols = ['Index', 'EXT_LIKE', 'EXT', 'EXT_RATE_PN', 'PNT_RATE_PN', 'EXT_RATE_MOS', 'PNT_RATE_MOS', 'EXT_BG_RATE_PN']
cols_lables = ['EXT_STAT', 'EXT\n(arcseconds)', 'EXT_RATE_PN\n(counts/s)', 'PNT_RATE_PN\n(counts/s)', 'EXT_RATE_MOS\n(counts/s)', 'PNT_RATE_MOS\n(counts/s)', 'EXT_BG_RATE_PN\n(counts/s)']




Data = Data[cols]

# Reduce DataConf to id, mean and std
DataConf = DataConf[['id', 'Conf mean', 'Conf std']]

# Match data sets
Data = Data.merge(DataConf, left_on='Index', right_on='id')

# drop label col from list
cols = cols[1:]

# order based on confidence
Data = Data.sort_values('Conf mean')

# Floor extent like values to 10^-4
Data["EXT_LIKE"][Data["EXT_LIKE"] < 1.5*10**-4] = 10**-4

# add sample labels
corerad_cut = 5
extlike_C1_cut = 33
extlike_C2_cut = 15

# --- Corner plot confidence ------------------------------------------------------------------------------------------------

major_ticks = FixedLocator(10**np.linspace(-10,10,21)) 

print(major_ticks)

p_size = 5

fig, axs = plt.subplots(len(cols), len(cols))
fig.set_figheight(7.3)
fig.set_figwidth(11.7)

# Make top right plots blank
for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        axs[i,j].axis('off')
        
# Automaticaly set ranges for each column
#ranges = [[max(0.9*min(Data[cols[i]]),10**(-8)), 1.1*max(Data[cols[i]])] for i in range(len(cols))]




for i in range(len(ranges)):
    print(cols[i], ":", ranges[i])

# Iterate over bottom left off axis scatter plots
for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        # Plot scatter
        axs[j,i].scatter(Data[cols[i]], Data[cols[j]], c = Data['Conf mean'], vmin=0, vmax=1, s = p_size, alpha = 1, cmap=cm.viridis)
        
        # Log axis
        axs[j,i].set_xscale('log')
        axs[j,i].set_yscale('log')
        
        # Set axis range
        axs[j,i].set_ylim(ranges[j][0], ranges[j][1])
        axs[j,i].set_xlim(ranges[i][0], ranges[i][1])
        
        # Remove tick labels
        axs[j,i].tick_params(axis='both', which='both', labelleft=False, left=True, labelbottom=False, bottom=True)
        
        # Set major tick values
        axs[j,i].xaxis.set_major_locator(major_ticks)
        axs[j,i].yaxis.set_major_locator(major_ticks)

# Plot diagonals and eddit axis labels and ticks
for i in range(len(cols)):
    
    # Plot scatter
    axs[i,i].scatter(Data[cols[i]], Data['Conf mean'], s = p_size, alpha = 1)
    
    # Calcualte average by bin
    bins = 10**np.linspace(np.log10(ranges[i][0]),np.log10(ranges[i][1]), num = 21)
    
    count_per_bin, bin_edges = np.histogram(Data[cols[i]], bins = bins)
    conf_sum_per_bin, bin_edges = np.histogram(Data[cols[i]], bins = bins, weights = Data['Conf mean'])
    
    mean_conf_per_bin = conf_sum_per_bin/count_per_bin
        
    axs[i,i].plot(bins[:-1] + (bins[1:]-bins[:-1])/2, mean_conf_per_bin, c = 'black')
    
    # Log x axis
    axs[i,i].set_xscale('log')
    
    # Set range
    axs[i,i].set_xlim(ranges[i][0], ranges[i][1])
    axs[i,i].set_ylim(-0.07, 1)

    # Label bottom edge plots
    if(i%2 == 0):
        tmp = cols_lables[i]
    else:
        tmp = "\n\n" + cols_lables[i]
    axs[-1,i].set_xlabel(tmp, size = 11, fontweight = 'bold')
        
    axs[-1,i].tick_params(axis='both', which='both', labelleft=False, labelbottom=True)
    plt.setp(axs[-1,i].get_xticklabels()[::3], visible=False)
    plt.setp(axs[-1,i].get_xticklabels()[::2], visible=False)
    
    # Label left edge plots
    if(i != 0):
        axs[i,0].set_ylabel(cols_lables[i], size = 11, fontweight = 'bold', rotation = 0, horizontalalignment = 'right')
    
    axs[i,0].tick_params(axis='both', which='both', labelleft=True, labelbottom=False)
    plt.setp(axs[i,0].get_yticklabels()[::3], visible=False)
    plt.setp(axs[i,0].get_yticklabels()[::2], visible=False)
    
    # Set ticks
    axs[i,i].set_ylabel('Confidence\nvalue', size = 9, fontweight = 'bold')
    axs[i,i].yaxis.set_label_position("right")
    axs[i,i].tick_params(axis='both', which='both', left = False, labelleft=False, right = True, labelright = True, labelbottom=False)
    
    # Set major tick values
    axs[i,i].xaxis.set_major_locator(major_ticks)

#fix bottom left
axs[-1, 0].tick_params(axis='both', which='both', left = True, labelleft=True, bottom=True, labelbottom=True)
plt.setp(axs[-1,0].get_xticklabels()[::3], visible=False)
plt.setp(axs[-1,0].get_xticklabels()[::2], visible=False)
#plt.setp(axs[-1,0].get_yticklabels()[::3], visible=False)
#plt.setp(axs[-1,0].get_yticklabels()[::2], visible=False)

#fix bottom right
axs[-1,-1].tick_params(axis='both', which='both', left = False, labelleft=False, right = True, labelright = True, bottom = True, labelbottom=True)
#plt.setp(axs[-1,-1].get_xticklabels()[::3], visible=False)
#plt.setp(axs[-1,-1].get_xticklabels()[::2], visible=False)
    
# Remove space between plots
fig.subplots_adjust(hspace = 0, wspace = 0)


# -- colour bar --
im = plt.gca().get_children()[0]
cax = fig.add_axes([0.93,0.1,0.01,0.8]) 
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Confidence value', fontsize=20)

# Adjust padding
plt.subplots_adjust(left=0.17, right=0.87, top=0.99, bottom=0.15)

plt.savefig(output_file)
plt.show()

