#from statistics import correlation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import copy as cp
import scipy.stats as stats
from matplotlib import cm

# --- plot paramiters --------------------------------------------------------------------------------------------

# -- EXT vs EXT_LIKE plots --

# C1C2 sample cuts
EXT_cut = 5
EXT_LIKE_C1_cut = 33
EXT_LIKE_C2_cut = 15

Sample_line_width = 1

# source property plotting limits
EXT_lim          = (0.8*10**(-3), 2.0*10**3)
EXT_LIKE_lim     = (0.8*10**(-4), 2.0*10**4)
EXT_RATE_PN_lim  = (0.5*10**(-8), 1.0*10**1)
PNT_RATE_PN_lim  = (    10**(-8), 0.5*10**1)
EXT_RATE_MOS_lim = (0.5*10**(-8), 3.0*10**0)
PNT_RATE_MOS_lim = (    10**(-8), 0.5*10**1)

# confidence range
conf_lim = (0, 1)

# -- Confidence histogram --

# confidence bins
conf_bin_edges = [0.05*i for i in range(0,21)]
conf_hist_lim = (0,0.87)

C1_hist_max = 30
C2_hist_max = 30

# count ticks
y_ticks = [0,10,20]
y_ticks_minor = [2*i for i in range(15)]

# -- Confidence histogram logged --

C1C2_hist_range_log = (0.9, 50)
Non_C1C2_hist_range_log = (0.9, 40000)

hist_figure_ratio = np.log10(Non_C1C2_hist_range_log[1]/Non_C1C2_hist_range_log[0]) / np.log10(C1C2_hist_range_log[1]/C1C2_hist_range_log[0])

# count ticks
y_ticks_log = [10**i for i in range(0,5)]
y_ticks_minor_log = []

for i in range(0,5):
    for j in range(1,10):
        y_ticks_minor_log.append(j*(10**i))
        
# -- Standard deviation as a function of confidence --
conf_standard_deviation_bin_edges = np.linspace(0, 1, 21)
conf_standard_deviation_bin_centers = (conf_standard_deviation_bin_edges[:-1] + conf_standard_deviation_bin_edges[1:])/2

conf_standard_deviation_conf_lim = (0, 0.89)
conf_standard_deviation_std_lim  = (0, 0.017)

# --- Data setup -------------------------------------------------------------------------------------------------

# load in north
Data_N = pd.read_csv("../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_03.csv")
DataConf_N = pd.read_csv("Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv")
DataConf_N_south_trained = pd.read_csv("Sample_opposing_field/BG_Rate/South/10Iters_ARD43s_North_Sample_Results.csv")
Camira_matched = pd.read_csv("Camira_matched.csv")

# load in south
Data_S = pd.read_csv("../../../../Data/PipelineV4.3/XXLs_Reduced_Cols_03.csv")
DataConf_S = pd.read_csv("Sample_opposing_field/BG_Rate/South/10Iters_ARD43s_Training_Results.csv")
DataConf_S_north_trained = pd.read_csv("Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_South_Sample_Results.csv")

# Reduce DataConf to id, mean and std
DataConf_N = DataConf_N[['id', 'Conf mean', 'Conf std']]
DataConf_S = DataConf_S[['id', 'Conf mean', 'Conf std']]

DataConf_N_south_trained = DataConf_N_south_trained[['id', 'Conf mean', 'Conf std']]
DataConf_S_north_trained = DataConf_S_north_trained[['id', 'Conf mean', 'Conf std']]

DataConf_N_south_trained = DataConf_N_south_trained.rename(columns={'Conf mean' : 'Conf mean south trained', 'Conf std' : 'Conf std south trained'})
DataConf_S_north_trained = DataConf_S_north_trained.rename(columns={'Conf mean' : 'Conf mean north trained', 'Conf std' : 'Conf std north trained'})

# Match data sets
Data_N = Data_N.merge(DataConf_N, left_on='Index', right_on='id')
Data_S = Data_S.merge(DataConf_S, left_on='Index', right_on='id')

Data_N = Data_N.merge(DataConf_N_south_trained, left_on='Index', right_on='id')
Data_S = Data_S.merge(DataConf_S_north_trained, left_on='Index', right_on='id')

# order based on North trained confidence
Data_N = Data_N.sort_values('Conf mean')
Data_S = Data_S.sort_values('Conf mean north trained')

# reset index to so it follows new order
Data_N = Data_N.reset_index()
Data_S = Data_S.reset_index()

# Set all values of EXT_LIKE below 10^-4 to 10^-4
Data_N['EXT_LIKE'] = Data_N['EXT_LIKE'].apply(lambda x: 10**(-4) if x < 10**(-4) else x)
Data_S['EXT_LIKE'] = Data_S['EXT_LIKE'].apply(lambda x: 10**(-4) if x < 10**(-4) else x)

# Split into C1, C2 and Non C1C2 subsets
Data_N_C1       = Data_N.loc[Data_N['C1C2'] == 1]
Data_N_C2       = Data_N.loc[Data_N['C1C2'] == 2]
Data_N_Non_C1C2 = Data_N.loc[Data_N['C1C2'] == 0]

Data_S_C1       = Data_S.loc[Data_S['C1C2'] == 1]
Data_S_C2       = Data_S.loc[Data_S['C1C2'] == 2]
Data_S_Non_C1C2 = Data_S.loc[Data_S['C1C2'] == 0]

# --- Plot EXT EXT_LIKE distribution -----------------------------------------------------------------------------

def Plot_EXT_vs_EXT_LIKE(C1_data, C2_data, Non_C1C2_data, label = '', colour_by_conf = False, all_data=None):

    fig, axs = plt.subplots()

    # plot C1C2 cuts
    # EXT cut
    axs.plot(EXT_LIKE_lim, (EXT_cut, EXT_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    # EXT_LIKE cuts
    axs.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    axs.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)

    # plot data
    if(colour_by_conf):
        sctr = axs.scatter(Non_C1C2_data['EXT_LIKE'], Non_C1C2_data['EXT'], c = Non_C1C2_data['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
        axs.scatter(C2_data      ['EXT_LIKE'], C2_data      ['EXT'], c = C2_data      ['Conf mean'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
        axs.scatter(C1_data      ['EXT_LIKE'], C1_data      ['EXT'], c = C1_data      ['Conf mean'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
        
        cbar = plt.colorbar(sctr, ticks = [0.1*i for i in range(0,11)])
        cbar.set_label('Confidence Value', size = 'xx-large')
        cbar.ax.tick_params(labelsize = 'x-large')
        
    else:
        axs.scatter(Non_C1C2_data['EXT_LIKE'], Non_C1C2_data['EXT'], alpha = 0.2, edgecolor = 'none', s = 70, marker = 'o', c = 'green' , edgecolors='none', zorder = 1)
        axs.scatter(C2_data      ['EXT_LIKE'], C2_data      ['EXT'], alpha = 0.2, edgecolor = 'none', s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
        axs.scatter(C1_data      ['EXT_LIKE'], C1_data      ['EXT'], alpha = 0.2, edgecolor = 'none', s = 70, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3)

    if all_data is not None:
        x_edges = 10**np.linspace(-5, 5,50)
        print(x_edges)
        x_edges = (x_edges[:-1] + x_edges[1:])/2
        print(x_edges)
        y_edges = 10**np.linspace(np.log(EXT_lim[0]), np.log(EXT_lim[1]), 50)
        hist, xedges, yedges = np.histogram2d(all_data['EXT_LIKE'], all_data['EXT'], bins = [x_edges, y_edges])
        
        min_density = np.min(hist)
        max_density = np.max(hist)
        print(f"min = {min_density}")
        print(f"max = {max_density}")
        contours = [1,10,100,1000]

        CS = axs.contour(x_edges[:-1], y_edges[:-1], hist.T, contours, colors='k')


    # set limits
    axs.set_xlim(EXT_LIKE_lim)
    axs.set_ylim(EXT_lim)

    # log axies
    axs.set_xscale("log")
    axs.set_yscale("log")

    # add axis titles
    axs.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    axs.set_ylabel("EXT (arcseconds)", fontsize = 'xx-large')

    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 16)
    axs.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    axs.xaxis.set_minor_locator(locmin)

    # set size of tick lables
    axs.tick_params(axis='both', which='major', labelsize = 'x-large')

    # set plot size
    fig.set_size_inches(7,5.5)
    
    # Adjust padding
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    
    # Add lables
    #plt.text(5000, 700, 'C1', size = 'xx-large')
    #plt.text(  17, 700, 'C2', size = 'xx-large')
    #plt.text(0.0001, 2200, label, size = 'xx-large')
    
    plt.show()
    
def Plot_EXT_vs_EXT_LIKE_comparison(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2):
    
    fig, ((N_data_N_trained, S_data_N_trained), (N_data_S_trained, S_data_S_trained)) = plt.subplots(2,2, sharex=True, sharey=True)
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    
    # Plot C1C2 cuts
    # EXT cut
    N_data_N_trained.plot(EXT_LIKE_lim, (EXT_cut, EXT_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    S_data_N_trained.plot(EXT_LIKE_lim, (EXT_cut, EXT_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    N_data_S_trained.plot(EXT_LIKE_lim, (EXT_cut, EXT_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    S_data_S_trained.plot(EXT_LIKE_lim, (EXT_cut, EXT_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    
    # EXT LIKE cuts
    N_data_N_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    N_data_N_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    
    S_data_N_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_N_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    
    N_data_S_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    N_data_S_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0) 
    
    S_data_S_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_S_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    # Plot data
    N_data_N_trained.scatter(Data_N_Non_C1C2['EXT_LIKE'], Data_N_Non_C1C2['EXT'], c = Data_N_Non_C1C2['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    N_data_N_trained.scatter(Data_N_C2      ['EXT_LIKE'], Data_N_C2      ['EXT'], c = Data_N_C2      ['Conf mean'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    N_data_N_trained.scatter(Data_N_C1      ['EXT_LIKE'], Data_N_C1      ['EXT'], c = Data_N_C1      ['Conf mean'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    S_data_N_trained.scatter(Data_S_Non_C1C2['EXT_LIKE'], Data_S_Non_C1C2['EXT'], c = Data_S_Non_C1C2['Conf mean north trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(Data_S_C2      ['EXT_LIKE'], Data_S_C2      ['EXT'], c = Data_S_C2      ['Conf mean north trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(Data_S_C1      ['EXT_LIKE'], Data_S_C1      ['EXT'], c = Data_S_C1      ['Conf mean north trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    N_data_S_trained.scatter(Data_N_Non_C1C2['EXT_LIKE'], Data_N_Non_C1C2['EXT'], c = Data_N_Non_C1C2['Conf mean south trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    N_data_S_trained.scatter(Data_N_C2      ['EXT_LIKE'], Data_N_C2      ['EXT'], c = Data_N_C2      ['Conf mean south trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    N_data_S_trained.scatter(Data_N_C1      ['EXT_LIKE'], Data_N_C1      ['EXT'], c = Data_N_C1      ['Conf mean south trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    S_data_S_trained.scatter(Data_S_Non_C1C2['EXT_LIKE'], Data_S_Non_C1C2['EXT'], c = Data_S_Non_C1C2['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(Data_S_C2      ['EXT_LIKE'], Data_S_C2      ['EXT'], c = Data_S_C2      ['Conf mean'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(Data_S_C1      ['EXT_LIKE'], Data_S_C1      ['EXT'], c = Data_S_C1      ['Conf mean'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    # set limits
    N_data_N_trained.set_xlim(EXT_LIKE_lim)
    N_data_N_trained.set_ylim(EXT_lim)
    
    S_data_N_trained.set_xlim(EXT_LIKE_lim)
    S_data_N_trained.set_ylim(EXT_lim)
    
    N_data_S_trained.set_xlim(EXT_LIKE_lim)
    N_data_S_trained.set_ylim(EXT_lim)
    
    S_data_S_trained.set_xlim(EXT_LIKE_lim)
    S_data_S_trained.set_ylim(EXT_lim)
    
    # log axies
    N_data_N_trained.set_xscale("log")
    N_data_N_trained.set_yscale("log")
    
    S_data_N_trained.set_xscale("log")
    S_data_N_trained.set_yscale("log")
    
    N_data_S_trained.set_xscale("log")
    N_data_S_trained.set_yscale("log")
    
    S_data_S_trained.set_xscale("log")
    S_data_S_trained.set_yscale("log")
    
    # Add axis lables
    N_data_S_trained.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    S_data_S_trained.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    
    N_data_N_trained.set_ylabel("GP trained on XXL North\n\nEXT (arcseconds)", fontsize = 'xx-large')
    N_data_S_trained.set_ylabel("GP trained on XXL South\n\nEXT (arcseconds)", fontsize = 'xx-large')
    
    # Title columns
    N_data_N_trained.set_title('North XXL catalogue', fontsize = 'xx-large')
    S_data_N_trained.set_title('South XXL catalogue', fontsize = 'xx-large')
    
    # Remove ticks
    S_data_N_trained.tick_params(which = 'both', left = False)
    S_data_S_trained.tick_params(which = 'both', left = False)
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 16)
    N_data_S_trained.xaxis.set_major_locator(locmaj)
    S_data_S_trained.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.xaxis.set_minor_locator(locmin)
    S_data_S_trained.xaxis.set_minor_locator(locmin)
    
    # add colourbar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax = (S_data_N_trained, S_data_S_trained), shrink = 1, anchor = (0,0))
    
    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')
    
    plt.show()



# Plot initial samples
Plot_EXT_vs_EXT_LIKE(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, label = 'North', colour_by_conf = False, all_data = Data_N)
Plot_EXT_vs_EXT_LIKE(Data_S_C1, Data_S_C2, Data_S_Non_C1C2, label = 'South', colour_by_conf = False)

# Plot confidence distribution
Plot_EXT_vs_EXT_LIKE(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, colour_by_conf = True)
Plot_EXT_vs_EXT_LIKE(Data_S_C1, Data_S_C2, Data_S_Non_C1C2, colour_by_conf = True)

# Plot confidence values for different GPs and catalogues
# Plot_EXT_vs_EXT_LIKE_comparison(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)

# -- Plot confidence value distribution -------------------------------------------------------------------------------

def Plot_Confidence_Histogram_Log(C1_data, C2_data, Non_C1C2_data, label = ''):
    
    # Create subplots
    fig, (C1_axis, C2_axis, Non_C1C2_axis) = plt.subplots(3, sharex = True, gridspec_kw={'height_ratios': [1, 1, hist_figure_ratio]})
    
    # Remove vertical space
    fig.subplots_adjust(hspace=0)
    
    # Plot histograms
    C1_axis.hist(C1_data['Conf mean'], bins = conf_bin_edges, color = 'blue')
    C2_axis.hist(C2_data['Conf mean'], bins = conf_bin_edges, color = 'orange')
    Non_C1C2_axis.hist(Non_C1C2_data['Conf mean'], bins = conf_bin_edges, color = 'green')
    
    # Lable histograms
    C1_axis      .text(0.845, 13, 'C1', size = 'xx-large')
    C2_axis      .text(0.845, 13, 'C2', size = 'xx-large')
    Non_C1C2_axis.text(0.69, 14000, 'Non C1C2', size = 'xx-large')
    
    # Lable plot
    C1_axis      .text(0.01, 13, label, size = 'xx-large')
    
    # -- set y lable stuff --
    
    # log y axis
    C1_axis.set_yscale('log')
    C2_axis.set_yscale('log')
    Non_C1C2_axis.set_yscale('log')
    
    # set y ticks
    C1_axis      .set_yticks(y_ticks_log)
    C1_axis      .set_yticks(y_ticks_minor_log, minor = True)
    C2_axis      .set_yticks(y_ticks_log)
    C2_axis      .set_yticks(y_ticks_minor_log, minor = True)
    Non_C1C2_axis.set_yticks(y_ticks_log)
    Non_C1C2_axis.set_yticks(y_ticks_minor_log, minor = True)
    
    # set y tick size
    C1_axis      .tick_params(axis='y', which='major', labelsize='x-large')
    C2_axis      .tick_params(axis='y', which='major', labelsize='x-large')
    Non_C1C2_axis.tick_params(axis='y', which='major', labelsize='x-large')
    
    # set y range
    C1_axis      .set_ylim(C1C2_hist_range_log)
    C2_axis      .set_ylim(C1C2_hist_range_log)
    Non_C1C2_axis.set_ylim(Non_C1C2_hist_range_log)
    
    # -- Set x lable stuff --
    
    # change confidence value limits
    plt.xlim(conf_hist_lim)
    
    # eddit ticks
    Non_C1C2_axis.set_xticks([0.1*i for i in range(10)])
    Non_C1C2_axis.set_xticks([0.05*i for i in range(19)], minor = True)
    Non_C1C2_axis.tick_params(axis='x', which='major', labelsize='x-large')
    
    # -- general formating --
    plt.xlabel("Confidence value", size = 'xx-large')
    
    # set plot size
    plt.gcf().set_size_inches(7,5.5)
    
    # Adjust padding
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    
    plt.show()
    
def Plot_Confidence_Histogram_Comparison_Log(C1_data_N, C2_data_N, Non_C1C2_data_N, C1_data_S, C2_data_S, Non_C1C2_data_S):
    
    # Create subplots
    fig, axis = plt.subplots(7, 2, sharex = True, gridspec_kw={'height_ratios':[1, 1, hist_figure_ratio, 0.2, 1, 1, hist_figure_ratio]})
    
    # Set axies
    C1_axis_N_data_N_trained       = axis[0,0]
    C2_axis_N_data_N_trained       = axis[1,0]
    Non_C1C2_axis_N_data_N_trained = axis[2,0]
    empty_left                     = axis[3,0]
    C1_axis_N_data_S_trained       = axis[4,0]
    C2_axis_N_data_S_trained       = axis[5,0]
    Non_C1C2_axis_N_data_S_trained = axis[6,0]
    C1_axis_S_data_N_trained       = axis[0,1]
    C2_axis_S_data_N_trained       = axis[1,1]
    Non_C1C2_axis_S_data_N_trained = axis[2,1]
    empty_right                    = axis[3,1]
    C1_axis_S_data_S_trained       = axis[4,1]
    C2_axis_S_data_S_trained       = axis[5,1]
    Non_C1C2_axis_S_data_S_trained = axis[6,1]
    
    #Set empty plots invisible
    empty_left .set_visible(False)
    empty_right.set_visible(False)
    
    # Remove vertical space
    fig.subplots_adjust(hspace=0)
    
    # Set horizontal space
    fig.subplots_adjust(wspace=0.03)
    
    # Plot C1 histograms
    C1_axis_N_data_N_trained.hist(C1_data_N['Conf mean'], bins = conf_bin_edges, color = 'blue')
    C1_axis_N_data_S_trained.hist(C1_data_N['Conf mean south trained'], bins = conf_bin_edges, color = 'blue')
    C1_axis_S_data_N_trained.hist(C1_data_S['Conf mean north trained'], bins = conf_bin_edges, color = 'blue')
    C1_axis_S_data_S_trained.hist(C1_data_S['Conf mean'], bins = conf_bin_edges, color = 'blue')
    
    # Plot C2 histograms
    C2_axis_N_data_N_trained.hist(C2_data_N['Conf mean'], bins = conf_bin_edges, color = 'orange')
    C2_axis_N_data_S_trained.hist(C2_data_N['Conf mean south trained'], bins = conf_bin_edges, color = 'orange')
    C2_axis_S_data_N_trained.hist(C2_data_S['Conf mean north trained'], bins = conf_bin_edges, color = 'orange')
    C2_axis_S_data_S_trained.hist(C2_data_S['Conf mean'], bins = conf_bin_edges, color = 'orange')
    
    #Plot Non-C1C2 histograms
    Non_C1C2_axis_N_data_N_trained.hist(Non_C1C2_data_N['Conf mean'], bins = conf_bin_edges, color = 'green')
    Non_C1C2_axis_N_data_S_trained.hist(Non_C1C2_data_N['Conf mean south trained'], bins = conf_bin_edges, color = 'green')
    Non_C1C2_axis_S_data_N_trained.hist(Non_C1C2_data_S['Conf mean north trained'], bins = conf_bin_edges, color = 'green')
    Non_C1C2_axis_S_data_S_trained.hist(Non_C1C2_data_S['Conf mean'], bins = conf_bin_edges, color = 'green')
    
    # Lable histograms
    C1_axis_N_data_N_trained      .text(0.79, 13, 'C1', size = 'xx-large')
    C2_axis_N_data_N_trained      .text(0.79, 13, 'C2', size = 'xx-large')
    Non_C1C2_axis_N_data_N_trained.text(0.59, 11000, 'Non C1C2', size = 'xx-large')
    
    C1_axis_N_data_S_trained      .text(0.79, 13, 'C1', size = 'xx-large')
    C2_axis_N_data_S_trained      .text(0.79, 13, 'C2', size = 'xx-large')
    Non_C1C2_axis_N_data_S_trained.text(0.59, 11000, 'Non C1C2', size = 'xx-large')
    
    C1_axis_S_data_N_trained      .text(0.79, 13, 'C1', size = 'xx-large')
    C2_axis_S_data_N_trained      .text(0.79, 13, 'C2', size = 'xx-large')
    Non_C1C2_axis_S_data_N_trained.text(0.59, 11000, 'Non C1C2', size = 'xx-large')
    
    C1_axis_S_data_S_trained      .text(0.79, 13, 'C1', size = 'xx-large')
    C2_axis_S_data_S_trained      .text(0.79, 13, 'C2', size = 'xx-large')
    Non_C1C2_axis_S_data_S_trained.text(0.59, 11000, 'Non C1C2', size = 'xx-large')
    
    # -- set y lable stuff --
    
    # log y axis
    C1_axis_N_data_N_trained.set_yscale('log')
    C2_axis_N_data_N_trained.set_yscale('log')
    Non_C1C2_axis_N_data_N_trained.set_yscale('log')
    
    C1_axis_N_data_S_trained.set_yscale('log')
    C2_axis_N_data_S_trained.set_yscale('log')
    Non_C1C2_axis_N_data_S_trained.set_yscale('log')
    
    C1_axis_S_data_N_trained.set_yscale('log')
    C2_axis_S_data_N_trained.set_yscale('log')
    Non_C1C2_axis_S_data_N_trained.set_yscale('log')
    
    C1_axis_S_data_S_trained.set_yscale('log')
    C2_axis_S_data_S_trained.set_yscale('log')
    Non_C1C2_axis_S_data_S_trained.set_yscale('log')
    
    # set y ticks
    C1_axis_N_data_N_trained      .set_yticks(y_ticks_log)
    C1_axis_N_data_N_trained      .set_yticks(y_ticks_minor_log, minor = True)
    C2_axis_N_data_N_trained      .set_yticks(y_ticks_log)
    C2_axis_N_data_N_trained      .set_yticks(y_ticks_minor_log, minor = True)
    Non_C1C2_axis_N_data_N_trained.set_yticks(y_ticks_log)
    Non_C1C2_axis_N_data_N_trained.set_yticks(y_ticks_minor_log, minor = True)
    
    C1_axis_N_data_S_trained      .set_yticks(y_ticks_log)
    C1_axis_N_data_S_trained      .set_yticks(y_ticks_minor_log, minor = True)
    C2_axis_N_data_S_trained      .set_yticks(y_ticks_log)
    C2_axis_N_data_S_trained      .set_yticks(y_ticks_minor_log, minor = True)
    Non_C1C2_axis_N_data_S_trained.set_yticks(y_ticks_log)
    Non_C1C2_axis_N_data_S_trained.set_yticks(y_ticks_minor_log, minor = True)
    
    # set y tick size
    C1_axis_N_data_N_trained      .tick_params(axis='y', which='major', labelsize='x-large')
    C2_axis_N_data_N_trained      .tick_params(axis='y', which='major', labelsize='x-large')
    Non_C1C2_axis_N_data_N_trained.tick_params(axis='y', which='major', labelsize='x-large')
    
    C1_axis_N_data_S_trained      .tick_params(axis='y', which='major', labelsize='x-large')
    C2_axis_N_data_S_trained      .tick_params(axis='y', which='major', labelsize='x-large')
    Non_C1C2_axis_N_data_S_trained.tick_params(axis='y', which='major', labelsize='x-large')
    
    # Remove y ticks from right column
    C1_axis_S_data_N_trained      .tick_params(axis='y', which='both', left=False, labelleft=False)
    C2_axis_S_data_N_trained      .tick_params(axis='y', which='both', left=False, labelleft=False)
    Non_C1C2_axis_S_data_N_trained.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    C1_axis_S_data_S_trained      .tick_params(axis='y', which='both', left=False, labelleft=False)
    C2_axis_S_data_S_trained      .tick_params(axis='y', which='both', left=False, labelleft=False)
    Non_C1C2_axis_S_data_S_trained.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # set y range
    C1_axis_N_data_N_trained      .set_ylim(C1C2_hist_range_log)
    C2_axis_N_data_N_trained      .set_ylim(C1C2_hist_range_log)
    Non_C1C2_axis_N_data_N_trained.set_ylim(Non_C1C2_hist_range_log)
    
    C1_axis_N_data_S_trained      .set_ylim(C1C2_hist_range_log)
    C2_axis_N_data_S_trained      .set_ylim(C1C2_hist_range_log)
    Non_C1C2_axis_N_data_S_trained.set_ylim(Non_C1C2_hist_range_log)
    
    C1_axis_S_data_N_trained      .set_ylim(C1C2_hist_range_log)
    C2_axis_S_data_N_trained      .set_ylim(C1C2_hist_range_log)
    Non_C1C2_axis_S_data_N_trained.set_ylim(Non_C1C2_hist_range_log)
    
    C1_axis_S_data_S_trained      .set_ylim(C1C2_hist_range_log)
    C2_axis_S_data_S_trained      .set_ylim(C1C2_hist_range_log)
    Non_C1C2_axis_S_data_S_trained.set_ylim(Non_C1C2_hist_range_log)
    
    # -- Set x lable stuff --
    
    # eddit ticks
    Non_C1C2_axis_N_data_S_trained.set_xticks([0.1*i for i in range(10)])
    Non_C1C2_axis_N_data_S_trained.set_xticks([0.05*i for i in range(19)], minor = True)
    Non_C1C2_axis_N_data_S_trained.tick_params(axis='x', which='major', labelsize='x-large')
    
    Non_C1C2_axis_S_data_S_trained.set_xticks([0.1*i for i in range(10)])
    Non_C1C2_axis_S_data_S_trained.set_xticks([0.05*i for i in range(19)], minor = True)
    Non_C1C2_axis_S_data_S_trained.tick_params(axis='x', which='major', labelsize='x-large')
    
    # Remove X ticks from plots above gap
    Non_C1C2_axis_N_data_N_trained.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    Non_C1C2_axis_S_data_N_trained.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # change confidence value limits
    C1_axis_N_data_N_trained.set_xlim(conf_hist_lim)
   
    
    # -- general formating --
    
    # lable north and south catalogue
    C1_axis_N_data_N_trained.set_title("North XXL source catalogue", size = 'xx-large')
    C1_axis_S_data_N_trained.set_title("South XXL source catalogue", size = 'xx-large')
    
    # lable north and south trained GP
    Non_C1C2_axis_N_data_N_trained.text(-0.24, 10, 'GP trained on the North\n        XXL catalogue', size = 'xx-large', rotation = 'vertical')
    Non_C1C2_axis_N_data_S_trained.text(-0.24, 10, 'GP trained on the Sotuh\n        XXL catalogue', size = 'xx-large', rotation = 'vertical')
    
    # label y axis
    Non_C1C2_axis_N_data_S_trained.set_xlabel("Confidence value", size = 'xx-large')
    Non_C1C2_axis_S_data_S_trained.set_xlabel("Confidence value", size = 'xx-large')
    
    # set plot size
    plt.gcf().set_size_inches(10,10)
    
    # Adjust padding
    #plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    
    plt.show()

Plot_Confidence_Histogram_Log(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, label = 'North catalogue')
Plot_Confidence_Histogram_Log(Data_S_C1, Data_S_C2, Data_S_Non_C1C2, label = 'South catalogue')

Plot_Confidence_Histogram_Comparison_Log(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)

# -- Plot standard deviation on confidence values ----------------------------------------------------------------------------

def plot_conf_standard_deviation(Data_N, Data_N_C1, Data_N_C2, Data_N_non_C1C2, Data_S, Data_S_C1, Data_S_C2, Data_S_non_C1C2):
    
    # conf_standard_deviation_bin_edges
    # conf_standard_deviation_bin_centers

    # conf_standard_deviation_conf_lim
    # conf_standard_deviation_std_lim
    
    # Create subplots
    fig, axis = plt.subplots(2, 2, sharex = True, sharey = True)
    
    N_data_N_trained = axis[0,0]
    N_data_S_trained = axis[1,0]
    S_data_N_trained = axis[0,1]
    S_data_S_trained = axis[1,1]
    
    # Plot non-C1C2 data
    N_data_N_trained.scatter(Data_N_C1['Conf mean']              , Data_N_C1['Conf std']              /np.sqrt(10), alpha = 1, s = 70, marker = 'v', c = 'blue', edgecolors='none', zorder = 3)
    N_data_S_trained.scatter(Data_N_C1['Conf mean south trained'], Data_N_C1['Conf std south trained']/np.sqrt(10), alpha = 1, s = 70, marker = 'v', c = 'blue', edgecolors='none', zorder = 3)
    S_data_N_trained.scatter(Data_S_C1['Conf mean north trained'], Data_S_C1['Conf std north trained']/np.sqrt(10), alpha = 1, s = 70, marker = 'v', c = 'blue', edgecolors='none', zorder = 3)
    S_data_S_trained.scatter(Data_S_C1['Conf mean']              , Data_S_C1['Conf std']              /np.sqrt(10), alpha = 1, s = 70, marker = 'v', c = 'blue', edgecolors='none', zorder = 3)
    
    # Plot C2 data
    N_data_N_trained.scatter(Data_N_C2['Conf mean']              , Data_N_C2['Conf std']              /np.sqrt(10), alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
    N_data_S_trained.scatter(Data_N_C2['Conf mean south trained'], Data_N_C2['Conf std south trained']/np.sqrt(10), alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
    S_data_N_trained.scatter(Data_S_C2['Conf mean north trained'], Data_S_C2['Conf std north trained']/np.sqrt(10), alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
    S_data_S_trained.scatter(Data_S_C2['Conf mean']              , Data_S_C2['Conf std']              /np.sqrt(10), alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
    
    # Plot non-C1C2 data
    N_data_N_trained.scatter(Data_N_non_C1C2['Conf mean']              , Data_N_non_C1C2['Conf std']              /np.sqrt(10), alpha = 1, s = 70, marker = 'o', c = 'green', edgecolors='none', zorder = 1)
    N_data_S_trained.scatter(Data_N_non_C1C2['Conf mean south trained'], Data_N_non_C1C2['Conf std south trained']/np.sqrt(10), alpha = 1, s = 70, marker = 'o', c = 'green', edgecolors='none', zorder = 1)
    S_data_N_trained.scatter(Data_S_non_C1C2['Conf mean north trained'], Data_S_non_C1C2['Conf std north trained']/np.sqrt(10), alpha = 1, s = 70, marker = 'o', c = 'green', edgecolors='none', zorder = 1)
    S_data_S_trained.scatter(Data_S_non_C1C2['Conf mean']              , Data_S_non_C1C2['Conf std']              /np.sqrt(10), alpha = 1, s = 70, marker = 'o', c = 'green', edgecolors='none', zorder = 1)
    
    # -- calculate and plot bined avverage --
    
    # First get the sum of all standard deviations in each bin
    N_data_N_trained_average = np.histogram(Data_N['Conf mean']              , weights = Data_N['Conf std']              /np.sqrt(10), bins = conf_standard_deviation_bin_edges)[0]
    N_data_S_trained_average = np.histogram(Data_N['Conf mean south trained'], weights = Data_N['Conf std south trained']/np.sqrt(10), bins = conf_standard_deviation_bin_edges)[0]
    S_data_N_trained_average = np.histogram(Data_S['Conf mean north trained'], weights = Data_S['Conf std north trained']/np.sqrt(10), bins = conf_standard_deviation_bin_edges)[0]
    S_data_S_trained_average = np.histogram(Data_S['Conf mean']              , weights = Data_S['Conf std']              /np.sqrt(10), bins = conf_standard_deviation_bin_edges)[0]
    
    # Divide by the number of objects in each bin
    N_data_N_trained_average /= np.histogram(Data_N['Conf mean']              , bins = conf_standard_deviation_bin_edges)[0]
    N_data_S_trained_average /= np.histogram(Data_N['Conf mean south trained'], bins = conf_standard_deviation_bin_edges)[0]
    S_data_N_trained_average /= np.histogram(Data_S['Conf mean north trained'], bins = conf_standard_deviation_bin_edges)[0]
    S_data_S_trained_average /= np.histogram(Data_S['Conf mean']              , bins = conf_standard_deviation_bin_edges)[0]
    
    # plot bineed average
    N_data_N_trained.plot(conf_standard_deviation_bin_centers, N_data_N_trained_average, c = 'black', zorder = 4)
    N_data_S_trained.plot(conf_standard_deviation_bin_centers, N_data_S_trained_average, c = 'black', zorder = 4)
    S_data_N_trained.plot(conf_standard_deviation_bin_centers, S_data_N_trained_average, c = 'black', zorder = 4)
    S_data_S_trained.plot(conf_standard_deviation_bin_centers, S_data_S_trained_average, c = 'black', zorder = 4)
    
    
    # -- Formatting --
    
    # Remove vertical space
    fig.subplots_adjust(hspace = 0, wspace = 0)
        
    # set plot size
    plt.gcf().set_size_inches(10,9)
    
    # Adjust padding
    plt.subplots_adjust(left=0.20, right=0.9, top=0.9, bottom=0.15)
    
    # - y axis -
    
    # tick lable sizes
    N_data_N_trained.tick_params(axis='y', which='major', labelsize='x-large')
    N_data_S_trained.tick_params(axis='y', which='major', labelsize='x-large')
    
    # Add ticks
    N_data_N_trained.set_yticks(np.linspace(0,0.02,5))
    N_data_N_trained.set_yticks(np.linspace(0,0.02,21), minor = True)
    N_data_S_trained.set_yticks(np.linspace(0,0.02,5))
    N_data_S_trained.set_yticks(np.linspace(0,0.02,21), minor = True)
    
    # remove ticks from right hand plots
    S_data_N_trained.tick_params(axis='y', which='both', left=False, labelleft=False)
    S_data_S_trained.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    # lable GP used
    N_data_N_trained.text( -0.45, 0.002, 'GP trained on the North\n        XXL catalogue', size = 'xx-large', rotation = 'vertical')
    N_data_S_trained.text( -0.45, 0.002, 'GP trained on the Sotuh\n        XXL catalogue', size = 'xx-large', rotation = 'vertical')
    
    # lable axis
    N_data_N_trained.set_ylabel('Standard error in\nconfidence value', size = 'x-large')
    N_data_S_trained.set_ylabel('Standard error in\nconfidence value', size = 'x-large')
    
    # Set limits
    N_data_N_trained.set_ylim(conf_standard_deviation_std_lim)
    
    # - x axis -
    
    # tick lable sizes
    N_data_S_trained.tick_params(axis='x', which='major', labelsize='x-large')
    S_data_S_trained.tick_params(axis='x', which='major', labelsize='x-large')
    
    
    
    # lable axis
    N_data_S_trained.set_xlabel('Confidence value', size = 'xx-large')
    S_data_S_trained.set_xlabel('Confidence value', size = 'xx-large')
    
    # lable catalogue
    N_data_N_trained.set_title('North XXL catalogue', size = 'xx-large')
    S_data_N_trained.set_title('South XXL catalogue', size = 'xx-large')
    
    # Add ticks
    N_data_S_trained.set_xticks(np.linspace(0,1,11))
    N_data_S_trained.set_xticks(np.linspace(0,1,21), minor = True)
    S_data_S_trained.set_xticks(np.linspace(0,1,11))
    S_data_S_trained.set_xticks(np.linspace(0,1,21), minor = True)
    
    # Set limits
    N_data_N_trained.set_xlim(conf_standard_deviation_conf_lim)
    
    plt.show()

plot_conf_standard_deviation(Data_N, Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)


# -- Plot comparison for confidence values from north and south trained GP ---------------------------------------------------

def plot_conf_comparison(C1_conf_north, C2_conf_north, non_C1C2_conf_north, C1_conf_south, C2_conf_south, non_C1C2_conf_south):
    
    # plot one to one line
    plt.plot([0,1],[0,1], c = 'black', zorder = 0, linestyle = '--')

    # plot sources
    plt.scatter(non_C1C2_conf_north, non_C1C2_conf_south, alpha = 1, s = 100, marker = 'o', c = 'green' , edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    plt.scatter(C2_conf_north      , C2_conf_south      , alpha = 1, s = 100, marker = '^', c = 'orange', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    plt.scatter(C1_conf_north      , C1_conf_south      , alpha = 1, s = 100, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    # set plot limits
    plt.xlim(0,0.9)
    plt.ylim(0,0.9)
    
    # set aspect raio to equal
    plt.gca().set_aspect('equal')
    
    # lable axies
    plt.xlabel('Confidence value from GP \n trained on XXL North', size = 'xx-large')
    plt.ylabel('Confidence value from GP \n trained on XXL South', size = 'xx-large')
    
    # set tick values
    # x ticks
    plt.gca().set_xticks([0.1 *i for i in range(10)])
    plt.gca().set_xticks([0.05*i for i in range(19)], minor = True)
    
    # y ticks
    plt.gca().set_yticks([0.1 *i for i in range(10)])
    plt.gca().set_yticks([0.05*i for i in range(19)], minor = True)
    
    #set tick lable size
    plt.gca().tick_params(axis='both', which='major', labelsize='x-large')
    
    # set plot size
    plt.gcf().set_size_inches(7,7)
    
    # Adjust padding
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    
    plt.show()
    
# north and south relate to the catalogue the GP was trained on.
# N and S relate to the catalogue the data coresponds to. 
def plot_conf_comparison_both(conf_north_N, C1_conf_north_N, C2_conf_north_N, non_C1C2_conf_north_N, conf_south_N, C1_conf_south_N, C2_conf_south_N, non_C1C2_conf_south_N,
                              conf_north_S, C1_conf_north_S, C2_conf_north_S, non_C1C2_conf_north_S, conf_south_S, C1_conf_south_S, C2_conf_south_S, non_C1C2_conf_south_S):

    # -- Calculate person correlation --

    corelation_N = np.corrcoef(conf_north_N, conf_south_N)[0,1]
    corelation_S = np.corrcoef(conf_north_S, conf_south_S)[0,1]

    # -- Calculate average line --
    
    conf_bins = np.linspace(0,1,21)
    conf_bins_center = (conf_bins[1:] + conf_bins[:-1])/2
    
    # Sum confidence value asigned by GP trained on South binned on confidence value asigned by GP trained on North catalogue
    conf_south_average_N = np.histogram(conf_north_N, weights = conf_south_N, bins = conf_bins)[0]
    conf_south_average_S = np.histogram(conf_north_S, weights = conf_south_S, bins = conf_bins)[0]
    
    # Divide by number of sources in bin on confidence value asigned by GP trained on North catalogue
    conf_south_average_N /= np.histogram(conf_north_N, bins = conf_bins)[0]
    conf_south_average_S /= np.histogram(conf_north_S, bins = conf_bins)[0]
    
    # -- Plot data --
    
    fig, (axis_N, axis_S) = plt.subplots(2, 1, sharex=True)
    
    # plot one to one line
    axis_N.plot([0,1],[0,1], c = 'black', zorder = 0, linestyle = '--')
    axis_S.plot([0,1],[0,1], c = 'black', zorder = 0, linestyle = '--')

    # plot sources
    axis_N.scatter(non_C1C2_conf_north_N, non_C1C2_conf_south_N, alpha = 1, s = 100, marker = 'o', c = 'green' , edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    axis_N.scatter(C2_conf_north_N      , C2_conf_south_N      , alpha = 1, s = 100, marker = '^', c = 'orange', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    axis_N.scatter(C1_conf_north_N      , C1_conf_south_N      , alpha = 1, s = 100, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    axis_S.scatter(non_C1C2_conf_north_S, non_C1C2_conf_south_S, alpha = 1, s = 100, marker = 'o', c = 'green' , edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    axis_S.scatter(C2_conf_north_S      , C2_conf_south_S      , alpha = 1, s = 100, marker = '^', c = 'orange', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    axis_S.scatter(C1_conf_north_S      , C1_conf_south_S      , alpha = 1, s = 100, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    # plot average
    #axis_N.plot(conf_bins_center, conf_south_average_N, c = 'black', zorder = 4)
    #axis_S.plot(conf_bins_center, conf_south_average_S, c = 'black', zorder = 4)
    
    # set aspect raio to equal
    axis_N.set_aspect('equal')
    axis_S.set_aspect('equal')
    
    # lable axies
    axis_S.set_xlabel('Confidence value from GP \n trained on XXL North', size = 'xx-large')
    axis_N.set_ylabel('Confidence value from GP \n trained on XXL South', size = 'xx-large')
    axis_S.set_ylabel('Confidence value from GP \n trained on XXL South', size = 'xx-large')
    
    # set tick values
    # x ticks
    axis_S.set_xticks([0.1 *i for i in range(10)])
    axis_S.set_xticks([0.05*i for i in range(19)], minor = True)
    
    # y ticks
    axis_N.set_yticks([0.1 *i for i in range(10)])
    axis_N.set_yticks([0.05*i for i in range(19)], minor = True)
    axis_S.set_yticks([0.1 *i for i in range(10)])
    axis_S.set_yticks([0.05*i for i in range(19)], minor = True)
    
    #set tick lable size
    axis_N.tick_params(axis='both', which='major', labelsize='x-large')
    axis_S.tick_params(axis='both', which='major', labelsize='x-large')
    
    # set plot limits
    axis_S.set_xlim(0,0.85)
    axis_N.set_ylim(0,0.85)
    axis_S.set_ylim(0,0.85)
    
    # set plot size
    fig.set_size_inches(7,14)
    
    # Adjust padding
    plt.subplots_adjust(left=0.1, right=1, top=0.95, bottom=0.1, hspace = 0)
    
    axis_N.text(0.01, 0.77, 'North source  catalogue\n    Correlation = ' + str(round(corelation_N,2)), size = 'xx-large')
    axis_S.text(0.01, 0.77, 'South source  catalogue\n    Correlation = ' + str(round(corelation_S,2)), size = 'xx-large')
    
    plt.show()
    
    
plot_conf_comparison(Data_N_C1['Conf mean']              , Data_N_C2['Conf mean']              , Data_N_Non_C1C2['Conf mean'],
                     Data_N_C1['Conf mean south trained'], Data_N_C2['Conf mean south trained'], Data_N_Non_C1C2['Conf mean south trained'])
                     
plot_conf_comparison(Data_S_C1['Conf mean north trained'], Data_S_C2['Conf mean north trained'], Data_S_Non_C1C2['Conf mean north trained'],
                     Data_S_C1['Conf mean']              , Data_S_C2['Conf mean']              , Data_S_Non_C1C2['Conf mean'])

plot_conf_comparison_both(Data_N['Conf mean']              , Data_N_C1['Conf mean']              , Data_N_C2['Conf mean']              , Data_N_Non_C1C2['Conf mean'],
                          Data_N['Conf mean south trained'], Data_N_C1['Conf mean south trained'], Data_N_C2['Conf mean south trained'], Data_N_Non_C1C2['Conf mean south trained'],
                          Data_S['Conf mean north trained'], Data_S_C1['Conf mean north trained'], Data_S_C2['Conf mean north trained'], Data_S_Non_C1C2['Conf mean north trained'],
                          Data_S['Conf mean']              , Data_S_C1['Conf mean']              , Data_S_C2['Conf mean']              , Data_S_Non_C1C2['Conf mean'])



# -- Plot EXT_LIKE against EXT_RATE_PN for XXL north -----------------------------------------------------------------------------------------------

def Plot_EXT_RATE_PN_vs_EXT_LIKE(C1_data, C2_data, Non_C1C2_data, colour_by_conf = False):

    # calculate correlation
    full_data = pd.concat([C1_data, C2_data, Non_C1C2_data])
    correlation_val = np.corrcoef(full_data['EXT_LIKE'], full_data['EXT_RATE_PN'])[0,1]

    # plot one to one line
    plt.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    plt.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    # plot data
    if(colour_by_conf):
        plt.scatter(Non_C1C2_data['EXT_LIKE'], Non_C1C2_data['EXT_RATE_PN'], c = Non_C1C2_data['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
        plt.scatter(C2_data      ['EXT_LIKE'], C2_data      ['EXT_RATE_PN'], c = C2_data      ['Conf mean'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
        plt.scatter(C1_data      ['EXT_LIKE'], C1_data      ['EXT_RATE_PN'], c = C1_data      ['Conf mean'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
        
        cbar = plt.colorbar(ticks = [0.1*i for i in range(0,11)])
        cbar.set_label('Confidence Value', size = 'xx-large')
        cbar.ax.tick_params(labelsize = 'x-large')
        
    else:
        plt.scatter(Non_C1C2_data['EXT_LIKE'], Non_C1C2_data['EXT_RATE_PN'], alpha = 1, s = 70, marker = 'o', c = 'green' , edgecolors='none', zorder = 1)
        plt.scatter(C2_data      ['EXT_LIKE'], C2_data      ['EXT_RATE_PN'], alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
        plt.scatter(C1_data      ['EXT_LIKE'], C1_data      ['EXT_RATE_PN'], alpha = 1, s = 70, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3)

    # set limits
    plt.xlim(EXT_LIKE_lim)
    plt.ylim(EXT_RATE_PN_lim)

    # log axies
    plt.xscale("log")
    plt.yscale("log")

    # add axis titles
    plt.xlabel("EXT_STAT", fontsize = 'xx-large')
    plt.ylabel("EXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    plt.gca().xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 17)
    plt.gca().xaxis.set_minor_locator(locmin)

    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    plt.gca().yaxis.set_major_locator(locmaj)

    # add minor ticks to y
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 17)
    plt.gca().yaxis.set_minor_locator(locmin)
    
    # remove every other major tick lable
    for label in plt.gca().xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # set size of tick lables
    plt.gca().tick_params(axis='y', which='major', labelsize = 'x-large')
    plt.gca().tick_params(axis='x', which='major', labelsize = 'x-large')

    # set plot size
    plt.gcf().set_size_inches(7,5.5)
    
    # set aspect ratio
    #plt.gca().set_aspect('equal')
    
    # Adjust padding
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

    print(correlation_val)

    # add correlation value
    plt.gca().text(0.0001, 3, 'Correlation = ' + str(round(correlation_val,2)), size = 'xx-large')
    
    plt.show()
    

def Plot_EXT_RATE_PN_vs_EXT_LIKE_comp(C1_data_N, C2_data_N, Non_C1C2_data_N, C1_data_S, C2_data_S, Non_C1C2_data_S):
    
    fig, ((N_data_N_trained, S_data_N_trained), (N_data_S_trained, S_data_S_trained)) = plt.subplots(2,2, sharex=True, sharey=True)
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    
    # plot EXT_LIKE cuts
    N_data_N_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    N_data_S_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_N_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_S_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    N_data_N_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    N_data_S_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_N_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_S_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    
    # Plot all North trained data
    
    # order based on North trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean')
    C1_data_S = C1_data_S.sort_values('Conf mean north trained')
    C2_data_N = C2_data_N.sort_values('Conf mean')
    C2_data_S = C2_data_S.sort_values('Conf mean north trained')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean north trained')
    
    # Plot non-C1C2 data
    N_data_N_trained.scatter(Non_C1C2_data_N['EXT_LIKE'], Non_C1C2_data_N['EXT_RATE_PN'], c = Non_C1C2_data_N['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(Non_C1C2_data_S['EXT_LIKE'], Non_C1C2_data_S['EXT_RATE_PN'], c = Non_C1C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_N_trained.scatter(C2_data_N['EXT_LIKE'], C2_data_N['EXT_RATE_PN'], c = C2_data_N['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C2_data_S['EXT_LIKE'], C2_data_S['EXT_RATE_PN'], c = C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_N_trained.scatter(C1_data_N['EXT_LIKE'], C1_data_N['EXT_RATE_PN'], c = C1_data_N['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C1_data_S['EXT_LIKE'], C1_data_S['EXT_RATE_PN'], c = C1_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    # Plot all South trained data
    
    # order based on South trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean south trained')
    C1_data_S = C1_data_S.sort_values('Conf mean')
    C2_data_N = C2_data_N.sort_values('Conf mean south trained')
    C2_data_S = C2_data_S.sort_values('Conf mean')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean south trained')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean')
    
    # Plot non-C1C2 data
    N_data_S_trained.scatter(Non_C1C2_data_N['EXT_LIKE'], Non_C1C2_data_N['EXT_RATE_PN'], c = Non_C1C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(Non_C1C2_data_S['EXT_LIKE'], Non_C1C2_data_S['EXT_RATE_PN'], c = Non_C1C2_data_S['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_S_trained.scatter(C2_data_N['EXT_LIKE'], C2_data_N['EXT_RATE_PN'], c = C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C2_data_S['EXT_LIKE'], C2_data_S['EXT_RATE_PN'], c = C2_data_S['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_S_trained.scatter(C1_data_N['EXT_LIKE'], C1_data_N['EXT_RATE_PN'], c = C1_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C1_data_S['EXT_LIKE'], C1_data_S['EXT_RATE_PN'], c = C1_data_S['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    # log axies
    N_data_N_trained.set_xscale("log")
    N_data_N_trained.set_yscale("log")
    
    S_data_N_trained.set_xscale("log")
    S_data_N_trained.set_yscale("log")
    
    N_data_S_trained.set_xscale("log")
    N_data_S_trained.set_yscale("log")
    
    S_data_S_trained.set_xscale("log")
    S_data_S_trained.set_yscale("log")
    
    # Add axis lables
    N_data_S_trained.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    S_data_S_trained.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    
    N_data_N_trained.set_ylabel("GP trained on XXL North\n\nEXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    N_data_S_trained.set_ylabel("GP trained on XXL South\n\nEXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    
    # Title columns
    N_data_N_trained.set_title('North XXL catalogue', fontsize = 'xx-large')
    S_data_N_trained.set_title('South XXL catalogue', fontsize = 'xx-large')
    
    # Remove ticks
    S_data_N_trained.tick_params(which = 'both', left = False)
    S_data_S_trained.tick_params(which = 'both', left = False)
    
    # add colourbar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax = (S_data_N_trained, S_data_S_trained), shrink = 1, anchor = (0,0))
    
    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')
    
    # set tick paramiters
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.xaxis.set_major_locator(locmaj)
    S_data_S_trained.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.xaxis.set_minor_locator(locmin)
    S_data_S_trained.xaxis.set_minor_locator(locmin)
    
    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.yaxis.set_major_locator(locmaj)
    S_data_S_trained.yaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.yaxis.set_minor_locator(locmin)
    S_data_S_trained.yaxis.set_minor_locator(locmin)
    
    # set limits
    N_data_N_trained.set_xlim(EXT_LIKE_lim)
    N_data_N_trained.set_ylim(EXT_RATE_PN_lim)
    
    # set plot size
    #plt.gcf().set_size_inches(7,5.5)
    
    plt.show()
    


Plot_EXT_RATE_PN_vs_EXT_LIKE(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, colour_by_conf = True)

# Plot_EXT_RATE_PN_vs_EXT_LIKE_comp(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)

# -- Plot EXT_LIKE against EXT_RATE_MOS for XXL -----------------------------------------------------------------------------------------------------

def Plot_EXT_RATE_MOS_vs_EXT_LIKE_comp(C1_data_N, C2_data_N, Non_C1C2_data_N, C1_data_S, C2_data_S, Non_C1C2_data_S):
    
    fig, ((N_data_N_trained, S_data_N_trained), (N_data_S_trained, S_data_S_trained)) = plt.subplots(2,2, sharex=True, sharey=True)
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    
    # plot EXT_LIKE cuts
    N_data_N_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    N_data_S_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_N_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_S_trained.plot((EXT_LIKE_C1_cut, EXT_LIKE_C1_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    N_data_N_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    N_data_S_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_N_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    S_data_S_trained.plot((EXT_LIKE_C2_cut, EXT_LIKE_C2_cut), EXT_RATE_MOS_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    
    # Plot all North trained data
    
    # order based on North trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean')
    C1_data_S = C1_data_S.sort_values('Conf mean north trained')
    C2_data_N = C2_data_N.sort_values('Conf mean')
    C2_data_S = C2_data_S.sort_values('Conf mean north trained')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean north trained')
    
    # Plot non-C1C2 data
    N_data_N_trained.scatter(Non_C1C2_data_N['EXT_LIKE'], Non_C1C2_data_N['EXT_RATE_MOS'], c = Non_C1C2_data_N['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(Non_C1C2_data_S['EXT_LIKE'], Non_C1C2_data_S['EXT_RATE_MOS'], c = Non_C1C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_N_trained.scatter(C2_data_N['EXT_LIKE'], C2_data_N['EXT_RATE_MOS'], c = C2_data_N['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C2_data_S['EXT_LIKE'], C2_data_S['EXT_RATE_MOS'], c = C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_N_trained.scatter(C1_data_N['EXT_LIKE'], C1_data_N['EXT_RATE_MOS'], c = C1_data_N['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C1_data_S['EXT_LIKE'], C1_data_S['EXT_RATE_MOS'], c = C1_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    # Plot all South trained data
    
    # order based on South trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean south trained')
    C1_data_S = C1_data_S.sort_values('Conf mean')
    C2_data_N = C2_data_N.sort_values('Conf mean south trained')
    C2_data_S = C2_data_S.sort_values('Conf mean')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean south trained')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean')
    
    # Plot non-C1C2 data
    N_data_S_trained.scatter(Non_C1C2_data_N['EXT_LIKE'], Non_C1C2_data_N['EXT_RATE_MOS'], c = Non_C1C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(Non_C1C2_data_S['EXT_LIKE'], Non_C1C2_data_S['EXT_RATE_MOS'], c = Non_C1C2_data_S['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_S_trained.scatter(C2_data_N['EXT_LIKE'], C2_data_N['EXT_RATE_MOS'], c = C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C2_data_S['EXT_LIKE'], C2_data_S['EXT_RATE_MOS'], c = C2_data_S['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_S_trained.scatter(C1_data_N['EXT_LIKE'], C1_data_N['EXT_RATE_MOS'], c = C1_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C1_data_S['EXT_LIKE'], C1_data_S['EXT_RATE_MOS'], c = C1_data_S['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    # log axies
    N_data_N_trained.set_xscale("log")
    N_data_N_trained.set_yscale("log")
    
    S_data_N_trained.set_xscale("log")
    S_data_N_trained.set_yscale("log")
    
    N_data_S_trained.set_xscale("log")
    N_data_S_trained.set_yscale("log")
    
    S_data_S_trained.set_xscale("log")
    S_data_S_trained.set_yscale("log")
    
    # Add axis lables
    N_data_S_trained.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    S_data_S_trained.set_xlabel("EXT_STAT", fontsize = 'xx-large')
    
    N_data_N_trained.set_ylabel("GP trained on XXL North\n\nEXT_RATE_MOS (counts/s)", fontsize = 'xx-large')
    N_data_S_trained.set_ylabel("GP trained on XXL South\n\nEXT_RATE_MOS (counts/s)", fontsize = 'xx-large')
    
    # Title columns
    N_data_N_trained.set_title('North XXL catalogue', fontsize = 'xx-large')
    S_data_N_trained.set_title('South XXL catalogue', fontsize = 'xx-large')
    
    # Remove ticks
    S_data_N_trained.tick_params(which = 'both', left = False)
    S_data_S_trained.tick_params(which = 'both', left = False)
    
    # add colourbar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax = (S_data_N_trained, S_data_S_trained), shrink = 1, anchor = (0,0))
    
    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')
    
    # set tick paramiters
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.xaxis.set_major_locator(locmaj)
    S_data_S_trained.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.xaxis.set_minor_locator(locmin)
    S_data_S_trained.xaxis.set_minor_locator(locmin)
    
    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.yaxis.set_major_locator(locmaj)
    S_data_S_trained.yaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.yaxis.set_minor_locator(locmin)
    S_data_S_trained.yaxis.set_minor_locator(locmin)
    
    # set limits
    N_data_N_trained.set_xlim(EXT_LIKE_lim)
    N_data_N_trained.set_ylim(EXT_RATE_MOS_lim)
    
    # set plot size
    #plt.gcf().set_size_inches(7,5.5)
    
    plt.show()

# Plot_EXT_RATE_MOS_vs_EXT_LIKE_comp(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)

# -- Plot PNT_RATE_PN against EXT_RATE_PN for XXL north -----------------------------------------------------------------------------------------------

def Plot_EXT_RATE_PN_vs_PNT_RATE_PN(C1_data, C2_data, Non_C1C2_data, colour_by_conf = False):

    # plot one to one line
    plt.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    
    # plot data
    if(colour_by_conf):
        plt.scatter(Non_C1C2_data['EXT_RATE_PN'], Non_C1C2_data['PNT_RATE_PN'], c = Non_C1C2_data['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
        plt.scatter(C2_data      ['EXT_RATE_PN'], C2_data      ['PNT_RATE_PN'], c = C2_data      ['Conf mean'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
        plt.scatter(C1_data      ['EXT_RATE_PN'], C1_data      ['PNT_RATE_PN'], c = C1_data      ['Conf mean'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
        
        cbar = plt.colorbar(ticks = [0.1*i for i in range(0,11)])
        cbar.set_label('Confidence Value', size = 'xx-large')
        cbar.ax.tick_params(labelsize = 'x-large')
        
    else:
        plt.scatter(Non_C1C2_data['EXT_RATE_PN'], Non_C1C2_data['PNT_RATE_PN'], alpha = 1, s = 70, marker = 'o', c = 'green' , edgecolors='none', zorder = 1)
        plt.scatter(C2_data      ['EXT_RATE_PN'], C2_data      ['PNT_RATE_PN'], alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
        plt.scatter(C1_data      ['EXT_RATE_PN'], C1_data      ['PNT_RATE_PN'], alpha = 1, s = 70, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3)

    # set limits
    plt.xlim(EXT_RATE_PN_lim)
    plt.ylim(PNT_RATE_PN_lim)

    # log axies
    plt.xscale("log")
    plt.yscale("log")

    # add axis titles
    plt.xlabel("EXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    plt.ylabel("PNT_RATE_PN (counts/s)", fontsize = 'xx-large')
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    plt.gca().xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 17)
    plt.gca().xaxis.set_minor_locator(locmin)

    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    plt.gca().yaxis.set_major_locator(locmaj)

    # add minor ticks to y
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 17)
    plt.gca().yaxis.set_minor_locator(locmin)

    # set size of tick lables
    plt.gca().tick_params(axis='both', which='major', labelsize = 'x-large')

    # set plot size
    plt.gcf().set_size_inches(7,5.5)
    
    # set aspect ratio
    plt.gca().set_aspect('equal')
    
    # Adjust padding
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    
    plt.show()

def Plot_initial_and_conf_EXT_RATE_PN_vs_PNT_RATE_PN(C1_data, C2_data, Non_C1C2_data):

    # Calculate correlation value
    full_data = pd.concat([C1_data, C2_data, Non_C1C2_data])
    correlation_val = np.corrcoef(full_data['EXT_RATE_PN'], full_data['PNT_RATE_PN'])[0, 1]
    
    fig, (axis_init, axis_conf) = plt.subplots(2, 1, sharex=True)
    
    # Remove vertical space
    fig.subplots_adjust(hspace=0)
    
    # plot one to one line
    axis_init.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle=':', linewidth = Sample_line_width, zorder = 0)
    axis_conf.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle=':', linewidth = Sample_line_width, zorder = 0)
    
    # plot initial data
    axis_init.scatter(Non_C1C2_data['EXT_RATE_PN'], Non_C1C2_data['PNT_RATE_PN'], alpha = 1, s = 70, marker = 'o', c = 'green' , edgecolors='none', zorder = 1)
    axis_init.scatter(C2_data      ['EXT_RATE_PN'], C2_data      ['PNT_RATE_PN'], alpha = 1, s = 70, marker = '^', c = 'orange', edgecolors='none', zorder = 2)
    axis_init.scatter(C1_data      ['EXT_RATE_PN'], C1_data      ['PNT_RATE_PN'], alpha = 1, s = 70, marker = 'v', c = 'blue'  , edgecolors='none', zorder = 3)
    
    # plot confidence data
    axis_conf.scatter(Non_C1C2_data['EXT_RATE_PN'], Non_C1C2_data['PNT_RATE_PN'], c = Non_C1C2_data['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    axis_conf.scatter(C2_data      ['EXT_RATE_PN'], C2_data      ['PNT_RATE_PN'], c = C2_data      ['Conf mean'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    axis_conf.scatter(C1_data      ['EXT_RATE_PN'], C1_data      ['PNT_RATE_PN'], c = C1_data      ['Conf mean'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    # Add colourbar
    
    # we need to create the colour map and mappable in order to produce a colour bar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])
    
    # set limits
    axis_conf.set_xlim(EXT_RATE_PN_lim)
    axis_init.set_ylim(PNT_RATE_PN_lim)
    axis_conf.set_ylim(PNT_RATE_PN_lim)

    # log axies
    axis_init.set_xscale("log")
    axis_init.set_yscale("log")
    axis_conf.set_xscale("log")
    axis_conf.set_yscale("log")

    # add axis titles
    axis_conf.set_xlabel("EXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    axis_init.set_ylabel("PNT_RATE_PN (counts/s)", fontsize = 'xx-large')
    axis_conf.set_ylabel("PNT_RATE_PN (counts/s)", fontsize = 'xx-large')
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    axis_conf.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 17)
    axis_conf.xaxis.set_minor_locator(locmin)
    
    # remove every other major x tick lable
    for label in axis_conf.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    axis_conf.yaxis.set_major_locator(locmaj)
    axis_init.yaxis.set_major_locator(locmaj)

    # add minor ticks to y
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 17)
    axis_init.yaxis.set_minor_locator(locmin)
    axis_conf.yaxis.set_minor_locator(locmin)

    # set size of tick lables
    axis_init.tick_params(axis='y'   , which='major', labelsize = 'x-large')
    axis_conf.tick_params(axis='both', which='major', labelsize = 'x-large')

    # set plot size
    plt.gcf().set_size_inches(7,9)
    
    # set aspect ratio
    axis_init.set_aspect('equal')
    axis_conf.set_aspect('equal')
    
    # Adjust padding
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    
    # create colourbar
    cbar = fig.colorbar(sm, ax = (axis_init, axis_conf), shrink = 0.5, anchor = (0,0))
    
    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')

    # add correlation value to plot
    axis_init.text(6E-9, 1, 'Correlation = '  + str(round(correlation_val, 2)), fontsize = 'x-large')
    
    plt.show()
    
def Plot_EXT_RATE_PN_vs_PNT_RATE_PN_comp(C1_data_N, C2_data_N, Non_C1C2_data_N, C1_data_S, C2_data_S, Non_C1C2_data_S):
    
    fig, ((N_data_labels, S_data_labels, colour_bar_1), (N_data_N_trained, S_data_N_trained, colour_bar_2), (N_data_S_trained, S_data_S_trained, colour_bar_3)) = plt.subplots(3,3, sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 0.1]})
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    
    # Set clour bar sections invisible
    colour_bar_1.set_visible(False)
    colour_bar_2.set_visible(False)
    colour_bar_3.set_visible(False)
    
    # plot one to one line
    N_data_labels.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    S_data_labels.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    
    N_data_N_trained.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    N_data_S_trained.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    S_data_N_trained.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    S_data_S_trained.plot(EXT_RATE_PN_lim, EXT_RATE_PN_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    
    # plot initial state
    N_data_labels.scatter(Non_C1C2_data_N['EXT_RATE_PN'], Non_C1C2_data_N['PNT_RATE_PN'], c = 'green', alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_labels.scatter(Non_C1C2_data_S['EXT_RATE_PN'], Non_C1C2_data_S['PNT_RATE_PN'], c = 'green', alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    N_data_labels.scatter(C2_data_N['EXT_RATE_PN'], C2_data_N['PNT_RATE_PN'], c = 'orange', alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_labels.scatter(C2_data_S['EXT_RATE_PN'], C2_data_S['PNT_RATE_PN'], c = 'orange', alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    N_data_labels.scatter(C1_data_N['EXT_RATE_PN'], C1_data_N['PNT_RATE_PN'], c = 'blue', alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_labels.scatter(C1_data_S['EXT_RATE_PN'], C1_data_S['PNT_RATE_PN'], c = 'blue', alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    
    
    
    # -- Plot all North trained data --
    
    # order based on North trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean')
    C1_data_S = C1_data_S.sort_values('Conf mean north trained')
    C2_data_N = C2_data_N.sort_values('Conf mean')
    C2_data_S = C2_data_S.sort_values('Conf mean north trained')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean north trained')
    
    # Plot non-C1C2 data
    N_data_N_trained.scatter(Non_C1C2_data_N['EXT_RATE_PN'], Non_C1C2_data_N['PNT_RATE_PN'], c = Non_C1C2_data_N['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(Non_C1C2_data_S['EXT_RATE_PN'], Non_C1C2_data_S['PNT_RATE_PN'], c = Non_C1C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_N_trained.scatter(C2_data_N['EXT_RATE_PN'], C2_data_N['PNT_RATE_PN'], c = C2_data_N['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C2_data_S['EXT_RATE_PN'], C2_data_S['PNT_RATE_PN'], c = C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_N_trained.scatter(C1_data_N['EXT_RATE_PN'], C1_data_N['PNT_RATE_PN'], c = C1_data_N['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C1_data_S['EXT_RATE_PN'], C1_data_S['PNT_RATE_PN'], c = C1_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    # -- Plot all South trained data --
    
    # order based on South trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean south trained')
    C1_data_S = C1_data_S.sort_values('Conf mean')
    C2_data_N = C2_data_N.sort_values('Conf mean south trained')
    C2_data_S = C2_data_S.sort_values('Conf mean')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean south trained')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean')
    
    # Plot non-C1C2 data
    N_data_S_trained.scatter(Non_C1C2_data_N['EXT_RATE_PN'], Non_C1C2_data_N['PNT_RATE_PN'], c = Non_C1C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(Non_C1C2_data_S['EXT_RATE_PN'], Non_C1C2_data_S['PNT_RATE_PN'], c = Non_C1C2_data_S['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_S_trained.scatter(C2_data_N['EXT_RATE_PN'], C2_data_N['PNT_RATE_PN'], c = C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C2_data_S['EXT_RATE_PN'], C2_data_S['PNT_RATE_PN'], c = C2_data_S['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_S_trained.scatter(C1_data_N['EXT_RATE_PN'], C1_data_N['PNT_RATE_PN'], c = C1_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C1_data_S['EXT_RATE_PN'], C1_data_S['PNT_RATE_PN'], c = C1_data_S['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    # -- manage axies --
    
    # log axies
    N_data_labels   .set_xscale("log")
    N_data_labels   .set_yscale("log")
    
    S_data_labels   .set_xscale("log")
    S_data_labels   .set_yscale("log")
    
    N_data_N_trained.set_xscale("log")
    N_data_N_trained.set_yscale("log")
    
    S_data_N_trained.set_xscale("log")
    S_data_N_trained.set_yscale("log")
    
    N_data_S_trained.set_xscale("log")
    N_data_S_trained.set_yscale("log")
    
    S_data_S_trained.set_xscale("log")
    S_data_S_trained.set_yscale("log")
    
    # Add axis lables
    N_data_S_trained.set_xlabel("EXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    S_data_S_trained.set_xlabel("EXT_RATE_PN (counts/s)", fontsize = 'xx-large')
    
    N_data_labels   .set_ylabel("Initial XAMIN labels \n\nPNT_RATE_PN (counts/s)", fontsize = 'large')
    N_data_N_trained.set_ylabel("GP trained on XXL North\n\nPNT_RATE_PN (counts/s)", fontsize = 'large')
    N_data_S_trained.set_ylabel("GP trained on XXL South\n\nPNT_RATE_PN (counts/s)", fontsize = 'large')
    
    # Title columns
    N_data_labels.set_title('North XXL catalogue', fontsize = 'xx-large')
    S_data_labels.set_title('South XXL catalogue', fontsize = 'xx-large')
    
    # Remove ticks
    S_data_labels   .tick_params(which = 'both', left = False)
    S_data_N_trained.tick_params(which = 'both', left = False)
    S_data_S_trained.tick_params(which = 'both', left = False)
    
    # add colourbar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax = (colour_bar_2, colour_bar_3), fraction = 1, shrink = 1, pad = 0.5, anchor = (0,0))
    
    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')
    
    # set tick paramiters
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.xaxis.set_major_locator(locmaj)
    S_data_S_trained.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.xaxis.set_minor_locator(locmin)
    S_data_S_trained.xaxis.set_minor_locator(locmin)
    
    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.yaxis.set_major_locator(locmaj)
    S_data_S_trained.yaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.yaxis.set_minor_locator(locmin)
    S_data_S_trained.yaxis.set_minor_locator(locmin)
    
    # set limits
    N_data_N_trained.set_xlim(PNT_RATE_PN_lim)
    N_data_N_trained.set_ylim(EXT_RATE_PN_lim)
    
    # set plot size
    plt.gcf().set_size_inches(12,10)
    
    plt.show()
    
def Plot_EXT_RATE_MOS_vs_PNT_RATE_MOS_comp(C1_data_N, C2_data_N, Non_C1C2_data_N, C1_data_S, C2_data_S, Non_C1C2_data_S):
    
    fig, ((N_data_labels, S_data_labels, colour_bar_1), (N_data_N_trained, S_data_N_trained, colour_bar_2), (N_data_S_trained, S_data_S_trained, colour_bar_3)) = plt.subplots(3,3, sharex=True, sharey=True, gridspec_kw={'width_ratios': [1, 1, 0.1]})
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    
    # Set clour bar sections invisible
    colour_bar_1.set_visible(False)
    colour_bar_2.set_visible(False)
    colour_bar_3.set_visible(False)
    
    # plot one to one line
    N_data_labels.plot(EXT_RATE_MOS_lim, EXT_RATE_MOS_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    S_data_labels.plot(EXT_RATE_MOS_lim, EXT_RATE_MOS_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    
    N_data_N_trained.plot(EXT_RATE_MOS_lim, EXT_RATE_MOS_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    N_data_S_trained.plot(EXT_RATE_MOS_lim, EXT_RATE_MOS_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    S_data_N_trained.plot(EXT_RATE_MOS_lim, EXT_RATE_MOS_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    S_data_S_trained.plot(EXT_RATE_MOS_lim, EXT_RATE_MOS_lim, c = 'black', linestyle='-', linewidth = Sample_line_width, zorder = 4)
    
    # plot initial state
    N_data_labels.scatter(Non_C1C2_data_N['EXT_RATE_MOS'], Non_C1C2_data_N['PNT_RATE_MOS'], c = 'green', alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_labels.scatter(Non_C1C2_data_S['EXT_RATE_MOS'], Non_C1C2_data_S['PNT_RATE_MOS'], c = 'green', alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    N_data_labels.scatter(C2_data_N['EXT_RATE_MOS'], C2_data_N['PNT_RATE_MOS'], c = 'orange', alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_labels.scatter(C2_data_S['EXT_RATE_MOS'], C2_data_S['PNT_RATE_MOS'], c = 'orange', alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    N_data_labels.scatter(C1_data_N['EXT_RATE_MOS'], C1_data_N['PNT_RATE_MOS'], c = 'blue', alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_labels.scatter(C1_data_S['EXT_RATE_MOS'], C1_data_S['PNT_RATE_MOS'], c = 'blue', alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    
    # -- Plot all North trained data --
    
    # order based on North trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean')
    C1_data_S = C1_data_S.sort_values('Conf mean north trained')
    C2_data_N = C2_data_N.sort_values('Conf mean')
    C2_data_S = C2_data_S.sort_values('Conf mean north trained')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean north trained')
    
    # Plot non-C1C2 data
    N_data_N_trained.scatter(Non_C1C2_data_N['EXT_RATE_MOS'], Non_C1C2_data_N['PNT_RATE_MOS'], c = Non_C1C2_data_N['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(Non_C1C2_data_S['EXT_RATE_MOS'], Non_C1C2_data_S['PNT_RATE_MOS'], c = Non_C1C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_N_trained.scatter(C2_data_N['EXT_RATE_MOS'], C2_data_N['PNT_RATE_MOS'], c = C2_data_N['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C2_data_S['EXT_RATE_MOS'], C2_data_S['PNT_RATE_MOS'], c = C2_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_N_trained.scatter(C1_data_N['EXT_RATE_MOS'], C1_data_N['PNT_RATE_MOS'], c = C1_data_N['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_N_trained.scatter(C1_data_S['EXT_RATE_MOS'], C1_data_S['PNT_RATE_MOS'], c = C1_data_S['Conf mean north trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    # -- Plot all South trained data --
    
    # order based on South trained confidence
    C1_data_N = C1_data_N.sort_values('Conf mean south trained')
    C1_data_S = C1_data_S.sort_values('Conf mean')
    C2_data_N = C2_data_N.sort_values('Conf mean south trained')
    C2_data_S = C2_data_S.sort_values('Conf mean')
    Non_C1C2_data_N = Non_C1C2_data_N.sort_values('Conf mean south trained')
    Non_C1C2_data_S = Non_C1C2_data_S.sort_values('Conf mean')
    
    # Plot non-C1C2 data
    N_data_S_trained.scatter(Non_C1C2_data_N['EXT_RATE_MOS'], Non_C1C2_data_N['PNT_RATE_MOS'], c = Non_C1C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(Non_C1C2_data_S['EXT_RATE_MOS'], Non_C1C2_data_S['PNT_RATE_MOS'], c = Non_C1C2_data_S['Conf mean'], alpha = 1, s = 70, marker = 'o', edgecolors='none', zorder = 1, vmin = 0, vmax = 1)
    
    # Plot C2 data
    N_data_S_trained.scatter(C2_data_N['EXT_RATE_MOS'], C2_data_N['PNT_RATE_MOS'], c = C2_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C2_data_S['EXT_RATE_MOS'], C2_data_S['PNT_RATE_MOS'], c = C2_data_S['Conf mean']              , alpha = 1, s = 70, marker = '^', edgecolors='none', zorder = 2, vmin = 0, vmax = 1)
    
    # Plot C1 data
    N_data_S_trained.scatter(C1_data_N['EXT_RATE_MOS'], C1_data_N['PNT_RATE_MOS'], c = C1_data_N['Conf mean south trained'], alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    S_data_S_trained.scatter(C1_data_S['EXT_RATE_MOS'], C1_data_S['PNT_RATE_MOS'], c = C1_data_S['Conf mean']              , alpha = 1, s = 70, marker = 'v', edgecolors='none', zorder = 3, vmin = 0, vmax = 1)
    
    
    # -- manage axies --
    
    # log axies
    N_data_labels   .set_xscale("log")
    N_data_labels   .set_yscale("log")
    
    S_data_labels   .set_xscale("log")
    S_data_labels   .set_yscale("log")
    
    N_data_N_trained.set_xscale("log")
    N_data_N_trained.set_yscale("log")
    
    S_data_N_trained.set_xscale("log")
    S_data_N_trained.set_yscale("log")
    
    N_data_S_trained.set_xscale("log")
    N_data_S_trained.set_yscale("log")
    
    S_data_S_trained.set_xscale("log")
    S_data_S_trained.set_yscale("log")
    
    # Add axis lables
    N_data_S_trained.set_xlabel("EXT_RATE_MOS (counts/s)", fontsize = 'xx-large')
    S_data_S_trained.set_xlabel("EXT_RATE_MOS (counts/s)", fontsize = 'xx-large')
    
    N_data_labels   .set_ylabel("Initial XAMIN labels \n\nPNT_RATE_MOS (counts/s)", fontsize = 'large')
    N_data_N_trained.set_ylabel("GP trained on XXL North\n\nPNT_RATE_MOS (counts/s)", fontsize = 'large')
    N_data_S_trained.set_ylabel("GP trained on XXL South\n\nPNT_RATE_MOS (counts/s)", fontsize = 'large')
    
    # Title columns
    N_data_labels.set_title('North XXL catalogue', fontsize = 'xx-large')
    S_data_labels.set_title('South XXL catalogue', fontsize = 'xx-large')
    
    # Remove ticks
    S_data_labels   .tick_params(which = 'both', left = False)
    S_data_N_trained.tick_params(which = 'both', left = False)
    S_data_S_trained.tick_params(which = 'both', left = False)
    
    # add colourbar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax = (colour_bar_2, colour_bar_3), fraction = 1, shrink = 1, pad = 0.5, anchor = (0,0))
    
    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')
    
    # set tick paramiters
    
    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.xaxis.set_major_locator(locmaj)
    S_data_S_trained.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.xaxis.set_minor_locator(locmin)
    S_data_S_trained.xaxis.set_minor_locator(locmin)
    
    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 17)
    N_data_S_trained.yaxis.set_major_locator(locmaj)
    S_data_S_trained.yaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    N_data_S_trained.yaxis.set_minor_locator(locmin)
    S_data_S_trained.yaxis.set_minor_locator(locmin)
    
    # set limits
    N_data_N_trained.set_xlim(PNT_RATE_MOS_lim)
    N_data_N_trained.set_ylim(EXT_RATE_MOS_lim)
    
    # set plot size
    plt.gcf().set_size_inches(12,10)
    
    plt.show()

Plot_EXT_RATE_PN_vs_PNT_RATE_PN(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, colour_by_conf = False)
Plot_EXT_RATE_PN_vs_PNT_RATE_PN(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, colour_by_conf = True)

# Plot_initial_and_conf_EXT_RATE_PN_vs_PNT_RATE_PN(Data_N_C1, Data_N_C2, Data_N_Non_C1C2)


# Plot_EXT_RATE_PN_vs_PNT_RATE_PN_comp(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)

# Plot_EXT_RATE_MOS_vs_PNT_RATE_MOS_comp(Data_N_C1, Data_N_C2, Data_N_Non_C1C2, Data_S_C1, Data_S_C2, Data_S_Non_C1C2)















