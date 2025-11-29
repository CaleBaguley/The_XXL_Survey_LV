import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy as cp
import scipy.stats as stats
import matplotlib.ticker as ticker
from matplotlib import cm

# sample cuts
corerad_cut = 5
extlike_C1_cut = 33
extlike_C2_cut = 15

Sample_line_width = 1

# source property plotting limits
EXT_lim         = (0.8*10**(-3), 2.0*10**3)
EXT_LIKE_lim    = (0.8*10**(-4), 2.0*10**4)

# -- Load and match data ------------------------------------------------------------------------------------------------------------

# load north and south data from gp trained on each field
North_trained = pd.read_csv("Sample_opposing_field/BG_Rate/North/10Iters_ARD43n_Pointings_Cleaned_Training_Results.csv")

# reduce columns
North_trained = North_trained[["id","Conf mean","Conf std"]]
# Load source table to get C1C2 labels
North_XXL = pd.read_csv("../../../../Data/PipelineV4.3/XXLn_Reduced_Cols_Pointings_Cleaned_03.csv")[["Index", "C1C2", "EXT", "EXT_LIKE"]]

# Set all values of EXT_LIKE below 10^-4 to 10^-4
North_XXL['EXT_LIKE'] = North_XXL['EXT_LIKE'].apply(lambda x: 10**(-4) if x < 10**(-4) else x)

# Match by source id
North = pd.merge(North_trained, North_XXL, left_on = "id", right_on = "Index")

# Split data into C1 C2 and non-C1C2 for plotting
North_C1       = North.loc[North["C1C2"] == 1]
North_C2       = North.loc[North["C1C2"] == 2]
North_Non_C1C2 = North.loc[North["C1C2"] == 0]

# Load camira and Gama matched sources
Camira = pd.read_csv("../../../../Data/PipelineV4.3/Camira_Matched_XXLn_Redused_Cols_Pointings_Cleaned_03.csv")[["Index","Separation","N_mem"]]
Gama = pd.read_csv("../../../../Data/PipelineV4.3/Gama_Matched_XXLn_Redused_Cols_Pointings_Cleaned_03.csv")[["Index","Separation","N_mem","Nfof"]]

#Camira = Camira.loc[Camira["N_mem"] > 15]

# Only consider sources within 15"
Camira = Camira.loc[Camira["Separation"] < 15]
Gama = Gama.loc[Gama["Separation"] < 15]
Gama = Gama.loc[Gama["Nfof"] >= 10]

# Merge with data
def merge_with_data(Cat, C1, C2, Non_C1C2, Full_cat):
    matched_C1 = pd.merge(C1, Cat, left_on="id", right_on="Index")
    matched_C2 = pd.merge(C2, Cat, left_on="id", right_on="Index")
    matched_C1C2 = pd.merge(Non_C1C2, Cat, left_on="id", right_on="Index")

    matched_full = pd.merge(Full_cat, Cat, left_on="id", right_on="Index")

    return matched_C1, matched_C2, matched_C1C2, matched_full

Camira_North_C1, Camira_North_C2, Camira_North_Non_C1C2, Camira_North = merge_with_data(Camira,
                                                                                North_C1,
                                                                                North_C2,
                                                                                North_Non_C1C2,
                                                                                North)

Gama_North_C1, Gama_North_C2, Gama_North_Non_C1C2, Gama_North = merge_with_data(Gama,
                                                                                North_C1,
                                                                                North_C2,
                                                                                North_Non_C1C2,
                                                                                North)


# -- C1C2 content -------------------------------------------------------------------------------------------------------------------

print("-- Camira --")
print("Camira C1 count:"      , len(Camira_North_C1))
print("Camira C2 count:"      , len(Camira_North_C2))
print("Camira non-C1C2 count:", len(Camira_North_Non_C1C2))
print("Camira total count:"   , len(Camira_North))

print("-- Gama --")
print("Gama C1 count:"      , len(Gama_North_C1))
print("Gama C2 count:"      , len(Gama_North_C2))
print("Gama non-C1C2 count:", len(Gama_North_Non_C1C2))
print("Gama total count:"   , len(Gama_North))

print("-- XXL North --")
print("C1 count:", len(North_C1))
print("C2 count:", len(North_C2))
print("non-C1C2 count:", len(North_Non_C1C2))
print("XXL total:", len(North))

# calculate probability of getting equal to or more C1 or C2 sources in a sample.

print("\n-- CAMIRA --")
C1_prob = 1 - stats.binom.cdf(len(Camira_North_C1) - 1,len(Camira_North), len(North_C1)/len(North))
C2_prob = 1 - stats.binom.cdf(len(Camira_North_C2) - 1,len(Camira_North), len(North_C2)/len(North))
C1C2_prob = 1 - stats.binom.cdf(len(Camira_North_C1) + len(Camira_North_C2) - 1,len(Camira_North), (len(North_C1) + len(North_C2))/len(North))
print(f"Probability of >= C1 count ({len(Camira_North_C1)}) = {C1_prob:e}")
print(f"Probability of >= C2 count ({len(Camira_North_C2)}) = {C2_prob:e}")
print(f"Probability of >= C1C2 count ({len(Camira_North_C1) + len(Camira_North_C2)}) = {C1C2_prob:e}")


print("-- Gama --")
C1_prob = 1 - stats.binom.cdf(len(Gama_North_C1) - 1,len(Gama_North), len(North_C1)/len(North))
C2_prob = 1 - stats.binom.cdf(len(Gama_North_C2) - 1,len(Gama_North), len(North_C2)/len(North))
C1C2_prob = 1 - stats.binom.cdf(len(Gama_North_C1) + len(Gama_North_C2) - 1,len(Gama_North), (len(North_C1) + len(North_C2))/len(North))
print(f"Probability of >= C1 count ({len(Gama_North_C1)}) = {C1_prob:e}")
print(f"Probability of >= C2 count ({len(Gama_North_C2)}) = {C2_prob:e}")
print(f"Probability of >= C1C2 count ({len(Gama_North_C1) + len(Camira_North_C2)}) = {C1C2_prob:e}")


# --- Coreradius vs extentliklyhood confidence distribution camira matched---------------------------------------------------

def plot_ext_extlike(Matched_North_C1_camira, Matched_North_C2_camira, Matched_North_Non_C1C2_camira,
                     Matched_North_C1_gama,   Matched_North_C2_gama,   Matched_North_Non_C1C2_gama):

    fig, ((camira_axis, cbar_axis_1), (gama_axis, cbar_axis_2)) =\
        plt.subplots(2,2, sharex=True, sharey=True, gridspec_kw = {'width_ratios' : [1, 0.1]})

    cbar_axis_1.set_visible(False)
    cbar_axis_2.set_visible(False)

    plt.subplots_adjust(wspace = 0, hspace = 0)


    # -- Plot all data --
    camira_axis.scatter(North_Non_C1C2['EXT_LIKE'], North_Non_C1C2['EXT'], c = North_Non_C1C2['Conf mean'],
                        edgecolors = 'none', alpha = 0.1, s = 70, zorder = 1, marker = 'o')
    camira_axis.scatter(North_C2      ['EXT_LIKE'], North_C2      ['EXT'], c = North_C2      ['Conf mean'],
                        edgecolors = 'none', alpha = 0.1, s = 70, zorder = 2, marker = '^')
    camira_axis.scatter(North_C1      ['EXT_LIKE'], North_C1      ['EXT'], c = North_C1      ['Conf mean'],
                        edgecolors = 'none', alpha = 0.1, s = 70, zorder = 3, marker = 'v')

    gama_axis.scatter(North_Non_C1C2['EXT_LIKE'], North_Non_C1C2['EXT'], c = North_Non_C1C2['Conf mean'],
                      edgecolors = 'none', alpha = 0.1, s = 70, zorder = 1, marker = 'o')
    gama_axis.scatter(North_C2['EXT_LIKE'], North_C2['EXT'], c = North_C2['Conf mean'],
                      edgecolors = 'none', alpha = 0.1, s = 70, zorder = 2, marker = '^')
    gama_axis.scatter(North_C1['EXT_LIKE'], North_C1['EXT'], c = North_C1['Conf mean'],
                      edgecolors = 'none', alpha = 0.1, s = 70, zorder = 3, marker = 'v')

    # -- plot matched on top with ring --
    camira_axis.scatter(Matched_North_Non_C1C2_camira['EXT_LIKE'], Matched_North_Non_C1C2_camira['EXT'],
                        c = Matched_North_Non_C1C2_camira['Conf mean'], edgecolors='black', alpha = 1, s = 70,
                        zorder = 4, marker ='o')
    camira_axis.scatter(Matched_North_C2_camira      ['EXT_LIKE'], Matched_North_C2_camira      ['EXT'],
                        c = Matched_North_C2_camira      ['Conf mean'], edgecolors='black', alpha = 1, s = 70,
                        zorder = 5, marker ='^')
    camira_axis.scatter(Matched_North_C1_camira      ['EXT_LIKE'], Matched_North_C1_camira      ['EXT'],
                        c = Matched_North_C1_camira      ['Conf mean'], edgecolors='black', alpha = 1, s = 70,
                        zorder = 6, marker ='v')

    gama_axis.scatter(Matched_North_Non_C1C2_gama['EXT_LIKE'], Matched_North_Non_C1C2_gama['EXT'],
                        c = Matched_North_Non_C1C2_gama['Conf mean'], edgecolors='grey', alpha = 1, s = 70,
                        zorder = 4, marker ='o')
    gama_axis.scatter(Matched_North_C2_gama      ['EXT_LIKE'], Matched_North_C2_gama      ['EXT'],
                        c = Matched_North_C2_gama      ['Conf mean'], edgecolors='grey', alpha = 1, s = 70,
                        zorder = 5, marker ='^')
    gama_axis.scatter(Matched_North_C1_gama      ['EXT_LIKE'], Matched_North_C1_gama      ['EXT'],
                        c = Matched_North_C1_gama      ['Conf mean'], edgecolors='grey', alpha = 1, s = 70,
                        zorder = 6, marker ='v')

    # -- label axise --
    gama_axis  .set_xlabel('EXT_LIKE'        , fontsize='xx-large')

    camira_axis.set_ylabel('EXT (arcseconds)', fontsize='xx-large')
    gama_axis  .set_ylabel('EXT (arcseconds)', fontsize='xx-large')

    camira_axis.set_xscale("log")
    camira_axis.set_yscale("log")

    # set y ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 16)
    camira_axis.yaxis.set_major_locator(locmaj)
    gama_axis  .yaxis.set_major_locator(locmaj)

    # add minor ticks to y
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    camira_axis.yaxis.set_minor_locator(locmin)
    gama_axis  .yaxis.set_minor_locator(locmin)

    # set x ticks to every order of magnitude
    locmaj = ticker.LogLocator(base = 10.0, subs = (1.0, ), numticks = 16)
    gama_axis.xaxis.set_major_locator(locmaj)

    # add minor ticks to x
    locmin = ticker.LogLocator(base = 10.0, subs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, ), numticks = 16)
    gama_axis.xaxis.set_minor_locator(locmin)

    # plot C1C2 cuts
    # EXT cut
    camira_axis.plot(EXT_LIKE_lim, (corerad_cut, corerad_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    gama_axis  .plot(EXT_LIKE_lim, (corerad_cut, corerad_cut), c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    # EXT_LIKE cuts
    camira_axis.plot((extlike_C1_cut, extlike_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    gama_axis  .plot((extlike_C1_cut, extlike_C1_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    camira_axis.plot((extlike_C2_cut, extlike_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)
    gama_axis  .plot((extlike_C2_cut, extlike_C2_cut), EXT_lim, c = 'black', linestyle='dashed', linewidth = Sample_line_width, zorder = 0)

    # set size of tick lables
    camira_axis.tick_params(axis='both', which='major', labelsize = 'x-large')
    gama_axis  .tick_params(axis='both', which='major', labelsize = 'x-large')

    # Combine colour bar axes.
    gs = fig.add_gridspec(2, 2)
    cbar_axis = fig.add_subplot(gs[:, 1])
    cbar_axis.set_visible(False)

    # add colourbar
    cmap = plt.get_cmap("viridis")
    sm = cm.ScalarMappable(plt.Normalize(0,1), cmap = cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax = cbar_axis, shrink = 1)

    # lable colourbar
    cbar.set_label('Confidence Value', size = 'xx-large')
    cbar.ax.tick_params(labelsize = 'x-large')


    # Set limits
    camira_axis.set_xlim(EXT_LIKE_lim[0], EXT_LIKE_lim[1])
    camira_axis.set_ylim(EXT_lim[0], EXT_lim[1])

    # label each subplot
    # Add text to top left corner
    camira_axis.text(10**-4, 0.9*10**3, "Camira", size = 'xx-large')
    gama_axis  .text(10**-4, 0.9*10**3, "Gama",   size = 'xx-large')

    # set plot size
    fig.set_size_inches(7,10)

    # Adjust padding
    plt.subplots_adjust(left=0.2, right=0.85, top=0.95, bottom=0.1)

    plt.savefig("camira_gama_EXT_vs_EXT_LIKE.png", dpi = 300)

    plt.show()

plot_ext_extlike(Camira_North_C1, Camira_North_C2, Camira_North_Non_C1C2,
                 Gama_North_C1,   Gama_North_C2,   Gama_North_Non_C1C2)

# -- Confidence distribution -------------------------------------------------------------------------------------------------------

bins = np.linspace(-0.05,1,22)

# -- non C1C2 --
conf_dist_North, bin_edges = np.histogram(North_Non_C1C2['Conf mean'], bins = bins)
Camira_matched_conf_dist_North, bin_edges = np.histogram(Camira_North_Non_C1C2['Conf mean'], bins = bins)
Gama_matched_conf_dist_North, bin_edges = np.histogram(Gama_North_Non_C1C2['Conf mean'], bins = bins)

#scale North distribution to camira matched distributions
scale_Camira = len(Camira_North_Non_C1C2) / len(North_Non_C1C2)
scale_Gama = len(Gama_North_Non_C1C2) / len(North_Non_C1C2)

# -- compare fraction of camira matched > 0.1 confidence --
North_num_above_0_1 = len(np.where(North_Non_C1C2['Conf mean']>0.1)[0])
North_num_camira_above_0_1 = len(np.where(Camira_North_Non_C1C2['Conf mean']>0.1)[0])
North_num_gama_above_0_1 = len(np.where(Gama_North_Non_C1C2['Conf mean']>0.1)[0])

North_p_above_0_1 = North_num_above_0_1/len(North_Non_C1C2)
North_p_camira_above_0_1 = North_num_camira_above_0_1/len(Camira_North_Non_C1C2)
North_p_gama_above_0_1 = North_num_gama_above_0_1/len(Gama_North_Non_C1C2)

# -- calcualte the expected number of sources with confidence above 0.1 --
North_expected_num_camira_above_0_1 = scale_Camira * North_num_above_0_1
North_expected_num_gama_above_0_1 = scale_Gama * North_num_above_0_1

print("\n -- Counts --")
print("Number of non C1/C2 with confidence above 0.1: North {}".format(North_num_above_0_1))
print("non C1/C2 fraction with confidence above 0.1: North {}".format(North_p_above_0_1))
print("\nCamira:")
print("Expected number of Camira matched non-C1C2 sources with confidence above 0.1: North {}".format(North_expected_num_camira_above_0_1))
print("Observed number of Camira matched non-C1C2 sources with confidence above 0.1: North {}".format(North_num_camira_above_0_1))
print("Camira matched non C1/C2 fraction with confidence above 0.1: North {}".format(North_p_camira_above_0_1))
print("\nGama:")
print("Expected number of Gama matched non-C1C2 sources with confidence above 0.1: North {}".format(North_expected_num_gama_above_0_1))
print("Observed number of Gama matched non-C1C2 sources with confidence above 0.1: North {}".format(North_num_gama_above_0_1))
print("Gama matched non C1/C2 fraction with confidence above 0.1: North {}".format(North_p_gama_above_0_1))

# calc proability of a random sample with the same number of or more sources with confidence over 0.1 than the camira sample
print("\n -- Probability --")
print("Camira")
North_prob = 1 - stats.binom.cdf(North_num_camira_above_0_1 - 1,len(Camira_North_Non_C1C2['Conf mean']), North_p_above_0_1)
print("Probability of {} or more sources with confidence over 0.1 from a random sample of {} sources trained on the North is: {}".format(North_num_camira_above_0_1, len(Camira_North_Non_C1C2['Conf mean']), North_prob))

print("Gama")
North_prob = 1 - stats.binom.cdf(North_num_gama_above_0_1 - 1,len(Gama_North_Non_C1C2['Conf mean']), North_p_above_0_1)
print("Probability of {} or more sources with confidence over 0.1 from a random sample of {} sources trained on the North is: {}".format(North_num_gama_above_0_1, len(Gama_North_Non_C1C2['Conf mean']), North_prob))

print("\n")
# -- Plot cumulative for non-C1C2 --

conf_dist_cumulative_North = cp.deepcopy(conf_dist_North)
Camira_matched_conf_dist_cumulative_North = cp.deepcopy(Camira_matched_conf_dist_North)
Gama_matched_conf_dist_cumulative_North = cp.deepcopy(Gama_matched_conf_dist_North)

for i in range(1,len(conf_dist_cumulative_North)):
    conf_dist_cumulative_North[i] += conf_dist_cumulative_North[i-1]
    Camira_matched_conf_dist_cumulative_North[i] += Camira_matched_conf_dist_cumulative_North[i-1]
    Gama_matched_conf_dist_cumulative_North[i] += Gama_matched_conf_dist_cumulative_North[i-1]

North_Non_C1C2       .sort_values(by = ['Conf mean'], inplace = True)
Camira_North_Non_C1C2.sort_values(by = ['Conf mean'], inplace = True)
Gama_North_Non_C1C2  .sort_values(by = ['Conf mean'], inplace = True)

# -- plot confidence distribution with inlaide cumulative plot --

fig, (North_axis) = plt.subplots(1, 1, sharex = True)

plt.subplots_adjust(wspace = 0, hspace = 0)

# plot distribution first
North_axis.fill_between(bin_edges[1:], conf_dist_North / len(North_Non_C1C2), step='pre', fc ='none', edgecolor='blue', hatch ='-', linewidth = 2)
North_axis.fill_between(bin_edges[1:], Camira_matched_conf_dist_North / len(Camira_North_Non_C1C2), step='pre', fc = 'none', edgecolor='black', hatch = '//', linewidth = 2)
North_axis.fill_between(bin_edges[1:], Gama_matched_conf_dist_North / len(Gama_North_Non_C1C2), step='pre', fc = 'none', edgecolor='grey', hatch = '\ \ ', linewidth = 2)

# Set limits
North_axis.set_ylim(0, 0.8)
# Label
North_axis.set_ylabel('Count', fontsize = 'xx-large')
North_axis.set_xlabel('Confidence value', fontsize = 'xx-large')
# set size of tick lables
North_axis.tick_params(axis='both', which='major', labelsize = 'large')

# create inset axis
North_inset_axis = North_axis.inset_axes([0.3,0.3,0.6,0.6])

# plot cumulative data
North_inset_axis.step(North_Non_C1C2       ['Conf mean'], [i/(len(North_Non_C1C2) + 1) for i in range(1, len(North_Non_C1C2) + 1)], c ='blue')
North_inset_axis.step(Camira_North_Non_C1C2['Conf mean'], [i/(len(Camira_North_Non_C1C2) + 1) for i in range(1, len(Camira_North_Non_C1C2)+1)], c = 'black')
North_inset_axis.step(Gama_North_Non_C1C2  ['Conf mean'], [i/(len(Gama_North_Non_C1C2) + 1) for i in range(1, len(Gama_North_Non_C1C2)+1)], c = 'grey')

# Set x ticks
North_inset_axis.set_xticks([0.1*i for i in range(10)])

# Set limits
North_inset_axis.set_xlim(0,   0.85)

North_inset_axis.set_ylim(0,  1.05)

# Label axies
North_inset_axis.set_ylabel('Cumulative count', fontsize = 'x-large')

North_inset_axis.set_xlabel('Confidence value', fontsize = 'x-large')

# set plot size
plt.gcf().set_size_inches(8,5)
    
# Adjust padding
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)

plt.savefig("Conf_dist_camira_gama.png", dpi = 300)

plt.show()
