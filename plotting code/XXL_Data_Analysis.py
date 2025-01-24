import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy as cp
import scipy.stats as stats
from matplotlib import cm

# --- Data setup -------------------------------------------------------------------------------------------------

Data = pd.read_csv("../../../Data/PipelineV4.3/XXLn_Reduced_Cols.csv")
DataConf = pd.read_csv("Sample simdata/North/10Iters_ADRn_training_Results.csv")
Relevance = pd.read_csv("Data_runs/No_SCTS/North/Relevance.csv") #pd.read_csv("ARD43n_Relevence.csv")
Camira_matched = pd.read_csv("Camira_matched.csv")

All_Cols = Data.keys()

# top n most relevent columns to be displayed
Num_Cols = 8

# Reduce number of columns in data to top 5 most relevent, id, corerad and detlike_ext
Relevance.sort_values(['Mean'], ascending=False, inplace=True)

cols = []

cols.append('index')
cols.append('EXT')
cols.append('EXT_LIKE')

try:
    Relevant_Cols = Relevance['Name'].head(Num_Cols)
except:
    Relevant_Cols_id = Relevance['id'].head(Num_Cols)
    Relevant_Cols = All_Cols[Relevant_Cols_id]

for col in Relevant_Cols:
    cols.append(col)

print(Data.columns)

Data = Data[cols]

# Reduce DataConf to id, mean and std
DataConf = DataConf[['id', 'Conf mean', 'Conf std']]

# Match data sets
Data = Data.merge(DataConf, left_on='index', right_on='id')

# drop label col from list
cols = cols[1:]

# order based on confidence
Data = Data.sort_values('Conf mean')

# add sample labels
corerad_cut = 5
extlike_C1_cut = 33
extlike_C2_cut = 15

sample = [0 for i in range(len(Data))]
C1_arg = []
C2_arg = []
else_arg = []
C1C2_arg = []

Camira_matched_arg = []
Camira_matched_C1_arg = []
Camira_matched_C2_arg = []
Camira_matched_else_arg = []

for i in range(len(sample)):
    
    if(Data['EXT'][i] > corerad_cut):
        if(Data['EXT_LIKE'][i] > extlike_C1_cut):
            sample[i] = 1
            C1_arg.append(i)
            C1C2_arg.append(i)
            
            if(np.isin(Data['index'][i], Camira_matched['XMM index'])):
                Camira_matched_arg.append(i)
                Camira_matched_C1_arg.append(i)
                
        elif(Data['EXT_LIKE'][i] > extlike_C2_cut):
            sample[i] = 2
            C2_arg.append(i)
            C1C2_arg.append(i)
            
            if(np.isin(Data['index'][i], Camira_matched['XMM index'])):
                Camira_matched_arg.append(i)
                Camira_matched_C2_arg.append(i)
            
        else:
            else_arg.append(i)
            
            if(np.isin(Data['index'][i], Camira_matched['XMM index'])):
                Camira_matched_arg.append(i)
                Camira_matched_else_arg.append(i)
            
    else:
        else_arg.append(i)
        
        if(np.isin(Data['index'][i], Camira_matched['XMM index'])):
            Camira_matched_arg.append(i)
            Camira_matched_else_arg.append(i)

else_high_conf_arg = []

for i in range(len(else_arg)):
    if(Data['Conf mean'][else_arg[i]]>0.15):
        else_high_conf_arg.append(else_arg[i])

Data['sample'] = sample

# --- Coreradius vs extentliklyhood cluster samples ---------------------------------------------------------------------------

plt.scatter(Data['EXT'][C1_arg], Data['EXT_LIKE'][C1_arg], alpha = 0.5, label = 'C1 sources')
plt.scatter(Data['EXT'][C2_arg], Data['EXT_LIKE'][C2_arg], alpha = 0.5, label = 'C2 sources')
plt.scatter(Data['EXT'][else_arg], Data['EXT_LIKE'][else_arg], alpha = 0.5, label = 'Non C1 or C2 sources')

# -- limits -- 
min_corerad = max(min(Data['EXT']), 10**(-4))
max_corerad = max(Data['EXT'])

min_extlike = max(min(Data['EXT_LIKE']), 10**(-4))
max_extlike = max(Data['EXT_LIKE'])

plt.xlim(min_corerad, max_corerad)
plt.ylim(min_extlike, max_extlike)

# -- sample dividing lines --
plt.plot([corerad_cut, corerad_cut], [min_extlike, max_extlike], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C1_cut, extlike_C1_cut], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C2_cut, extlike_C2_cut], c = 'black', linestyle='dashed')

# -- label axise --
plt.xlabel('EXT', fontsize=20)
plt.ylabel('EXT_LIKE', fontsize=20)

plt.loglog()

plt.legend(fontsize=20)

plt.show()

# --- Coreradius vs extentliklyhood confidence distribution ---------------------------------------------------------------------

plt.scatter(Data['EXT'], Data['EXT_LIKE'], c = Data['Conf mean'], alpha = 0.5)

# -- colour bar --
plt.clim(0,1)
cbar = plt.colorbar()
cbar.set_label('Confidence', fontsize=20)

# -- limits -- 
min_corerad = max(min(Data['EXT']), 10**(-4))
max_corerad = max(Data['EXT'])

min_extlike = max(min(Data['EXT_LIKE']), 10**(-4))
max_extlike = max(Data['EXT_LIKE'])

plt.xlim(min_corerad, max_corerad)
plt.ylim(min_extlike, max_extlike)

# -- sample dividing lines --
plt.plot([corerad_cut, corerad_cut], [min_extlike, max_extlike], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C1_cut, extlike_C1_cut], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C2_cut, extlike_C2_cut], c = 'black', linestyle='dashed')

# -- label axise --
plt.xlabel('EXT', fontsize=20)
plt.ylabel('EXT_LIKE', fontsize=20)

plt.loglog()

plt.show()

# --- Coreradius vs extentliklyhood source density --------------------------------------------------------------------------

bins = 100

# -- limits -- 
min_corerad = max(min(Data['EXT']), 10**(-4))
max_corerad = max(Data['EXT'])

min_extlike = max(min(Data['EXT_LIKE']), 10**(-4))
max_extlike = max(Data['EXT_LIKE'])

# Calculate bin edges
xbins = 10**np.linspace(np.log(min_corerad), np.log(max_corerad), bins)
ybins = 10**np.linspace(np.log(min_extlike), np.log(max_extlike), bins)

# Calculate the mean confidence
count, xedges, yedges, binnumber = stats.binned_statistic_2d(Data['EXT'], Data['EXT_LIKE'], Data['Conf mean'], statistic='count', bins = [xbins, ybins])

X, Y = np.meshgrid(xedges, yedges)

# Plot scatter
image = plt.pcolormesh(X, Y, count.T, cmap = 'Greys', norm = colors.LogNorm(vmin = max(count.min(),0.5), vmax = 10**3))

# -- colour bar --
cbar = plt.colorbar()
cbar.set_label('Count', fontsize=20)

# set limits
plt.xlim(min_corerad, max_corerad)
plt.ylim(min_extlike, max_extlike)

# -- sample dividing lines --
plt.plot([corerad_cut, corerad_cut], [min_extlike, max_extlike], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C1_cut, extlike_C1_cut], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C2_cut, extlike_C2_cut], c = 'black', linestyle='dashed')

# -- label axise --
plt.xlabel('EXT', fontsize=20)
plt.ylabel('EXT_LIKE', fontsize=20)

plt.loglog()

plt.show()

# --- Coreradius vs extentliklyhood mean confidence distribution ------------------------------------------------------------

bins = 100

# -- limits -- 
min_corerad = max(min(Data['EXT']), 10**(-4))
max_corerad = max(Data['EXT'])

min_extlike = max(min(Data['EXT_LIKE']), 10**(-4))
max_extlike = max(Data['EXT_LIKE'])

# Calculate bin edges
xbins = 10**np.linspace(np.log(min_corerad), np.log(max_corerad), bins)
ybins = 10**np.linspace(np.log(min_extlike), np.log(max_extlike), bins)

# Calculate the mean confidence
mean_conf, xedges, yedges, binnumber = stats.binned_statistic_2d(Data['EXT'], Data['EXT_LIKE'], Data['Conf mean'], statistic='mean', bins = [xbins, ybins])

X, Y = np.meshgrid(xedges, yedges)

# Plot scatter
image = plt.pcolormesh(X, Y, mean_conf.T, cmap = 'Greys')

# -- colour bar --
plt.clim(0,1)
cbar = plt.colorbar()
cbar.set_label('Mean Confidence', fontsize=20)

# set limits
plt.xlim(min_corerad, max_corerad)
plt.ylim(min_extlike, max_extlike)

# -- sample dividing lines --
plt.plot([corerad_cut, corerad_cut], [min_extlike, max_extlike], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C1_cut, extlike_C1_cut], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C2_cut, extlike_C2_cut], c = 'black', linestyle='dashed')

# -- label axise --
plt.xlabel('EXT', fontsize=20)
plt.ylabel('EXT_LIKE', fontsize=20)

plt.loglog()

plt.show()

# --- Coreradius vs extentliklyhood confidence distribution camira matched---------------------------------------------------

plt.scatter(Data['EXT'], Data['EXT_LIKE'], c = Data['Conf mean'], alpha = 0.05)

# -- plot camira matched on top with ring --
plt.scatter(Data['EXT'][Camira_matched_arg], Data['EXT_LIKE'][Camira_matched_arg], c = Data['Conf mean'][Camira_matched_arg], edgecolors='black', alpha = 0.8)


# -- colour bar --
plt.clim(0,1)
cbar = plt.colorbar()
cbar.set_label('Confidence', fontsize=20)

# -- limits -- 
min_corerad = max(min(Data['EXT']), 10**(-4))
max_corerad = max(Data['EXT'])

min_extlike = max(min(Data['EXT_LIKE']), 10**(-4))
max_extlike = max(Data['EXT_LIKE'])

plt.xlim(min_corerad, max_corerad)
plt.ylim(min_extlike, max_extlike)

# -- sample dividing lines --
plt.plot([corerad_cut, corerad_cut], [min_extlike, max_extlike], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C1_cut, extlike_C1_cut], c = 'black', linestyle='dashed')
plt.plot([min_corerad, max_corerad], [extlike_C2_cut, extlike_C2_cut], c = 'black', linestyle='dashed')

# -- label axise --
plt.xlabel('EXT', fontsize=20)
plt.ylabel('EXT_LIKE', fontsize=20)

plt.loglog()

plt.show()

# --- Confidence distribution -----------------------------------------------------------------------------------------------

bins = np.linspace(0,1,51)

total, bin_edges = np.histogram(Data['Conf mean'], bins = bins)
C1_dist, bin_edges = np.histogram(Data['Conf mean'][C1_arg], bins = bins)
C2_dist, bin_edges = np.histogram(Data['Conf mean'][C2_arg], bins = bins)
else_dist, bin_edges = np.histogram(Data['Conf mean'][else_arg], bins = bins)

plt.step(bin_edges[:-1], total, label = 'total')
plt.step(bin_edges[:-1], C1_dist, label = 'C1 objects')
plt.step(bin_edges[:-1], C2_dist, label = 'C2 objects')
plt.step(bin_edges[:-1], else_dist, label = 'non C1/C2 objects')

plt.ylabel('Count')
plt.yscale('log')

plt.xlabel('Confidence')

plt.legend()
plt.show()

# --- Confidence distribution Camira matched --------------------------------------------------------------------------------

# -- C1 --
bins = np.linspace(0,1,51)

else_dist, bin_edges = np.histogram(Data['Conf mean'][C1_arg], bins = bins)
Camira_matched_else_dist, bin_edges = np.histogram(Data['Conf mean'][Camira_matched_C1_arg], bins = bins)


plt.step(bin_edges[:-1], else_dist, label = 'C1 objects')
plt.step(bin_edges[:-1], Camira_matched_else_dist, label = 'Camira matched C1 objects')

plt.ylabel('Count')
plt.yscale('log')

plt.xlabel('Confidence')

plt.legend()
plt.show()

# -- C2 --
bins = np.linspace(0,1,51)

else_dist, bin_edges = np.histogram(Data['Conf mean'][C2_arg], bins = bins)
Camira_matched_else_dist, bin_edges = np.histogram(Data['Conf mean'][Camira_matched_C2_arg], bins = bins)


plt.step(bin_edges[:-1], else_dist, label = 'C2 objects')
plt.step(bin_edges[:-1], Camira_matched_else_dist, label = 'Camira matched C2 objects')

plt.ylabel('Count')
plt.yscale('log')

plt.xlabel('Confidence')

plt.legend()
plt.show()

# -- non C1/C2
bins = np.linspace(0,1,51)

else_dist, bin_edges = np.histogram(Data['Conf mean'][else_arg], bins = bins, density=True)
Camira_matched_else_dist, bin_edges = np.histogram(Data['Conf mean'][Camira_matched_else_arg], bins = bins, density=False)


plt.step(bin_edges[:-1], 88*else_dist, label = 'non C1/C2 objects')
plt.step(bin_edges[:-1], Camira_matched_else_dist, label = 'Camira matched non C1/C2 objects')

plt.ylabel('Count')
#plt.yscale('log')

plt.xlabel('Confidence')

plt.legend()
plt.show()

# -- compare fraction of camira matched > 0.1 confidence --
num_above_0_1 = len(np.where(Data['Conf mean'][else_arg]>0.1)[0])
num_camira_above_0_1 = len(np.where(Data['Conf mean'][Camira_matched_else_arg]>0.1)[0])

print("Number of non C1/C2 with confidence above 0.1: {}".format(num_above_0_1))
print("non C1/C2 fraction with confidence above 0.1: {}".format(num_above_0_1/len(else_arg)))
print("Camira matched non C1/C2 fraction with confidence above 0.1: {}".format(num_camira_above_0_1/len(Camira_matched_else_arg)))

print(" ")
# -- compare mean of camira matched --
camira_matched_else_mean_conf = np.mean(Data['Conf mean'][Camira_matched_else_arg])

print("Camira matched non C1/C2 count: {}".format(len(Camira_matched_else_arg)))
print("non C1/C2 confidence mean:", np.mean(Data['Conf mean'][else_arg]))
print("Camira matched non C1/C2 confidence mean:", camira_matched_else_mean_conf)

counter = 0
runs = 100
means = []
for i in range(runs):
    random_args = np.random.randint(0,len(else_arg), len(Camira_matched_else_arg))
    tmp = [else_arg[j] for j in random_args]
    mean = np.mean(Data['Conf mean'][tmp])
    if(mean > camira_matched_else_mean_conf):
        counter += 1
    means.append(mean)
    
print("Probability of random subset exceding {}: {}".format(camira_matched_else_mean_conf, counter/runs))

plt.hist(means, color = 'black')
plt.plot([camira_matched_else_mean_conf,camira_matched_else_mean_conf],[0,6000], linestyle='dashed', c = 'grey')
plt.xlabel("mean of confidence for randomly selected non C1/C2 sample")
plt.ylim(0,6000)
plt.show()

# --- Standard error of results ---------------------------------------------------------------------------------------------

Data.sort_values(['Conf mean'], ascending=True, inplace=True)

Conf_mean = Data['Conf mean'].to_numpy()

cumulative = np.cumsum(Data['Conf std'].to_numpy()/np.sqrt(10))



top_index = 1
bottom_index = 0
window = 0.05

moving_average = [0 for i in range(len(cumulative))]

for current_index in range(len(Data['Conf mean'])):
    
    # update bottom_index
    while(Conf_mean[current_index]-Conf_mean[bottom_index] > window):
        bottom_index += 1
    
    # update top_index
    while(Conf_mean[top_index-1]-Conf_mean[current_index] < window and top_index < len(Conf_mean)):
        top_index += 1
    
    moving_average[current_index] = (cumulative[top_index-1] - cumulative[bottom_index])/(top_index-1-bottom_index)
    

plt.scatter(Data['Conf mean'][C1_arg], Data['Conf std'][C1_arg]/np.sqrt(10), alpha = 0.5, label = 'C1 source')
plt.scatter(Data['Conf mean'][C2_arg], Data['Conf std'][C2_arg]/np.sqrt(10), alpha = 0.5, label = 'C2 source')
plt.scatter(Data['Conf mean'][else_arg], Data['Conf std'][else_arg]/np.sqrt(10), alpha = 0.5, label = 'Non-C1/C2 source')

#plt.plot(Data['Conf mean'], moving_average, c = 'black', label = 'Rolling average')

# -- set ranges --
plt.xlim(0,1)
plt.ylim(0, 1.1*max(Data['Conf std'])/np.sqrt(10))

# -- label axise --
plt.xlabel('Confidence value', fontsize=20)
plt.ylabel('Standard error in confidence', fontsize=20)

plt.legend(fontsize=20)

plt.show()

# --- Paramiter Relevance ----------------------------------------------------------------------------------------
TextSize = 16

try:
    col_names = Relevance[::-1]['Name']
except:
    col_id = Relevance[::-1]['id']
    col_names = All_Cols[col_id]

plt.rcParams.update({'font.size': TextSize})
fig, ax = plt.subplots()
ax.barh(col_names, Relevance[::-1]['Mean'], log = False, color = 'black')
plt.xscale('log')

for i, v in enumerate(Relevance[::-1]['Mean']):
    #if(i == 24):
    #    ax.text(v - 190, i - 0.3, "{:.2f}".format(v), fontsize=TextSize, color='w')
    #elif(i == 25):
    #    ax.text(v - 250, i - 0.3, "{:.2f}".format(v), fontsize=TextSize, color='w')
    #elif(i == 26):
    #    ax.text(v - 300, i - 0.3, "{:.2f}".format(v), fontsize=TextSize, color='w')
    #else:
    ax.text(v, i - 0.3 , "{:.2f}".format(v), fontsize=TextSize)
        

        
plt.title('Parameter weighting', fontsize=30)
plt.show()

