
#%% Following analyzes all of the processed data of edges with 100% contrast and 6 luminance
# Plots, stats
"""
@author: burakgur
"""
#%% Imports
import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

# Change to code folder
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/common')

import post_analysis_core as pac
#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
main_dir = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code'
data_dir = os.path.join(main_dir,'Figure1','processed_data') # Change this to load your data of interest
results_save_dir = os.path.join(main_dir,'Figure1/plots')
#%% Load data

# Lamina medulla data
load_path = os.path.join(data_dir, 'L2_-edges-real_lum_vals_rel0p5_maxt_0p3.pickle')
load_path = open(load_path, 'rb')
l2 = cPickle.load(load_path)

l2_df = l2['df'][['slope','flyID','Geno']]

load_path = os.path.join(data_dir, 'L3_-edges-real_lum_vals_rel0p5_maxt_0p3.pickle')
load_path = open(load_path, 'rb')
l3 = cPickle.load(load_path)

l3_df = l3['df'][['slope','flyID','Geno']]


# T4T5
load_path = os.path.join(data_dir, 'R64G09-Recomb-lexAopGC6f-edges-real_lum_vals_rel0p4.pickle')
load_path = open(load_path, 'rb')
t4t5 = cPickle.load(load_path)

t4t5_df = t4t5['df'][['slope','flyID','Geno']]


#%% Combine data
combined_tuning = np.concatenate((l2['tunings'], l3['tunings'],t4t5['tunings']))
combined_df = pd.concat([l2_df, l3_df, t4t5_df])

le = preprocessing.LabelEncoder()
le.fit(combined_df['flyID'])
combined_df['flyIDNum'] = le.transform(combined_df['flyID'])

#%% Colors

_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}


#%% Slope stats
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
from statsmodels.formula.api import ols
# import scikit_posthocs as sp
# Per fly data
a=pac.compute_over_samples_groups(data = combined_df['slope'], 
                                group_ids= np.array(combined_df['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df['Geno']))

# HERE PICK THE GENOTYPES to compare
# Figure 1: "T45","L3_","L2_"
data1 = np.array(a['experiment_ids']["R64G09-Recomb-lexAopGC6f"]['over_samples_means'])
data2 = np.array(a['experiment_ids']["L3_"]['over_samples_means'])
data3 = np.array(a['experiment_ids']["L2_"]['over_samples_means'])


data_all = [data1, data2, data3]
dfs = [pd.DataFrame({k: sample}) for k, sample in enumerate(data_all)]
stats_df = pd.concat(dfs,  ignore_index=True, axis=1)
stats_df.columns = ["R64G09-Recomb-lexAopGC6f","L3_","L2_"]

d_melt = pd.melt(stats_df.reset_index(), id_vars=['index'], value_vars=["R64G09-Recomb-lexAopGC6f","L3_","L2_"])
d_melt = d_melt.dropna()
d_melt.columns = ['index', 'geno', 'value']
fvalue, pvalue = stats.f_oneway(data1,data2,data3)

model = ols('value ~ C(geno)', data=d_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

# import scikit_posthocs as sp

comp = mc.MultiComparison(d_melt['value'], d_melt['geno'])
post_hoc_res = comp.tukeyhsd()
post_hoc_res.summary()
post_hoc_res.plot_simultaneous(ylabel= "Geno", xlabel= "Score Difference")


plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(post_hoc_res.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()



plt.close('all')


#%% Slope bar
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=0.3)
ax1=plt.subplot(grid[0,0])
for idx, geno in enumerate(np.unique(d_melt['geno'])):
    # sns.barplot(x='geno',y='value',data=d_melt,
    #                palette=c_dict,ax=ax1,ci="sd")
    pac.bar_bg(d_melt[d_melt['geno']==geno]['value'], idx+1, color=c_dict[geno], scat_s =7,ax=ax1, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label=geno)

plt.legend()
save_name = 'Slope_barr_Edges_L2L3T45'
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
#%%
plt.close('all')
fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
ax1=plt.subplot(grid[0,0])
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno]
    curr_neuron_mask = (combined_df['Geno']==geno) 

    curr_df = combined_df[curr_neuron_mask]
    
   
    luminances = l2['luminances']
    
    curr_tunings = combined_tuning[curr_neuron_mask]
    curr_tunings = curr_tunings/curr_tunings.max(axis=1).reshape(len(curr_tunings),1)
    # Normalized
    a=pac.compute_over_samples_groups(data = curr_tunings, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    
    
    fly_means = np.array(a['experiment_ids'][geno]['over_samples_means'])
    fly_means /= fly_means.max(axis=1).reshape(len(fly_means),1)
    fly_sem = np.std(fly_means,axis=0)/np.sqrt(fly_means.shape[0])
    fly_mean = np.mean(fly_means,axis=0)
    ax1.errorbar(luminances,fly_mean,fly_sem,
                 fmt='-o',alpha=1,color=geno_color,label=label)

ax1.set_ylim((0,1.04))
ax1.set_xscale('log')
ax1.legend()
ax1.set_xlim((10000,ax1.get_xlim()[1]))
# Saving figure
save_name = 'Edges_100p_Normalized_L2L3T4T5'.format(geno=geno)
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# %%
