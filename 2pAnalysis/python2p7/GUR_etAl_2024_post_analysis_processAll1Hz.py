#%% Following analyzes all of the processed data of 1Hz gratings with 100% contrast and 5 luminance
# This code is used for analyzing the processed data of responses to 1Hz gratings with constant contrast (100%) and changing luminance
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
load_path = os.path.join(data_dir, 'Lamina-Medulla-Neurons-sineGratings-real_lum_vals_rel0p6.pickle')
load_path = open(load_path, 'rb')
lam_med = cPickle.load(load_path)

lam_med_df = lam_med['df'][['slope','flyID','Geno']]


# Tm9 GluCl
load_path = os.path.join(data_dir, 'Tm9GluCl-sineGratings-real_lum_vals_rel0p5.pickle')
load_path = open(load_path, 'rb')
tm9glucl = cPickle.load(load_path)

tm9glucl_df = tm9glucl['df'][['slope','flyID','Geno']]

# Tm1 GluCl
load_path = os.path.join(data_dir, 'Tm1GluCl-sineGratings-real_lum_vals_rel0p5.pickle')
load_path = open(load_path, 'rb')
tm1glucl = cPickle.load(load_path)

tm1glucl_df = tm1glucl['df'][['slope','flyID','Geno']]


# Tm1 iGlu
load_path = os.path.join(data_dir, 'Tm1iGLu-sineGratings-real_lum_vals_rel0p3.pickle')
load_path = open(load_path, 'rb')
tm1iglu = cPickle.load(load_path)

tm1iglu_df = tm1iglu['df'][['slope','flyID','Geno']]

# T45 data
load_path = os.path.join(data_dir, 'T45-sineGratings-real_lum_vals-rel0p4.pickle')
load_path = open(load_path, 'rb')
t45 = cPickle.load(load_path)
t45['df']['Geno'] = np.array(np.tile("T45",t45['df'].shape[0]))
t45_df = t45['df'][['slope','flyID','Geno']]

# Tm9iGlu
load_path = os.path.join(data_dir, 'Tm9iGluSnfr-sineGratings-real_lum_vals_rel0p3.pickle')
load_path = open(load_path, 'rb')
tm9iGlu = cPickle.load(load_path)
tm9iGlu['df']['Geno'] = np.array(np.tile("9Glu",tm9iGlu['df'].shape[0]))
tm9iGlu_df = tm9iGlu['df'][['slope','flyID','Geno']]

#%% Combine data
combined_tuning = np.concatenate((tm9iGlu['tunings'], t45['tunings'],lam_med['tunings'],tm9glucl['tunings'],tm1glucl['tunings'],tm1iglu['tunings']))
combined_df = pd.concat([tm9iGlu_df, t45_df, lam_med_df, tm9glucl_df, tm1glucl_df,tm1iglu_df])

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
data1 = np.array(a['experiment_ids']["L2_"]['over_samples_means'])
data2 = np.array(a['experiment_ids']["Tm1"]['over_samples_means'])
data3 = np.array(a['experiment_ids']["Tm1GMR74G01Gal4_iGluSnfr"]['over_samples_means'])


# Normality check
# Perform Shapiro-Wilk test for each group
stat_group1, p_group1 = stats.shapiro(data1)
stat_group2, p_group2 = stats.shapiro(data2)
stat_group3, p_group3 = stats.shapiro(data3)

print("Shapiro-Wilk test results:")
print("Group 1 Shapiro normality p-value: {s}".format(s=p_group1))
print("Group 2 Shapiro normality p-value: {s}".format(s=p_group2))
print("Group 3 Shapiro normality p-value: {s}".format(s=p_group3))

# Check homogeneity of variances
# Perform Shapiro-Wilk test for each group
statistic, p_value = stats.levene(data1, data2, data3)

print("Levene test results:")
print("Levene variance check p-value: {s}".format(s=p_value))

# T test
t_stat, p_value = stats.ttest_ind(data2, data3)
print("T test of data2 and data3 p-value: {s}".format(s=p_value))

# ANOVA
data_all = [data1, data2, data3]
dfs = [pd.DataFrame({k: sample}) for k, sample in enumerate(data_all)]
stats_df = pd.concat(dfs,  ignore_index=True, axis=1)
stats_df.columns = ["L2_","Tm1","Tm1GMR74G01Gal4_iGluSnfr"]

d_melt = pd.melt(stats_df.reset_index(), id_vars=['index'], value_vars= ["L2_","Tm1","Tm1GMR74G01Gal4_iGluSnfr"])
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
ax1.set_ylim([-0.000002,0.000008])
plt.legend()
save_name = 'Slope_barr_Tm1iGlu'
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%% Plot all genotypes together
plt.close('all')
fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
ax1=plt.subplot(grid[0,0])
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno]
    curr_neuron_mask = (combined_df['Geno']==geno) 
    if not(geno in  ["L2_","Tm1","Tm1GMR74G01Gal4_iGluSnfr"]):
        continue
    curr_df = combined_df[curr_neuron_mask]
    
    if geno=="T45":
        luminances = np.array([ 12629.9564665 ,  26095.51250893,  53026.6245938 ,  79957.73667867,
       106888.84876353])[::-1]
    else:
        luminances = lam_med['luminances']
    
    curr_tunings = combined_tuning[curr_neuron_mask]
    curr_tunings = curr_tunings/curr_tunings.max(axis=1).reshape(len(curr_tunings),1)
    # curr_tunings = curr_tunings/(curr_tunings[:,0].reshape(len(curr_tunings),1))
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

ax1.set_ylim((0,1))
ax1.set_xscale('log')
ax1.legend()
ax1.set_xlim((10000,ax1.get_xlim()[1]))
# Saving figure
save_name = 'Sine_100p_Normalized_Tm1iGlu'.format(geno=geno)
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%% Gain analysis
gain_geno = "L3_"
gain_bool = (combined_df['Geno']==gain_geno) 
tuning_curves = combined_tuning[gain_bool]
# Normalized
tuning_curves = tuning_curves/tuning_curves.max(axis=1).reshape(len(tuning_curves),1)
a=pac.compute_over_samples_groups(data = tuning_curves, 
                            group_ids= np.array(combined_df[gain_bool]['flyIDNum']), 
                            error ='SEM',
                            experiment_ids = np.array(combined_df[gain_bool]['Geno']))
gain_data = a['experiment_ids'][gain_geno]['over_groups_mean']

plt.close('all')
fig = plt.figure(figsize=(5,5))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
ax1=plt.subplot(grid[0,0])
all_gains = {}
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if geno in ["L2_", "L3_", "T45", "Tm4", "Tm2"]: # if T4T5 will be analyzed be careful that luminances are the other way
        continue
    geno_color = c_dict[geno]
    
    curr_neuron_mask = (combined_df['Geno']==geno) 
    if geno=="T45":
        luminances = t45['luminances']
    else:
        luminances = lam_med['luminances']

    tuning_curves = combined_tuning[curr_neuron_mask]
    # Normalized
    gains = tuning_curves/tuning_curves.max(axis=1).reshape(len(tuning_curves),1)
    # gains /= (gain_data/gain_data.max())
    gains /= gain_data
    
    a=pac.compute_over_samples_groups(data = gains, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))
    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    
    all_gains[geno] = np.array(a['experiment_ids'][geno]['over_samples_means'])
    
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax1.errorbar(luminances,all_mean_data,all_yerr,
                 fmt='-o',alpha=1,color=geno_color,label=label)
# ax1.set_ylim((0,ax1.get_ylim()[1]))
ax1.set_title('Gain compared to {s}'.format(s=gain_geno))
ax1.legend()
ax1.set_xscale('log')
ax1.legend()
ax1.set_xlim((10000,ax1.get_xlim()[1]))

ax1.plot((0,ax1.get_xlim()[1]), [1, 1], color='k', linestyle='--', linewidth=2)
# Saving figure
save_name = 'Gain'
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# %%

all_gains
tm1_fly_n = np.shape(all_gains['Tm1'])[0]
tm9_fly_n = np.shape(all_gains['Tm9'])[0]

all_g = np.concatenate([all_gains['Tm1'].flatten(),all_gains['Tm9'].flatten()])

#create data
df = pd.DataFrame({'genotype': np.concatenate([np.tile("Tm1", tm1_fly_n*5), np.tile("Tm9", tm9_fly_n*5)]),
                   'luminance': np.tile(luminances, tm1_fly_n+tm9_fly_n),
                   'gain': all_g})



# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols

#perform two-way ANOVA
model = ols('gain ~ C(luminance) + C(genotype) + C(luminance):C(genotype)', data=df).fit()
sm.stats.anova_lm(model, typ=2)

# df['lum_str'] = [str(lum) for lum in df.luminance]
# df['combination'] = df.lum_str + " / " + df.genotype # combine combinations

# m_comp = mc.pairwise_tukeyhsd(endog=df['gain'], groups=df['combination'], alpha=0.05)

# m_comp.summary()



# %% Multiple comparisons of luminances using bonferonni method
print("highest luminance p", stats.ttest_ind(all_gains['Tm1'][:,0], all_gains['Tm9'][:,0]).pvalue*5)
print("2nd highest luminance p",stats.ttest_ind(all_gains['Tm1'][:,1], all_gains['Tm9'][:,1]).pvalue*5)
print("3rd highest luminance p", stats.ttest_ind(all_gains['Tm1'][:,2], all_gains['Tm9'][:,2]).pvalue* 5)
print("4th highest luminance p",stats.ttest_ind(all_gains['Tm1'][:,3], all_gains['Tm9'][:,3]).pvalue * 5)
print("5th highest luminance p",stats.ttest_ind(all_gains['Tm1'][:,4], all_gains['Tm9'][:,4]).pvalue * 5)
# %% Tm9 Tm1 DXM analysis (data from Freya)
from scipy.stats import linregress, spearmanr
import scipy.io as sio

data_dir = os.path.join(main_dir,'analyzed_data','220426_Thesis/DXM_data_Freya')

Tm9_data_c = sio.loadmat(os.path.join(data_dir,"Tm9_control_500_DXM.mat"))
Tm9_data_e = sio.loadmat(os.path.join(data_dir,"Tm9_exp_500_DXM.mat"))

Tm1_data_c = sio.loadmat(os.path.join(data_dir,"Tm1_control_500_DXM.mat"))
Tm1_data_e = sio.loadmat(os.path.join(data_dir,"Tm1_exp_500_DXM.mat"))

# %% Load luminances
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
measurements_f = '/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging/210716_Ultima_Luminances_ONOFFpaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])

Tm1_data_c['slopes'] = []
for fly_data in Tm1_data_c['fly_tunings']:
    lums = np.array([0.0625, 0.125, 0.25, 0.375, 0.5])
    candelas = res.intercept + res.slope*lums
    luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
    diff_luminances = luminances_photon
    Tm1_data_c['slopes'].append(linregress(diff_luminances, fly_data)[0])
    # Tm1_data_c['slopes'].append(spearmanr(fly_data,diff_luminances)[0])




Tm1_data_e['slopes'] = []
for fly_data in Tm1_data_e['fly_tunings']:
    lums = np.array([0.0625, 0.125, 0.25, 0.375, 0.5])
    candelas = res.intercept + res.slope*lums
    luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
    diff_luminances = luminances_photon
    Tm1_data_e['slopes'].append(linregress(diff_luminances, fly_data)[0])

Tm9_data_c['slopes'] = []
for fly_data in Tm9_data_c['fly_tunings']:
    lums = np.array([0.0625, 0.125, 0.25, 0.375, 0.5])
    candelas = res.intercept + res.slope*lums
    luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
    diff_luminances = luminances_photon
    Tm9_data_c['slopes'].append(linregress(diff_luminances, fly_data)[0])

Tm9_data_e['slopes'] = []
for fly_data in Tm9_data_e['fly_tunings']:
    lums = np.array([0.0625, 0.125, 0.25, 0.375, 0.5])
    candelas = res.intercept + res.slope*lums
    luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
    diff_luminances = luminances_photon
    Tm9_data_e['slopes'].append(linregress(diff_luminances, fly_data)[0])

#%% Slope bar
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=0.3)
ax1=plt.subplot(grid[0,0])
pac.bar_bg(Tm1_data_c['slopes'], 1, color=c_dict["Tm1"], scat_s =7,ax=ax1, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label="Tm1c")
pac.bar_bg(Tm1_data_e['slopes'], 2, color=c_dict["Tm1"], scat_s =7,ax=ax1, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label="Tm1e")

pac.bar_bg(Tm9_data_c['slopes'], 4, color=c_dict["Tm9"], scat_s =7,ax=ax1, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label="Tm9c")
pac.bar_bg(Tm9_data_e['slopes'], 5, color=c_dict["Tm9"], scat_s =7,ax=ax1, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label="Tm9e")
    

plt.legend()
save_name = 'Slope_barr_DXM'
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%%

#%% Stats
stats.ttest_ind(Tm1_data_c['slopes'], Tm1_data_e['slopes'])
# stats.ttest_ind(data1, data2)

#%% stats 2
fvalue, pvalue = stats.f_oneway(Tm1_data_e['fly_tunings'][:,0],
                                Tm1_data_e['fly_tunings'][:,1],
                                Tm1_data_e['fly_tunings'][:,2],
                                Tm1_data_e['fly_tunings'][:,3],
                                Tm1_data_e['fly_tunings'][:,4])

print(pvalue)
#%% Plot all genotypes together
plt.close('all')
fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
ax1=plt.subplot(grid[0,0])

label = '{g} n: {f}({ROI})'.format(g="Tm1c",
                                       f=Tm1_data_c['fly_tunings'].shape[0],
                                       ROI=Tm1_data_c['contrast_resp_filt'].shape[0])
    
    
fly_means = Tm1_data_c['fly_tunings']
fly_means /= fly_means.max(axis=1).reshape(len(fly_means),1)
fly_sem = np.std(fly_means,axis=0)/np.sqrt(fly_means.shape[0])
fly_mean = np.mean(fly_means,axis=0)
ax1.errorbar(diff_luminances,fly_mean,fly_sem,
                fmt='-o',alpha=1,color="k",label=label)

label = '{g} n: {f}({ROI})'.format(g="Tm1e",
                                       f=Tm1_data_e['fly_tunings'].shape[0],
                                       ROI=Tm1_data_e['contrast_resp_filt'].shape[0])
    
    
fly_means = Tm1_data_e['fly_tunings']
fly_means /= fly_means.max(axis=1).reshape(len(fly_means),1)
fly_sem = np.std(fly_means,axis=0)/np.sqrt(fly_means.shape[0])
fly_mean = np.mean(fly_means,axis=0)
ax1.errorbar(diff_luminances,fly_mean,fly_sem,
                fmt='-o',alpha=1,color=c_dict["Tm1"],label=label)


label = '{g} n: {f}({ROI})'.format(g="Tm9c",
                                       f=Tm9_data_c['fly_tunings'].shape[0],
                                       ROI=Tm9_data_c['contrast_resp_filt'].shape[0])
    
    
fly_means = Tm9_data_c['fly_tunings']
fly_means /= fly_means.max(axis=1).reshape(len(fly_means),1)
fly_sem = np.std(fly_means,axis=0)/np.sqrt(fly_means.shape[0])
fly_mean = np.mean(fly_means,axis=0)
ax1.errorbar(diff_luminances,fly_mean,fly_sem,
                fmt='-o',alpha=1,color="k",label=label)

label = '{g} n: {f}({ROI})'.format(g="Tm9e",
                                       f=Tm9_data_e['fly_tunings'].shape[0],
                                       ROI=Tm9_data_e['contrast_resp_filt'].shape[0])
    
    
fly_means = Tm9_data_e['fly_tunings']
fly_means /= fly_means.max(axis=1).reshape(len(fly_means),1)
fly_sem = np.std(fly_means,axis=0)/np.sqrt(fly_means.shape[0])
fly_mean = np.mean(fly_means,axis=0)
ax1.errorbar(diff_luminances,fly_mean,fly_sem,
                fmt='-o',alpha=1,color=c_dict["Tm9"],label=label)

                


ax1.set_ylim((0,ax1.get_ylim()[1]))
ax1.set_xscale('log')
ax1.legend()
ax1.set_xlim((10000,ax1.get_xlim()[1]))
# Saving figure
save_name = 'Sine_100p_Normalized_DXM'
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# %%
