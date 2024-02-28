#%% Following analyzes 5s full field flashes for Tm9 iGluSnFR recordings.
"""
Created on Wed Jan  8 15:33:56 2020

@author: burakgur
"""

#%%
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings


#change to code directory
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/common')

import post_analysis_core as pac
import ROI_mod
#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code'
# initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'raw_data')

results_save_dir = os.path.join(initialDirectory, 'FigureS4','plots')


# %% Load datasets and desired variables
exp_folder = 'Tm9iGluSnFRs' #Raw data folder: "Tm9iGluSnFR"
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
genotype_labels = 0

plot_only_cat = True
cat_dict={}
exp_t = 'Tm9iGluSNFR'
# datasets_to_load = os.listdir(data_dir)
# Initialize variables
final_rois_all = []
flyIDs = []
flash_resps = []
OFF_steps = []
OFF_plateau = []
OFF_int = []

ON_steps = []
ON_plateau = []
ON_int = []


flash_corr = []
genotypes = []

for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    if (not('5sec_220deg' in curr_rois[0].stim_name)):
        continue
    if dataset == '210511bg_fly4-TSeries-05112021-1137-005_manual.pickle':
        for roi in curr_rois:
            roi.experiment_info['Genotype'] = "exp_dNlg-CD19-Tm9_dSNR-L3"
    curr_rois = ROI_mod.conc_traces(curr_rois,int_rate=10)    
    
    geno = curr_rois[0].experiment_info['Genotype']
    if geno == '"ctr_dSNR-L3"':
        geno = "ctr_dSNR-L3"
    print(geno)
    print(curr_rois[0].stim_name)
    
    final_rois_all.append(workspace['final_rois'])
    
    stim_t = curr_rois[0].int_stim_trace
    fff_stim_trace = \
        np.around(np.concatenate((stim_t,stim_t[0:30]),axis=0))
    for roi in curr_rois:
        
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(geno)
        trace = roi.int_con_trace
        
        
        flash_t = np.concatenate((trace,trace[0:30]),axis=0)
        flash_resps.append(flash_t)

        # OFF properties
        off_step = trace[0:20].max()-trace[50:55].mean() # OFF STEP NOW
        OFF_steps.append(off_step)
        off_plat = trace[45:50].mean()-trace[95:100].mean()
        OFF_plateau.append(off_plat)
        off_int = trace[0:50].sum()
        OFF_int.append(off_int)
        

        # ON properties
        on_step = trace[50:70].max()-trace[45:50].mean() # OFF STEP NOW
        ON_steps.append(on_step)
        on_plat = trace[95:100].mean()-trace[45:50].mean()
        ON_plateau.append(on_plat)
        on_int = trace[50:].sum()
        ON_int.append(on_int)

        flash_corr.append(roi.corr_fff)
        
    
        
    print('{ds} successfully loaded\n'.format(ds=dataset))

#%% Flash figures
# Over flies and experiments
fff_dict = pac.compute_over_samples_groups(data = flash_resps,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
on_step_dict = pac.compute_over_samples_groups(data = ON_steps,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
on_plat_dict = pac.compute_over_samples_groups(data = ON_plateau,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
on_int_dict = pac.compute_over_samples_groups(data = ON_int,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)

off_step_dict = pac.compute_over_samples_groups(data = OFF_steps,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
off_plat_dict = pac.compute_over_samples_groups(data = OFF_plateau,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
off_int_dict = pac.compute_over_samples_groups(data = OFF_int,
                                           group_ids= flyIDs, error ='SEM',
                                           experiment_ids = genotypes)
#%% Conc responses
# Plotting parameters
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in np.unique(genotypes)}

plt.close('all')
fig = plt.figure(figsize=(15, 3))
fig.suptitle('5sFFF properties',fontsize=12)

grid = plt.GridSpec(2,6 ,wspace=0.5, hspace=0.4)

# FFF responses
ax=plt.subplot(grid[:,:3])
ax2=plt.subplot(grid[0,3])
ax3=plt.subplot(grid[0,4])
ax4=plt.subplot(grid[0,5])

ax5=plt.subplot(grid[1,3])
ax6=plt.subplot(grid[1,4])
ax7=plt.subplot(grid[1,5])

labels = ['']
fff_all_max = 0.0
t_trace = np.linspace(0,len(fff_stim_trace),len(fff_stim_trace))/10


if not genotype_labels:
    genotype_labels = fff_dict['experiment_ids'].keys()
for idx, genotype in enumerate(fff_dict['experiment_ids'].keys()):
    
    
    
    curr_data = fff_dict['experiment_ids'][genotype]
    gen_str = \
        '{gen} n: {nflies} ({nROIs})'.format(gen=genotype_labels[idx],
                                           nflies =\
                                               len(curr_data['over_samples_means']),
                                           nROIs=\
                                               len(curr_data['all_samples']))
    mean = curr_data['over_groups_mean']
    if np.max(mean) > fff_all_max:
        fff_all_max = np.max(mean)
    error = curr_data['over_groups_error']
    ub = mean + error
    lb = mean - error

    t_trace = t_trace[:120]
    plot_trace = np.concatenate((mean[80:],mean[30:100]),axis=0)
    plot_ub = np.concatenate((ub[80:],ub[30:100]),axis=0)
    plot_lb = np.concatenate((lb[80:],lb[30:100]),axis=0)
    ax.plot(t_trace,plot_trace,color=c_dict[genotype],alpha=.8,lw=3,label=gen_str)
    ax.fill_between(t_trace, plot_ub, plot_lb,
                     color=c_dict[genotype], alpha=.4)
    scaler = np.abs(np.max(mean) - np.min(mean))
    
    curr_data = on_step_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=c_dict[genotype], 
               ax=ax2,yerr=error)
    
    curr_data = on_plat_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=c_dict[genotype], 
               ax=ax3,yerr=error)
    
    curr_data = on_int_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=c_dict[genotype], 
               ax=ax4,yerr=error)


    curr_data = off_step_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=c_dict[genotype], 
               ax=ax5,yerr=error)
    
    curr_data = off_plat_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=c_dict[genotype], 
               ax=ax6,yerr=error)
    
    curr_data = off_int_dict['experiment_ids'][genotype]
    error = curr_data['over_groups_error']
    pac.bar_bg(curr_data['over_samples_means'], idx+1, color=c_dict[genotype], 
               ax=ax7,yerr=error)
    
    
    labels.append(genotype_labels[idx])
    
    
plot_stim = np.concatenate((fff_stim_trace[80:],fff_stim_trace[30:100]),axis=0) 
ax.plot(t_trace,
        plot_stim/6+ fff_all_max,'--k',lw=1.5,alpha=.8)

ax.set_title('Response')  
ax.set_xlabel('Time (s)')
ax.set_ylabel('$\Delta F/F$')
ax.legend()

ax2.set_title('ON step')  
ax2.set_ylabel('$\Delta F/F$')
ax2.xaxis.set_visible(False)
ax2.plot(list(ax2.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax3.set_title('ON plateau')  
ax3.set_ylabel('$\Delta F/F$')
ax3.xaxis.set_visible(False)
ax3.plot(list(ax3.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax4.set_title('ON integral')  
ax4.set_ylabel('$\Delta F/F$')
ax4.xaxis.set_visible(False)
ax4.plot(list(ax4.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax5.set_title('OFF step')  
ax5.set_ylabel('$\Delta F/F$')
ax5.xaxis.set_visible(False)
ax5.plot(list(ax5.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax6.set_title('OFF plateau')  
ax6.set_ylabel('$\Delta F/F$')
ax6.xaxis.set_visible(False)
ax6.plot(list(ax6.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

ax7.set_title('OFF integral')  
ax7.set_ylabel('$\Delta F/F$')
ax7.xaxis.set_visible(False)
ax7.plot(list(ax7.get_xlim()), [0, 0], "k",lw=plt.rcParams['axes.linewidth'])

fig.tight_layout()

if 1:
    # Saving figure
    save_name = '{exp_t}_FFF'.format(exp_t=exp_t)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
# plt.close('all')
