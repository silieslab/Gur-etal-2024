#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:06:30 2020

@author: burakgur
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import ROI_mod
import PyROI
from post_analysis_core import run_matplotlib_params

#%%
def make_exp_summary_TF(figtitle,extraction_type,mean_image,roi_image,roi_traces,
                     rawStimData,bf_image,
                     rois_df,rois,stimulus_information,save_fig,current_movie_ID,
                     summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors, _ = run_matplotlib_params()
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(figtitle,fontsize=12)
    
    grid = plt.GridSpec(2, 6, wspace=1, hspace=0.3)
    
    ## BF masks
    ax=plt.subplot(grid[0,:2])
    sns.heatmap(mean_image,cmap='gray',ax=ax,cbar=False)
    sns.heatmap(bf_image,cmap='plasma',
                cbar_kws={'ticks': np.unique(bf_image[~np.isnan(bf_image)]),
                          'fraction':0.1,
                          'shrink' : 1,
                          'label': 'Hz',},alpha=0.5,vmin=0.1,vmax=1.5)
    ax.axis('off')
    ax.set_title('BF map')   
    
    ## Histogram
    ax=plt.subplot(grid[0,2])
    chart = sns.countplot('BF',data =rois_df,palette='plasma')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, 
                          fontweight='bold')
    leg = chart.legend()
    leg.remove()
    
    
    ## Tuning curve
    ax=plt.subplot(grid[0,3:])
    # Plot tuning curves
    tunings = np.squeeze(list(map(lambda roi : roi.TF_curve_resp, rois)))
    mean_t = np.mean(tunings,axis=0)
    std_t = np.std(tunings,axis=0)
    ub = mean_t + std_t
    lb = mean_t - std_t
    # Tuning curve
    
    TF_stim = rois[0].TF_curve_stim
    ax.fill_between(TF_stim, ub, lb,
                     color=colors[0], alpha=.2)
    
    ax.plot(TF_stim,mean_t,'-o',lw=4,color=colors[0],
            markersize=10)
    #ax.plot(TF_stim,tunings.T,alpha=0.3,lw=1)
    
    ax.set_xscale('log') 
    ax.set_title('Frequency tuning curve')  
    ax.set_xlabel('Hz')
    ax.set_ylabel('$\Delta F/F$')
    ax.set_xlim((ax.get_xlim()[0],10)) 
    
    ## Plotting all tuning curves
    if len(rois) > 100:
        ax=plt.subplot(grid[1,:])
    elif len(rois) > 75:
        ax=plt.subplot(grid[1,:4])
    elif len(rois) > 50:
        ax=plt.subplot(grid[1,:3])
    elif len(rois) > 25:
        ax=plt.subplot(grid[1,:2])
    else:
        ax=plt.subplot(grid[1,1])
        
        
    non_edge_stims = (stimulus_information['stim_type'] != 50)
    uniq_freq_nums = len(np.where(np.unique(stimulus_information['epoch_frequency'][non_edge_stims])>0)[0])
    bfs = np.squeeze(list(map(lambda roi : roi.BF, rois)))
    sorted_indices = np.argsort(bfs)
    tf_tunings = tunings[sorted_indices,:]
    plot_tf_array = np.zeros(shape=(np.shape(tf_tunings)[0],np.shape(tf_tunings)[0]*np.shape(tf_tunings)[1]))
    plot_tf_array[:] = np.nan
    for i in range(np.shape(tf_tunings)[0]):
        
        curr_data = tf_tunings[i,:]
        curr_data = curr_data + np.mod(i,9)
        curve_start = i*np.shape(tf_tunings)[1] - (i*(uniq_freq_nums-1))
        plot_tf_array[i,curve_start:curve_start+uniq_freq_nums] = curr_data
        
    ax.plot(np.transpose(plot_tf_array),'-o',linewidth=2.0, alpha=.8,
            color=colors[0],markersize=0.4)
    ax.axis('off')
    ax.set_title('ROI tuning curves N: %s' % len(rois))
    
    if save_fig:
        # Saving figure 
        save_name = 'Summary_%s_%s' % (current_movie_ID, extraction_type)
        os.chdir(summary_save_dir)
        plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    return fig
        
def fffSummary(figtitle,stim_trace,
                roi_conc_traces,save_fig,current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors, _ = run_matplotlib_params()
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle(figtitle,fontsize=12)
    

    ## Conc responses
    ax=plt.axes()
    ax.plot(np.transpose(roi_conc_traces),color='k',alpha=.4,lw=1)
    mean_r = np.mean(roi_conc_traces,axis=0)
    std_r = np.std(roi_conc_traces,axis=0)
    ub = mean_r + std_r
    lb = mean_r - std_r
    ax.fill_between(range(len(mean_r)), ub, lb,
                     color=colors[3], alpha=.4)
    
    ax.plot(range(len(mean_r)),mean_r,lw=3,color=colors[3])
    scaler = np.abs(np.max(mean_r) - np.min(mean_r))
    plot_stim = np.array(stim_trace).astype(int)*(scaler*1) + np.max(roi_conc_traces)+(0.1*scaler)
    ax.plot(plot_stim,'k',lw=2.5)
    ax.set_title('5sFFF response')  
    ax.set_xlabel('Frames')
    ax.set_ylabel('$\Delta F/F$')

    if save_fig:
        # Saving figure 
        save_name = 'Summary_%s' % (current_movie_ID)
        os.chdir(summary_save_dir)
        plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
        
    return fig
def make_exp_summary_FFF(figtitle,mean_image,roi_image,roi_traces,
                         raw_stim_trace,stim_trace,roi_conc_traces,save_fig,
                         current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors, _ = run_matplotlib_params()
    fig = plt.figure(figsize=(7, 7))
    fig.suptitle(figtitle,fontsize=12)
    
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
     
    ## ROIs
    ax=plt.subplot(grid[0:1,0:1])
    
    sns.heatmap(mean_image,cmap='gist_yarg',ax=ax,cbar=False)
    sns.heatmap(roi_image,alpha=0.3,cmap = 'Dark2',ax=ax,
                cbar_kws={'fraction':0.1,
                          'shrink' : 0,
                          'ticks': []})
    ax.axis('off')
    ax.set_title('ROIs n:%d' % np.shape(roi_traces)[0])   
    
    ## Raw traces
    ax=plt.subplot(grid[1,:])
    adder = np.linspace(0, np.shape(roi_traces)[0]*1.5, 
                            np.shape(roi_traces)[0])[:,None]
    scaled_responses = roi_traces + adder
    # Finding stimulus
    stim_frames = raw_stim_trace[:,7]  # Frame information
    stim_vals = raw_stim_trace[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    # Make normalized values of stimulus values for plotting
    stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
        *np.max(scaled_responses)/6
    stim_df = pd.DataFrame(stim_vals+np.max(scaled_responses),
                           columns=['Stimulus'],dtype='float')
    resp_df = pd.DataFrame(np.transpose(scaled_responses),dtype='float')
    resp_df.plot(legend=False,alpha=0.8,lw=2,ax=ax,cmap='Dark2')   
    stim_df.plot(dashes=[2, 0.5],ax=ax,color='k',alpha=.6,lw=2)
    ax.get_legend().remove()
    ax.axis('off')
    ax.set_title('Raw traces')
    
    ## Conc responses
    ax=plt.subplot(grid[0,1:])
    ax.plot(np.transpose(roi_conc_traces),color='k',alpha=.4,lw=1)
    mean_r = np.mean(roi_conc_traces,axis=0)
    std_r = np.std(roi_conc_traces,axis=0)
    ub = mean_r + std_r
    lb = mean_r - std_r
    ax.fill_between(range(len(mean_r)), ub, lb,
                     color=colors[3], alpha=.4)
    
    ax.plot(range(len(mean_r)),mean_r,lw=3,color=colors[3])
    scaler = np.abs(np.max(mean_r) - np.min(mean_r))
    plot_stim = np.array(stim_trace).astype(int)/(scaler*5) + np.max(roi_conc_traces)+(0.1*scaler)
    ax.plot(plot_stim,'k',lw=2.5)
    ax.set_title('5sFFF response')  
    ax.set_xlabel('Frames')
    ax.set_ylabel('$\Delta F/F$')

    # if save_fig:
    #     # Saving figure 
    #     save_name = 'Summary_%s' % (current_movie_ID)
    #     os.chdir(summary_save_dir)
    #     plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
        
    return fig
        
def make_exp_summary_stripes(figtitle,analysis_params,mean_image,roi_image,
                             roi_traces,raw_stim_trace,roi_RF,save_fig,
                             current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors, _ = run_matplotlib_params()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(figtitle,fontsize=12)
    
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
    
    ## ROIs
    ax=plt.subplot(grid[0:1,0:1])
    
    sns.heatmap(mean_image,cmap='gist_yarg',ax=ax,cbar=False)
    sns.heatmap(roi_image,alpha=0.3,cmap = 'Dark2',ax=ax,
                cbar_kws={'fraction':0.1,
                          'shrink' : 0,
                          'ticks': []})
    ax.axis('off')
    ax.set_title('ROIs n:%d' % np.shape(roi_traces)[0])   
    
    ## Raw traces
    ax=plt.subplot(grid[1,:])
    factor = np.abs(np.max(roi_traces) - np.min(roi_traces))
    adder = np.linspace(0, np.shape(roi_traces)[0]*1.5, 
                            np.shape(roi_traces)[0])[:,None]
    scaled_responses = roi_traces + adder
    # Finding stimulus
    stim_frames = raw_stim_trace[:,7]  # Frame information
    stim_vals = raw_stim_trace[:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]
    # Make normalized values of stimulus values for plotting
    stim_vals = (stim_vals/np.max(np.unique(stim_vals))) \
        *np.max(scaled_responses)/3
    stim_df = pd.DataFrame(stim_vals+np.max(scaled_responses),
                           columns=['Stimulus'],dtype='float')
    resp_df = pd.DataFrame(np.transpose(scaled_responses),dtype='float')
    resp_df.plot(legend=False,alpha=0.8,lw=2,ax=ax,cmap='Dark2')   
    stim_df.plot(dashes=[2, 0.5],ax=ax,color='k',alpha=.6,lw=2)
    ax.get_legend().remove()
    ax.axis('off')
    ax.set_title('Raw traces')
    
    ## RF responses
    ax=plt.subplot(grid[0,1:])
    if (analysis_params['analysis_type'] == 'stripes_ON_vertRF_transfer')or\
        (analysis_params['analysis_type'] == 'stripes_OFF_vertRF_transfer'):
        sns.heatmap(np.transpose(roi_RF),cmap='coolwarm',center=0,ax=ax,yticklabels=10,
                xticklabels=2,cbar_kws={'label': '$\Delta F/F$'})
        ax.set_title('Receptive fields')  
        ax.set_ylabel('Position ($^\circ$)')
        ax.set_xlabel('ROI #')
    else:
        sns.heatmap(roi_RF,cmap='coolwarm',center=0,ax=ax,yticklabels=2,
                xticklabels=10,cbar_kws={'label': '$\Delta F/F$'})
        ax.set_title('Receptive fields')  
        ax.set_xlabel('Position ($^\circ$)')
        ax.set_ylabel('ROI #')
    
    

    if save_fig:
        # Saving figure 
        save_name = 'Summary_%s_%s' % (current_movie_ID)
        os.chdir(summary_save_dir)
        plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    return fig
                                
def make_exp_summary_luminance_edges(figtitle,rois,roi_image,
                                     current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    import matplotlib 
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    run_matplotlib_params()
    fig = plt.figure(figsize=(16, 3))
    fig.suptitle(figtitle,fontsize=12)
    
    grid = plt.GridSpec(1, 3, wspace=0.3, hspace=1)
    
    ## ROIs
    ax1=plt.subplot(grid[0,2])
    
    sns.heatmap(rois[0].source_image,cmap='gist_yarg',ax=ax1,cbar=False)
    slope_data = ROI_mod.data_to_list(rois, ['slope'])['slope']
    rangecolor= np.max(np.abs([np.min(slope_data),np.max(slope_data)]))
    sns.heatmap(roi_image,alpha=0.8,cmap = 'PRGn',ax=ax1,center=0,vmin=-rangecolor,
                vmax=rangecolor,cbar_kws={'fraction':0.1,
                          'shrink' : 1})
    ax1.axis('off')
    ax1.set_title('Slope of luminance sensitivity')   
    
    ## Raw traces
    ax2=plt.subplot(grid[0,0])
    ax3=plt.subplot(grid[0,1])
    
    diff_luminances = rois[0].luminances
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=np.min(diff_luminances), 
                                       vmax=np.max(diff_luminances))
    all_traces = \
        np.array(map(lambda roi : roi.edge_resp_traces_interpolated,
                         rois))
    sensitivities = np.array(map(lambda roi : roi.edge_resps,rois))
    sensitivities= np.transpose(sensitivities)
    
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    for idx, luminance in enumerate(diff_luminances):
        curr_traces = all_traces[:,idx,:]
        curr_traces_aligned=list(map(lambda trace : np.roll(trace, 20-np.argmax(trace)),
                   curr_traces))
        
        mean_t = np.mean(curr_traces_aligned,axis=0)
        x = np.linspace(0,len(mean_t),len(mean_t))
        ax2.plot(x,mean_t,'-',lw=3,color=cmap(norm(luminance)),alpha=.8,
                 label=luminance)
        
        
        curr_sensitivities = sensitivities[idx,:]
        curr_luminances = np.ones(curr_sensitivities.shape) * luminance
        df = pd.DataFrame.from_dict({'Luminance':curr_luminances,
                                     'Response':curr_sensitivities}) 
        rois_df = pd.DataFrame.from_dict(df)
        senst_df = senst_df.append(rois_df, ignore_index=True, sort=False)
        # bar_bg(curr_sensitivities, luminance, color=cmap(norm(luminance)), 
        #        scat_s =2,ax=ax3,alpha = .8,width=0.1)
    
    
    ax2.set_ylabel('$\Delta F/F$')
    ax2.set_title('Aligned mean responses n:%d' % len(rois))   
    
    sns.violinplot(x="Luminance", y="Response", data=senst_df, linewidth=1.5,
               inner="quartile", palette='plasma', ax=ax3)
    
    ax3.set_title('Luminance sensitivitiy')   
    ax3.set_ylabel('$\Delta F/F$')
    
    if 'OFF' in rois[0].stim_name:
        ax2.legend(title='X -> 0')
        ax3.set_xlabel('Preceeding luminance')
    else:
        ax2.legend(title='0 -> X')
        ax3.set_xlabel('Following luminance')
     # Saving figure 
    # save_name = 'Summary_%s' % (current_movie_ID)
    # os.chdir(summary_save_dir)
    # plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    return fig

def summarizeLuminanceEdges(figtitle,rois,roi_image,
                                current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    import matplotlib 
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    run_matplotlib_params()
    fig = plt.figure(figsize=(16, 3))
    fig.suptitle(figtitle,fontsize=12)
    
    grid = plt.GridSpec(1, 3, wspace=0.3, hspace=1)
    
    ## ROIs
    ax1=plt.subplot(grid[0,2])
    
    sns.heatmap(rois[0].source_image,cmap='gist_yarg',ax=ax1,cbar=False)
    slope_data = PyROI.dataToList(rois, ['slope'])['slope']
    rangecolor= np.max(np.abs([np.min(slope_data),np.max(slope_data)]))
    sns.heatmap(roi_image,alpha=0.8,cmap = 'PRGn',ax=ax1,center=0,vmin=-rangecolor,
                vmax=rangecolor,cbar_kws={'fraction':0.1,
                          'shrink' : 1})
    ax1.axis('off')
    ax1.set_title('Slope of luminance sensitivity')   
    
    ## Raw traces
    ax2=plt.subplot(grid[0,0])
    ax3=plt.subplot(grid[0,1])
    
    diff_luminances = [val for key, val in rois[0].epoch_luminances.iteritems()]
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=np.min(diff_luminances), 
                                       vmax=np.max(diff_luminances))
    all_traces = np.array(map(lambda roi : roi.int_resp_trace.values(),rois))
    sensitivities = np.array(map(lambda roi : roi.edge_step_responses.values(),rois))
    sensitivities= np.transpose(sensitivities)
    
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    for idx, luminance in enumerate(diff_luminances):
        curr_traces = all_traces[:,idx,:]
        curr_traces_aligned=list(map(lambda trace : np.roll(trace, 20-np.argmax(trace)),
                   curr_traces))
        
        mean_t = np.mean(curr_traces_aligned,axis=0)
        x = np.linspace(0,len(mean_t),len(mean_t))
        ax2.plot(x,mean_t,'-',lw=3,color=cmap(norm(luminance)),alpha=.8,
                 label=luminance)
        
        
        curr_sensitivities = sensitivities[idx,:]
        curr_luminances = np.ones(curr_sensitivities.shape) * luminance
        df = pd.DataFrame.from_dict({'Luminance':curr_luminances,
                                     'Response':curr_sensitivities}) 
        rois_df = pd.DataFrame.from_dict(df)
        senst_df = senst_df.append(rois_df, ignore_index=True, sort=False)
        # bar_bg(curr_sensitivities, luminance, color=cmap(norm(luminance)), 
        #        scat_s =2,ax=ax3,alpha = .8,width=0.1)
    
    
    ax2.set_ylabel('$\Delta F/F$')
    ax2.set_title('Aligned mean responses n:%d' % len(rois))   
    
    sns.violinplot(x="Luminance", y="Response", data=senst_df, linewidth=1.5,
               inner="quartile", palette='plasma', ax=ax3)
    
    ax3.set_title('Luminance sensitivitiy')   
    ax3.set_ylabel('$\Delta F/F$')
    
    if 'OFF' in rois[0].stim_info['meta']['stim_name']:
        ax2.legend(title='X -> 0')
        ax3.set_xlabel('Preceeding luminance')
    else:
        ax2.legend(title='0 -> X')
        ax3.set_xlabel('Following luminance')
     # Saving figure 
    # save_name = 'Summary_%s' % (current_movie_ID)
    # os.chdir(summary_save_dir)
    # plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    return fig

def make_exp_summary_luminance_steps(figtitle,rois,roi_image,
                                     current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    import matplotlib 
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    run_matplotlib_params()
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(figtitle,fontsize=12)
    
    coln = 3
    grid = plt.GridSpec(6, coln, wspace=0.3, hspace=0.1)
    
    ## ROIs
    ax1=plt.subplot(grid[0:2,0])
    
    sns.heatmap(rois[0].source_image,cmap='gist_yarg',ax=ax1,cbar=False)
    sns.heatmap(roi_image,alpha=0.3,cmap='Set2')
    ax1.axis('off')
    
    ## Traces
    ax2=plt.subplot(grid[0:2,1:])
    
    diff_luminances = rois[0].luminances
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=np.min(diff_luminances), 
                                       vmax=np.max(diff_luminances))
    all_traces = \
        np.array(map(lambda roi : roi.lum_resp_traces_interpolated,
                         rois))
    
    
    cur_row = 1
    for idx, luminance in enumerate(diff_luminances):
        
        if np.mod(idx,coln) == 0:
            cur_row +=1
        curr_traces = all_traces[:,idx,:]
        
        ax=plt.subplot(grid[cur_row,np.mod(idx,coln)])
        
        mean_r = np.mean(curr_traces,axis=0)
        std_r = np.std(curr_traces,axis=0)
        ub = mean_r + std_r
        lb = mean_r - std_r
        x = np.linspace(0,len(mean_r),len(mean_r))
        ax.plot(x,np.transpose(curr_traces),lw=1,color='k',alpha=.3)
        ax.plot(x,mean_r,'-',lw=3,color=cmap(norm(luminance)),alpha=.9,
                 label=luminance)
        ax.fill_between(x, ub, lb,color=cmap(norm(luminance)), alpha=.3)
        ax.legend()
        # ax.axis('off')
        ax.set_ylim(-0.5,np.max(all_traces.mean(axis=0)+0.3))
        ax2.plot(x,mean_r,'-',lw=3,color=cmap(norm(luminance)),alpha=.6,
                 label=luminance)
    ax2.set_ylabel('$\Delta F/F$')
    ax2.set_title('Luminance responses n:%d' % len(rois))   
    
    
    # save_name = 'Summary_%s' % (current_movie_ID)
    # os.chdir(summary_save_dir)
    # plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    return fig

def make_exp_summary_AB_steps(figtitle,rois,roi_image,
                                     current_movie_ID,summary_save_dir):
    """ Plots a summary of experiment with relevant information.
    
    """
    import matplotlib 
    
    plt.close('all')
    # Constructing the plot backbone, selecting colors
    colors, _ = run_matplotlib_params()
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(figtitle,fontsize=12)
    
    coln = 3
    grid = plt.GridSpec(4, coln, wspace=0.3, hspace=0.1)
    
    ## ROIs
    ax1=plt.subplot(grid[0:2,0])
    
    sns.heatmap(rois[0].source_image,cmap='gist_yarg',ax=ax1,cbar=False)
    sns.heatmap(roi_image,alpha=0.3,cmap='Set2')
    ax1.axis('off')
    
    ## Traces
    ax2=plt.subplot(grid[0:2,1:])
    
    diff_luminances = rois[0].epoch_luminance_B_steps
    diff_luminances = np.delete(diff_luminances,
                                rois[0].stim_info['baseline_epoch'])
    cmap = matplotlib.cm.get_cmap('inferno')
    norm = matplotlib.colors.Normalize(vmin=np.min(diff_luminances), 
                                       vmax=np.max(diff_luminances))
    all_traces = \
        np.array(map(lambda roi : roi.resp_traces_interpolated,
                         rois))
    
    cur_row = 1
    for idx, luminance in enumerate(diff_luminances):
        
        if np.mod(idx,coln) == 0:
            cur_row +=1
        curr_traces = all_traces[:,idx,:]
        
        ax=plt.subplot(grid[cur_row,np.mod(idx,coln)])
        
        mean_r = np.mean(curr_traces,axis=0)
        std_r = np.std(curr_traces,axis=0)
        ub = mean_r + std_r
        lb = mean_r - std_r
        x = np.linspace(0,len(mean_r),len(mean_r))
        ax.plot(x,np.transpose(curr_traces),lw=1,color='k',alpha=.3)
        ax.plot(x,mean_r,'-',lw=3,color=cmap(norm(luminance)),alpha=.9,
                 label=luminance)
        ax.fill_between(x, ub, lb,color=cmap(norm(luminance)), alpha=.3)
        ax.legend()
        # ax.axis('off')
        ax.set_ylim(-0.5,np.max(all_traces.mean(axis=0)+0.3))
        ax2.plot(x,mean_r,'-',lw=3,color=cmap(norm(luminance)),alpha=.6,
                 label=luminance)
    ax2.set_ylabel('$\Delta F/F$')
    ax2.set_title('Luminance responses n:%d' % len(rois))   
    
    
    save_name = 'Summary_%s' % (current_movie_ID)
    os.chdir(summary_save_dir)
    plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    
    
    fig2 = plt.figure(figsize=(12, 5))
    fig2.suptitle(figtitle,fontsize=12)
    
    plt.subplot(121)
    
    data = ROI_mod.data_to_list(rois, ['a_step_responses','b_step_responses'])
    
    a_resps = data['a_step_responses']
    b_resps = data['b_step_responses']
    
    a_lums = rois[0].epoch_luminance_A_steps
    b_lums = rois[0].epoch_luminance_B_steps
    
    a_err = np.nanstd(a_resps,axis=0)/np.sqrt(np.shape(a_resps)[0])
    b_err = np.nanstd(b_resps,axis=0)/np.sqrt(np.shape(b_resps)[0])
    plt.errorbar(a_lums,np.nanmean(a_resps,axis=0), a_err,
                 fmt='s',color=colors[2],label='A step',alpha=.9)
    plt.errorbar(b_lums,np.nanmean(b_resps,axis=0), b_err,
                 fmt='s',color=colors[3],label='B step',alpha=.9)
    plt.xlabel('Luminance')
    plt.ylabel('$\Delta F/F$')
    plt.legend()
    
    plt.subplot(122)
    
    data = ROI_mod.data_to_list(rois, ['a_step_responses','b_step_responses'])
    
    a_resps = data['a_step_responses']
    b_resps = data['b_step_responses']
    
    a_cont = rois[0].epoch_contrast_A_steps
    b_cont = rois[0].epoch_contrast_B_steps
    
    a_err = np.nanstd(a_resps,axis=0)/np.sqrt(np.shape(a_resps)[0])
    b_err = np.nanstd(b_resps,axis=0)/np.sqrt(np.shape(b_resps)[0])
    plt.errorbar(a_cont,np.nanmean(a_resps,axis=0), a_err,fmt='s',
                 color=colors[2],label='A step',alpha=.9)
    plt.errorbar(b_cont,np.nanmean(b_resps,axis=0), b_err,fmt='s',
                 color=colors[3],label='B step',alpha=.9)
    plt.xlabel('Contrast')
    plt.ylabel('$\Delta F/F$')
    plt.legend()
    
    save_name = 'Summary_%s_con_lum' % (current_movie_ID)
    os.chdir(summary_save_dir)
    plt.savefig('%s.pdf'% save_name, bbox_inches='tight',dpi=300)
    
    
    return fig, fig2
                                
