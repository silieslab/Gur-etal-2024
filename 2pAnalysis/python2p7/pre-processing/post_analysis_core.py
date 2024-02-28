#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:20:14 2020

@author: burakgur
"""

#%% Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mutual_info_score
#%% Functions
def run_matplotlib_params():
    plt.style.use('default')
    plt.style.use('seaborn-talk')
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)    
    plt.rcParams["axes.titlesize"] = 'medium'
    plt.rcParams["axes.labelsize"] = 'small'
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["legend.fontsize"] = 'small'
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["figure.titleweight"] = 'bold'
    plt.rcParams["figure.titlesize"] = 'medium'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['legend.loc'] = 'upper right'
    
    c_dict = {}
    c_dict['dark_gray'] = np.array([77,77,77]).astype(float)/255
    c_dict['light_gray'] = np.array([186,186,186]).astype(float)/255
    c_dict['green1'] = np.array([102,166,30]).astype(float)/255
    c_dict['green2']=np.array([179,226,205]).astype(float)/255
    c_dict['green3'] = np.array([27,158,119]).astype(float)/255
    c_dict['orange']  = np.array([201,102,47]).astype(float)/255
    c_dict['red']  = np.array([228,26,28]).astype(float)/255
    c_dict['magenta']  = np.array([231,41,138]).astype(float)/255
    c_dict['purple']  = np.array([117,112,179]).astype(float)/255
    c_dict['yellow'] = np.array([255,255,51]).astype(float)/255
    c_dict['brown'] = np.array([166,86,40]).astype(float)/255
    
    c_dict['L3_'] = np.array([102,166,30]).astype(float)/255 # Dark2 Green
    c_dict['Tm9'] = np.array([27,158,119]).astype(float)/255 # Dark2 Weird green
    c_dict['9Glu'] = np.array([27,158,119]).astype(float)/255 # Dark2 Weird green
    c_dict['L1_'] = np.array([230,171,2]).astype(float)/255 # Dark2 Yellow
    c_dict['L2_'] = np.array([55,126,184]).astype(float)/255 # blue
    c_dict['Mi1'] = np.array([166,118,29]).astype(float)/255 # Dark2 dark yellow
    c_dict['Mi4'] = np.array([217,95,2]).astype(float)/255 # Dark2 orange
    c_dict['Tm3'] = np.array([231,41,138]).astype(float)/255 # Dark2 magent
    c_dict['Tm1'] = np.array([231,41,138]).astype(float)/255 # Dark2 magent
    c_dict['Tm2'] = np.array([166,118,29]).astype(float)/255 # Dark2 dark yellow
    c_dict['Tm4'] = np.array([217,95,2]).astype(float)/255 # Dark2 orange
    c_dict['Tm9GMR42C08Gal4_iGluSnfr'] = np.array([27,158,119]).astype(float)/255 #Dark2 Weird green

    
    c_dict['T45'] = np.array([77,77,77]).astype(float)/255 # dark gray
    c_dict['Tm1GMR74G01Gal4_iGluSnfr'] = np.array([231,41,138]).astype(float)/255 # Dark2 magent
    c_dict['T5_Control'] = np.array([102,166,30]).astype(float)/255 # Dark2 Green
    c_dict['L3_Kir'] = np.array([128,177,211]).astype(float)/255 # Dark2 magenta
    c_dict['R64G09-Recomb-lexAopGC6f'] = np.array([166,86,40]).astype(float)/255
    c_dict['Contr_R64G09-T4T5-Recomb-Hete__UAS-Kir'] =np.array([102,166,30]).astype(float)/255 # Dark2 Green
    c_dict['L3Sil_R64G09-T4T5-Recomb-Hete_R64B03-AD__UAS-Kir_R14B0-DBD'] = np.array([128,177,211]).astype(float)/255 # Dark2 magenta

    c_dict["ctr_dSNR-L3"] = c_dict['dark_gray']
    c_dict["ctr_UAS-ShiTS-L3"] = c_dict['green1']
    c_dict["exp_dNlg-CD19-Tm9_dSNR-L3"] = c_dict['magenta']
    c_dict["exp_ICAM-CD19-Tm9_dSNR-L3"] = c_dict['red']
    c_dict["ctr_dNlg-CD19-Tm9"] = c_dict['light_gray']
    
    c_dict["ctr1_Tm9_no_GluClMiMIC"] = c_dict['dark_gray']
    c_dict["ctr2_Tm9_no_Flp"] = c_dict['light_gray']
    c_dict["exp_Tm9_Flp_GC6f_GluClflpSTOP_GluClMiMIC"] = c_dict['magenta']


    c_dict["ctr1_Tm1_no_GluClMiMIC"] = c_dict['dark_gray']
    c_dict["ctr2_Tm1_no_Flp"] = c_dict['light_gray']
    c_dict["exp_Tm1_Flp_GC6f_GluClflpSTOP_GluClMiMIC"] = c_dict['magenta']

    c_dict["ctr0_Tm9rec_x_UAS_Kir"] = c_dict['dark_gray']
    c_dict["ctrlTm16_Tm9rec_x_Tm16Gal4"] = c_dict['light_gray']
    c_dict["expTm16_Tm9rec_UAS_Kir_x_Tm16Gal4"] = c_dict['red']
    c_dict["ctrlDm12_Tm9rec_x_Dm12Gal4"] = c_dict['light_gray']
    c_dict["expDm12_Tm9rec_UAS_Kir_x_Dm12Gal4"] = c_dict['brown']
    c_dict["expDm4_Tm9rec_UAS_Kir_x_Dm4Gal4"] = c_dict['magenta']
    
    c = []
    c.append(c_dict['dark_gray'])
    c.append(c_dict['light_gray'])
    c.append(c_dict['green1']) # Green
    c.append(c_dict['orange']) # Orange
    c.append(c_dict['red']) # Red
    c.append(c_dict['magenta']) # magenta
    c.append(c_dict['purple'])# purple
    c.append(c_dict['green2']) # Green
    c.append(c_dict['yellow']) # Yellow
    c.append(c_dict['brown']) # Brown
    
    
    return c, c_dict


def compute_over_samples_groups(data = None, group_ids= None, error ='std',
                                   experiment_ids = None):
    """ 
    
    Computes averages and std or SEM of a given dataset over samples and
    groups
    """
    # Input check
    if (data is None):
        raise TypeError('Data missing.')
    elif (group_ids is None):
        raise TypeError('Sample IDs are missing.')
    elif (experiment_ids is None):
        raise TypeError('Experiment IDs are missing.')
    
    # Type check and conversion to numpy arrays
    if (type(data) is list):
        data = np.array(data)
    if (type(group_ids) is list):
        group_ids = np.array(group_ids)
    if (type(experiment_ids) is list):
        experiment_ids = np.array(experiment_ids)
    
    
    data_dict = {}
    data_dict['experiment_ids'] = {}
    
    unique_experiment_ids = np.unique(experiment_ids)
    for exp_id in unique_experiment_ids:
        data_dict['experiment_ids'][exp_id] = {}
        data_dict['experiment_ids'][exp_id]['over_samples_means'] = []
        data_dict['experiment_ids'][exp_id]['over_samples_errors'] = np.array([])
        data_dict['experiment_ids'][exp_id]['uniq_group_ids'] = np.array([])
        
        data_dict['experiment_ids'][exp_id]['all_samples'] = \
            data[experiment_ids == exp_id]
        
        curr_groups = group_ids[np.argwhere(experiment_ids == exp_id)]
        
        for group in np.unique(curr_groups):
            curr_mask = (group == group_ids)
            curr_data = np.nanmean(data[curr_mask],axis=0)
            if error == 'std':
                err = np.nanstd(data[curr_mask],axis=0)
            elif error == 'SEM':
                err = \
                    np.nanstd(data[curr_mask],axis=0) / np.sqrt(np.shape(data[curr_mask])[0])
            data_dict['experiment_ids'][exp_id]['over_samples_means'].append(\
                 curr_data)
            data_dict['experiment_ids'][exp_id]['over_samples_errors'] = \
                np.append(data_dict['experiment_ids'][exp_id]['over_samples_errors'],
                          err)
            data_dict['experiment_ids'][exp_id]['uniq_group_ids'] = \
                np.append(data_dict['experiment_ids'][exp_id]['uniq_group_ids'],
                          group)
        data_dict['experiment_ids'][exp_id]['over_groups_mean'] = \
            np.nanmean(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0)
        if error == 'std':
            err = \
                np.nanstd(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0)
        elif error == 'SEM':
            err = \
                np.nanstd(data_dict['experiment_ids'][exp_id]['over_samples_means'],
                    axis=0) / \
                    np.sqrt(np.shape(data_dict['experiment_ids'][exp_id]['over_samples_means'])[0])
        data_dict['experiment_ids'][exp_id]['over_groups_error'] =err
            
    return data_dict
        
def bar_bg(all_samples, x, color='k', scat_s =7,ax=None, yerr=None, 
           errtype='std',alpha = .6,width=0.8,label=None):
    
    """ Nice bar plot """
    if yerr is None:
        if errtype == 'std':
            yerr = np.std(all_samples)
        elif errtype == 'SEM':
            yerr = np.std(all_samples) / np.sqrt((len(all_samples)))
            
    if ax is None:
        asd =1
    else:
        ax.bar(x, np.mean(all_samples), color=color,alpha=alpha,
               width=width,label=label)
        x_noise = np.random.normal(size=len(all_samples))
        x_noise = (x_noise-min(x_noise))/(max(x_noise)-min(x_noise)) -0.5
        x_noise = x_noise/2
        scatt_x = np.zeros(np.shape(all_samples)) + x + x_noise
        ax.scatter(scatt_x,all_samples,color=color,s=scat_s)
        
        
        markers, caps, bars = ax.errorbar(x, np.mean(all_samples),
                                          yerr=yerr,fmt='.', 
                                          ecolor='black',capsize=0)
        markers.set_alpha(0)
        # loop through bars and caps and set the alpha value
        [bar.set_alpha(alpha) for bar in bars]
    return ax


            
            
def apply_threshold_df(threshold_dict, df):
    
    if threshold_dict is None:
        print('No threshold used.')
        return df
    
    pass_bool = np.ones((1,len(df)))
    
    for key, value in threshold_dict.items():

        pass_bool = pass_bool * np.array((df[key] > value))
        
    threshold_df = df[pass_bool.astype(bool)[0]]
   
    return threshold_df



def calc_MI(x, y, bins):
    """ Calculating MI """
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    
    # Uses natural log to express information so you can divide by log2 to
    # get bits
    
    return mi/np.log(2)
    

def convert_cd_to_photons(cdVal,wavelength=475):
    # From Madhura's matlab script
    cdSrPerWatt=683       # Wikipedia for Candela
    h=6.62607 * 10**(-34) # Planck's constant, J.s
    c=299792458           # speed of light, m/s
    acceptanceAngle=8.23  # degrees, from Gonzalez-Bellido et al
    ommatidiumDia=16.85*10**(-6) # m, Gonzalez-Bellido et al, 2011 Supp

    #converts cd/sq m values to photons per s per receptor
    #wavelength has to be in nm
    ## import photopic luminosity function (LS-100 uses it) and convert cd to watt
    #luminosity = pd.read_fwf('photopic luminosity function.txt')
    #read_in_table = pd.read_fwf('photopic luminosity function.txt')
    ##spectrum = csvread('relative spectral distribution_LED panels.csv')
    ##spectrum=spectrum(1:2:end,:); # downsampling to match luminosity function data
    luminosity_df = pd.read_csv('photopic_luminosity_function.txt', sep="\t",
                                header=None)
    luminosity = luminosity_df[luminosity_df[0]==wavelength][1]
    luminosity = float(luminosity)
    wattVal = cdVal/(cdSrPerWatt*luminosity)   # in the row of wavelength (multiples of 5)
    ## convert cd to photons per s per sr per sq m
    energyPerPhoton=h*c/(wavelength*10**-9)     # J
    photonsPerSSrSqM=wattVal/energyPerPhoton   # from here, we work on replacing per sr per sq m by per receptor
    ## convert acceptance angle to steradians (Integration over Gaussian angular sensitivity function, Dubs et al 1981)
    acceptanceAngleRad=acceptanceAngle*np.pi/180  # conversion to Rad
    acceptanceSteradians=acceptanceAngleRad**2*np.pi/(4*np.log(2))
    ## flux to one ommatidium/receptor
    ommatidiumArea=np.pi*(ommatidiumDia/2)**2      # sq m
    photonVal=photonsPerSSrSqM*ommatidiumArea*acceptanceSteradians # as required

    return photonVal