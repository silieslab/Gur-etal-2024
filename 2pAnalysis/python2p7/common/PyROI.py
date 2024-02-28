#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:09:54 2019

@author: burakgur
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import os
# import sima
from scipy import optimize
from scipy.stats import linregress
from scipy.stats.stats import pearsonr
from scipy import fft, signal, interpolate
from scipy.signal import blackman
from itertools import permutations

import post_analysis_core as pac
class ROI: 
    """A region of interest from an image sequence """
    
    def __init__(self,Mask = None, experiment_info = None,imaging_info = None, uniq_id = None): 
        """ 
        Initialized with a mask and optionally with experiment and imaging
        information
        """
        if (Mask is None):
            raise TypeError('ROI: ROI must be initialized with a mask (numpy array)')
        if (experiment_info is not None):
            self.experiment_info = experiment_info
        if (imaging_info is not None):
            self.imaging_info = imaging_info
            
        if (uniq_id is None):
            self.uniq_id = id(self) # Generate a unique ID everytime it is not given
        else:
            self.uniq_id = uniq_id # Useful during transfer

        self.mask = Mask
        
        
    def __str__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
    
    def __repr__(self):
        return '<ROI:{_id}>'.format(_id = self.uniq_id)
        
    def setSourceImage(self, Source_image):
        
        if np.shape(Source_image) == np.shape(self.mask):
            self.source_image = Source_image
        else:
            raise TypeError('ROI: source image dimensions has to match with\
                            ROI mask.')
    
    def showRoiMask(self, cmap = 'Dark2',source_image = None):
        
        if (source_image is None):
            source_image = self.source_image
        curr_mask = np.array(copy.deepcopy(self.mask),dtype=float)
        curr_mask[curr_mask==0] = np.nan
        plt.imshow(source_image,alpha=0.8,cmap = 'gray')
        plt.imshow(curr_mask, alpha=0.4,cmap = cmap)
        plt.axis('off')
        plt.title(self)

    def calculateDf(self,method='mean',moving_avg = False, bins = 3):
        try:
            self.raw_trace
        except NameError:
            raise NameError('ROI: for deltaF calculations, a raw trace \
                            needs to be provided: a.raw_trace')
            
        if method=='mean':
            df_trace = (self.raw_trace-self.raw_trace.mean(axis=0))/(self.raw_trace.mean(axis=0))
            self.baseline_method = method
        
        if moving_avg:
            self.df_trace = movingaverage(df_trace, bins)
        else:
            self.df_trace = df_trace
            
        return self.df_trace

    def plotDf(self, line_w = 1,color=plt.cm.Dark2(0)):
        a = plt.axes()
        a.plot(self.df_trace, lw=line_w, alpha=.8,color=color)
       
        try:
            stim_vals = self.stim_info['processed']['epoch_trace_frames']
            # Make normalized values of stimulus values for plotting
            stim_vals /= stim_vals.max()
            stim_vals *= self.df_trace.max()
            
            a.plot(stim_vals,'--', lw=1, alpha=.6,color='k')
            plt.show()
        except KeyError:
            print('No raw stimulus information found')
        return a
def dataToList(rois, data_name_list):
    """ Generates a dictionary with desired variables from ROIs.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    data_name_list: list
        A list of strings with desired variable names. The variables should be 
        written as defined in the ROI_bg class. 
        
    Returns
    =======
    
    roi_data_dict : dictionary 
        A dictionary with keys as desired data variable names and values as
        list of data.
    """   
    class my_dictionary(dict):  
  
        # __init__ function  
        def __init__(self):  
            self = dict()  
              
        # Function to add key:value  
        def add(self, key, value):  
            self[key] = value  
    
    roi_data_dict = my_dictionary()
    
    # Generate an empty dictionary
    for key in data_name_list:
        roi_data_dict.add(key, [])
    
    # Loop through ROIs and get the desired data            
    for iROI, roi in enumerate(rois):
        for key, value in roi_data_dict.items(): 
            if key in roi.__dict__.keys():
                value.append(roi.__dict__[key])
            else:
                value.append(np.nan)
    return roi_data_dict
    
def getMasksImage(rois):
    """ Generates an image of masks.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    
    Returns
    =======
    
    roi_data_dict : np array
        A numpy array with masks depicted in different integers
    """   
    roi_masks_image = np.array(map(lambda idx_roi_pair : \
                             idx_roi_pair[1].mask.astype(float) * (idx_roi_pair[0]+1), 
                             list(enumerate(rois)))).sum(axis=0)
    
    roi_masks_image[roi_masks_image==0] = np.nan
    
    
    return roi_masks_image

def generatePropertyMasks(rois, prop = 'BF'):
    """ Generates images of masks depending on DSI CSI Rel and BF

    TODO: Is it possible to generate something independent?
    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    
    Returns
    =======
    
    roi_data_dict : np array
        A numpy array with masks depicted in different integers
    """  
    if prop == 'BF':
        BF_image = np.zeros(np.shape(rois[0].mask))
        
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            BF_image = BF_image + (curr_mask * roi.BF)
        BF_image[BF_image==0] = np.nan
        
        return BF_image
    elif prop == 'CS':
        CSI_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_CS = roi.CS
            if curr_CS == 'OFF':
                curr_CSI = roi.CSI * -1
            else:
                curr_CSI = roi.CSI
            curr_mask = roi.mask.astype(int)
            CSI_image = CSI_image + (curr_mask * curr_CSI)
        CSI_image[CSI_image==0] = np.nan
        return CSI_image
    elif prop =='DSI':
        DSI_image  = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_DSI = roi.DSI
                
            curr_mask = roi.mask.astype(int)
            DSI_image = DSI_image + (curr_mask * curr_DSI)
        DSI_image[DSI_image==0] = np.nan
        return DSI_image
    elif prop =='PD':
        PD_image  = np.full(np.shape(rois[0].mask),np.nan)
        alpha_image  = np.full(np.shape(rois[0].mask),np.nan)
        for roi in rois:
            PD_image[roi.mask] = roi.PD
            alpha_image[roi.mask] = roi.DSI
        
        return PD_image
    
    elif prop == 'reliability':
        Corr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            Corr_image = Corr_image + (curr_mask * roi.reliability)
            
        Corr_image[Corr_image==0] = np.nan
        return Corr_image
    
    elif prop == 'SNR':
        snr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            snr_image = snr_image + (curr_mask * roi.SNR)
            
        snr_image[snr_image==0] = np.nan
        return snr_image
    elif prop == 'corr_fff':
        Corr_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            Corr_image = Corr_image + (curr_mask * roi.corr_fff)
            
        Corr_image[Corr_image==0] = np.nan
        return Corr_image
    elif prop == 'max_response':
        max_image = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            max_image = max_image + (curr_mask * roi.max_response)
            
        max_image[max_image==0] = np.nan
        return max_image
    elif prop == 'slope':
        
        slope_im = np.zeros(np.shape(rois[0].mask))
        for roi in rois:
            curr_mask = roi.mask.astype(int)
            slope_im = slope_im + (curr_mask * roi.slope)
            
        slope_im[slope_im==0] = np.nan
        return slope_im
    
    else:
        raise TypeError('Property %s not available for color mask generation' % prop)
        return 0

def interpolate_signal(signal, sampling_rate, int_rate=10,int_time = None):
    """
    """

    timeV = np.linspace(1/sampling_rate,(len(signal)+1)/sampling_rate,num = len(signal)) 
    if int_time == None:
        timeVI = np.linspace(1/float(int_rate), (len(signal)+1)/sampling_rate, num = int(round((len(signal)/sampling_rate)*int_rate)))
    else:
        timeVI = np.linspace(1/float(int_rate), int_time, num = int(round(int_time*int_rate)))
                         
    return np.interp(timeVI, timeV, signal)

def generateROIs(roi_masks, category_masks, category_names, source_im,
                           experiment_info = None, imaging_info =None):
    """ Generates ROI instances and adds the category information.

    Parameters
    ==========
    roi_masks : list
        A list of ROI masks in the form of numpy arrays.
        
    category_masks: list
        A list of category masks in the form of numpy arrays.
        
    category_names: list
        A list of category names.
        
    source_im : numpy array
        An array containing a representation of the source image where the 
        ROIs are found.
    
    Returns
    =======
    
    rois : list 
        A list containing instances of ROI_bg
    """    
    # if type(roi_masks) == sima.ROI.ROIList:
    #     roi_masks = list(map(lambda roi : np.array(roi)[0,:,:], roi_masks))
        
    # Generate instances of ROI from the masks
    rois = map(lambda mask : ROI(mask, experiment_info = experiment_info,
                                    imaging_info=imaging_info), roi_masks)

    def assign_region(roi, category_masks, category_names):
        """ Finds which layer the current mask is in"""
        for iLayer, category_mask in enumerate(category_masks):
            if np.sum(roi.mask*category_mask):
                roi.category = category_names[iLayer]
    

    # Add information            
    for iROI, roi in enumerate(rois):
        # Regions are assigned if there are category masks and names provided
        assign_region(roi, category_masks, category_names)
        roi.setSourceImage(source_im)

        roi.number_id = iROI # Assign ROIs numbers

        
    return rois

def getTimeTraces(rois, time_series,df_method = 'mean'):
    """ Computes the time traces of each ROI given a time series """
    # dF/F calculation
    for roi in rois:
        roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
        roi.calculateDf(method=df_method,moving_avg = False)
    return rois

def averageTrials(rois):
    """ Trial averaging """

    # Initialize trace dictionaries
    for roi in rois:
        roi.resp_traces = {} # Storing traces for stimulus epochs
        roi.whole_traces = {} # Storing traces for the whole trace (bg-stimEpoch-bg)
        roi.resp_trace_lens = {} # Length
        roi.whole_trace_lens = {} # Length
    # Compute the lengths
    stim_info = rois[0].stim_info
    randomization_condition = stim_info['meta']['randomization_condition']
    stim_coords = stim_info['processed']['trial_coordinates']

    # Initialize the randomization related variables
    if (randomization_condition == 1) or (randomization_condition == 3):
        baseline_epoch = 'epoch_1'
        resp_start_idx = 1
        resp_end_idx = 2
        base_start_idx = 0
        base_end_idx = 3

        base_dur = stim_info['meta']['epoch_infos'][baseline_epoch]['total_dur_sec']
        base_len = int(np.floor(base_dur*rois[0].imaging_info['frame_rate']))
    else:
        baseline_epoch = None
        resp_start_idx = 0
        resp_end_idx = 1
        base_start_idx = 0
        base_end_idx = 1
        base_len = 0

    # Trial averaging
    for epoch , epoch_coords in stim_coords.iteritems():
        if epoch == baseline_epoch: # Skip the baseline epoch
            continue
        if stim_info['meta']['epoch_infos'][epoch]['stim_type'] == 'movingStripe-v1':
            epoch_dur, _ = findMovingStripeDur(stim_info['meta']['epoch_infos'][epoch],stim_info['meta']['proj_params'])
        else:
            epoch_dur = stim_info['meta']['epoch_infos'][epoch]['total_dur_sec']

        resp_len = int(np.floor(epoch_dur*rois[0].imaging_info['frame_rate']))
        trial_len = base_len + resp_len + base_len
        # resp_len = np.min([trial_c[resp_end_idx]-trial_c[resp_start_idx] for trial_c in epoch_coords])
        # trial_len = np.min([trial_c[base_end_idx]-trial_c[base_start_idx] for trial_c in epoch_coords])

        trial_num = len(epoch_coords)
        for roi in rois:
            roi.base_len = base_len
            resp_mat = np.zeros([resp_len,trial_num])
            trial_mat = np.zeros([trial_len,trial_num])
            # Go over trials
            for iTrial in range(trial_num):
                trial_coords = epoch_coords[iTrial]
                resp_start = trial_coords[resp_start_idx]
                trial_start = trial_coords[base_start_idx]
                resp_mat[:,iTrial] = roi.df_trace[resp_start:resp_start+resp_len]
                trial_mat[:,iTrial] = roi.df_trace[trial_start:trial_start+trial_len]

            # Add the responses to the ROI
            roi.resp_trace_lens[epoch] = resp_len
            roi.whole_trace_lens[epoch] = trial_len
            roi.resp_traces[epoch] = resp_mat.mean(axis=1)
            roi.whole_traces[epoch] = trial_mat.mean(axis=1)
            
    return rois
    
def calculateReliability(rois):
    """ Calculates the correlation between single trials and assigns as reliability for each ROI """
    
    for roi in rois:
        # Compute the lengths
        stim_info = roi.stim_info
        randomization_condition = stim_info['meta']['randomization_condition']
        stim_coords = stim_info['processed']['trial_coordinates']

        roi.reliabilities = []
        # Initialize the randomization related variables
        if (randomization_condition == 1) or (randomization_condition == 3):
            baseline_epoch = 'epoch_1'
            resp_start_idx = 1
            resp_end_idx = 2
            base_start_idx = 0
            base_end_idx = 3

            base_dur = stim_info['meta']['epoch_infos'][baseline_epoch]['total_dur_sec']
            base_len = int(np.floor(base_dur*roi.imaging_info['frame_rate']))
        else:
            baseline_epoch = None
            resp_start_idx = 0
            resp_end_idx = 1
            base_start_idx = 0
            base_end_idx = 1
            base_len = 0
        
        for epoch , epoch_coords in stim_coords.iteritems():
            if epoch == baseline_epoch: # Skip the baseline epoch
                continue
            epoch_dur = stim_info['meta']['epoch_infos'][epoch]['total_dur_sec']

            resp_len = int(np.floor(epoch_dur*rois[0].imaging_info['frame_rate']))
            trial_len = base_len + resp_len + base_len
            
            
            trial_num = len(epoch_coords)
            resp_mat = np.zeros([resp_len,trial_num])
            for iTrial in range(trial_num):
                trial_coords = epoch_coords[iTrial]
                resp_start = trial_coords[resp_start_idx]
                resp_mat[:,iTrial] = roi.df_trace[resp_start:resp_start+resp_len]
            
            perm = permutations(range(trial_num), 2) 
            coeff =[]
            for iPerm, pair in enumerate(perm):
                curr_coeff, pval = pearsonr(resp_mat[:,pair[0]],
                                            resp_mat[:,pair[1]])
                coeff.append(curr_coeff)
                
            roi.reliabilities.append(np.array(coeff).mean())
        roi.reliability = np.max(roi.reliabilities)
    return rois

def plotAllTraces(rois, fig_save_dir = None):
    plt.close('')
    plt.style.use('default')
    stim_vals = rois[0].stim_info['processed']['epoch_trace_frames']
    stim_vals = stim_vals/stim_vals.max()
    stim_vals -= stim_vals.min()
    plt.plot(stim_vals,'--', lw=1, alpha=.6,color='k')

    scaler = float(len(rois))
    for idx, roi in enumerate(rois):
        plot_trace = (roi.df_trace+idx)/scaler
        plt.plot(plot_trace,lw=1/3.0, alpha=1)

    plt.xlabel('Frames')
    plt.title(rois[0].experiment_info['MovieID'])
    fig = plt.gcf()

    if fig_save_dir is not None:
        f_name = 'Traces_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f_name, bbox_inches='tight',transparent=False,dpi=300)
        plt.gcf()

    return fig

def data_to_list(rois, data_name_list):
    """ Generates a dictionary with desired variables from ROIs.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    data_name_list: list
        A list of strings with desired variable names. The variables should be 
        written as defined in the ROI_bg class. 
        
    Returns
    =======
    
    roi_data_dict : dictionary 
        A dictionary with keys as desired data variable names and values as
        list of data.
    """   
    class my_dictionary(dict):  
  
        # __init__ function  
        def __init__(self):  
            self = dict()  
              
        # Function to add key:value  
        def add(self, key, value):  
            self[key] = value  
    
    roi_data_dict = my_dictionary()
    
    # Generate an empty dictionary
    for key in data_name_list:
        roi_data_dict.add(key, [])
    
    # Loop through ROIs and get the desired data            
    for iROI, roi in enumerate(rois):
        for key, value in roi_data_dict.items(): 
            if key in roi.__dict__.keys():
                value.append(roi.__dict__[key])
            else:
                value.append(np.nan)
    return roi_data_dict

def threshold_ROIs(rois, threshold_dict):
    """ Thresholds given ROIs and returns the ones passing the threshold.

    Parameters
    ==========
    rois : list
        A list of ROI_bg instances.
        
    threshold_dict: dict
        A dictionary with desired ROI_bg property names that will be 
        thresholded as keys and the corresponding threshold values as values. 
    
    Returns
    =======
    
    thresholded_rois : list 
        A list containing instances of ROI_bg which pass the thresholding step.
    """
    # If there is no threshold
    if threshold_dict is None:
        print('No threshold used.')
        return rois
    vars_to_threshold = threshold_dict.keys()
    
    roi_data_dict = data_to_list(rois, vars_to_threshold)
    
    pass_bool = np.ones((1,len(rois)))
    
    for key, value in threshold_dict.items():
        
        if type(value) == tuple:
            if value[0] == 'b':
                pass_bool = \
                    pass_bool * (np.array(roi_data_dict[key]).flatten() > value[1])
                
            elif value[0] == 's':
                pass_bool = \
                    pass_bool * (np.array(roi_data_dict[key]).flatten() < value[1])
            else:
                raise TypeError("Tuple first value not understood: should be 'b' for bigger than or 's' for smaller than")
                
        else:
            pass_bool = pass_bool * (np.array(roi_data_dict[key]).flatten() > value)
    
    pass_indices = np.where(pass_bool)[1]
    
    thresholded_rois = []
    for idx in pass_indices:
        thresholded_rois.append(rois[idx])
    
    return thresholded_rois

def low_pass(trace, frame_rate, crit_freq=3,plot=False):
    """ Applies a 3rd order butterworth low pass filter for getting rid of noise
    """
    wn_norm = crit_freq / (frame_rate/2)
    b, a = signal.butter(3, wn_norm, 'low')
    filt_trace = signal.filtfilt(b, a, trace)
    
    if plot:
        fig1, ax1 = plt.subplots(2, 1, sharex=True, sharey=False)
        ax1[0].plot(trace,lw=0.4)
        ax1[1].plot(filt_trace,lw=0.4)

    return filt_trace

def interpolate_data_dyuzak(stimtimes, stimframes100hz, dsignal, imagetimes, freq):
    """Interpolates the stimulus frame numbers (*stimframes100hz*), signal
    traces (*dsignal*) by using the
    stimulus time (*stimtimes*)  and the image time stamps (*imagetimes*)
    recorded. Interpolation is done to a frequency (*freq*) defined by the
    user.
    recorded in

    Parameters
    ----------
    stimtimes : 1D array
        Stimulus time stamps obtained from stimulus_output file (with the
        rate of ~100Hz)
    stimframes100hz : 1D array
        Stimulus frame numbers through recording (with the rate of ~100Hz)
    dsignal : mxn 2D array
        Fluorescence responses of each ROI. Axis m is the number of ROIs while
        n is the time points of microscope recording with lower rate (10-15Hz)
    imagetimes : 1D array
        The time stamps of the image frames with the microscope recording rate
    freq : int
        The desired frequency to interpolate

    Returns
    -------
    newstimtimes : 1D array
        Stimulus time stamps with the rate of *freq*
    dsignal : mxn 2D array
        Fluorescence responses of each ROI with the rate of *freq*
    imagetimes : 1D array
        The time stamps of the image frames with the rate of *freq*
    """
    
    # Interpolation of stimulus frames and responses to freq

    # Creating time vectors of original 100 Hz(x) and freq Hz sampled(xi)
    # x = vector with 100Hz rate, xi = vector with user input rate (freq)
    x = np.linspace(0,len(stimtimes),len(stimtimes))
    xi = np.linspace(0,len(stimtimes),
                     np.round(int((np.max(stimtimes)-np.min(stimtimes))*freq)+1))

    # Get interpolated stimulus times for 20Hz
    # stimtimes and x has same rate (100Hz)
    # and newstimtimes is interpolated output of xi vector
    newstimtimes = np.interp(xi, x, stimtimes)
    newstimtimes =  np.array(newstimtimes,dtype='float32')

    # Get interpolated stimulus frame numbers for 20Hz
    # Below stimframes is a continuous function with stimtimes as x and
    # stimframes100Hz as y values
    stimframes = interpolate.interp1d(stimtimes,stimframes100hz,kind='nearest')
    # Below interpolated stimulus times are given as x values to the stimtimes
    # function to find interpolated stimulus frames (y value)
    stimframes = stimframes(newstimtimes)
    stimframes = stimframes.astype('int')

    #Get interpolated responses for 20Hz
    dsignal1 = np.empty(shape=(len(dsignal),
                               len(newstimtimes)),dtype=dsignal.dtype)
    dsignal=np.interp(newstimtimes, imagetimes, dsignal)


    return (newstimtimes, dsignal, stimframes)

def reverseCorrelation(rois,cf=2,filtering=True,
                                 poly_fitting=True,t_window=2000,
                                 stim_up_rate=20):
    """ Reverse correlation analysis 
        
    """
    
    
    freq = stim_up_rate # The update rate of stimulus frames is 20Hz
    window = int(t_window/(1.0/stim_up_rate * 1000)) 
    snippet = window

    for roi in rois:
        stim = roi.wn_stim
        # Stimulus related
        stim_info = roi.stim_info
        stim_timings = np.array(stim_info['sample_time']).astype(float)
        stim_frame_trace = np.array(stim_info['stim_info1']).astype(int)
        image_frame_trace = np.array(stim_info['imaging_frame']).astype(int)
        epoch_vals = np.array(stim_info['stimulus_epoch']).astype(int) # Epoch values
        frame_timings = roi.imaging_info['frame_timings']
        # Response related stuff
        raw_signal = roi.raw_trace
        
        fps = roi.imaging_info['frame_rate']
        if filtering:
            filtered = low_pass(raw_signal, fps, crit_freq=cf,plot=False)
            trace = filtered.copy()
        else:
            trace = raw_signal.copy()
        
        if poly_fitting:
            fit_x = np.linspace(0,len(trace),len(trace))
            poly_vals = \
                np.polyfit(fit_x, trace, 4)
            fit_trace = np.polyval(poly_vals,fit_x)
            trace = trace-fit_trace
        trace = trace-trace.min()
        
        # df/f using the gray epochs
        bg_frames = image_frame_trace[epoch_vals==1.0]
        bg_frames = bg_frames[bg_frames<len(raw_signal)] # Just take imaged frames since stim output sometimes produces 2-3 extra frames
        bg_mean = trace[np.unique(bg_frames)].mean()
        df_trace = (trace - bg_mean)/bg_mean

        newstimtimes, df, stimframes = interpolate_data_dyuzak(stim_timings,
                                                             stim_frame_trace,
                                                             df_trace,
                                                             frame_timings,
                                                             freq)

        # Finding where the stim frames is more preceeding frame length
        # Starting from these indices we will do the rev corr
        booleans = stimframes>=snippet
        # Padding stimulus frames by 0 (grey interleave) values
        padframes = np.concatenate(([0],stimframes,[0]))
        # Finding the points where frames shift between grey to frame numbers
        difs = np.diff(padframes>0)
        # Finding the indices of epoch start and end
        # rows = epoch no, columns = start and end
        epochind=np.where(difs==1)[0].reshape(-1,2)
        
        # Take the first epoch only since it contains a while WN stimulus
        analyzelimit = epochind[0,:]    #where to splice
        # Splicing data
        stimframesused = stimframes[analyzelimit[0]:analyzelimit[1]]
        dfused = df[analyzelimit[0]:analyzelimit[1]]
        
        # Used stimulus is sliced by using the frame information
        
        stimused = stim[np.min(stimframesused)-1:np.max(stimframesused),:,:]
        # Get unique frame numbers from trials
        uniquestim = np.unique(stimframesused)
        # Find frame nos grater than preceeding time window as a boolean matrix
        boolean2 = uniquestim >= snippet
        
        
        #centering df (mean substraction)
        
        centereddf = (dfused.T-np.mean(dfused)).T
        #centereddf = dfused[roi_ind] # no centering (mh, 11.02.19)
    
        #centering stimulus (mean substraction)
        stimused = stimused - np.mean(stimused)
        #stimused = stimused - 0.5 #shift around 0 (mh, 11.02.19) 
    
        avg = np.zeros(shape=(snippet,stimused.shape[1],stimused.shape[2]))
        # For loop iterates through different stimulus frame numbers
        # Finds the data where the specific frame is shown and calculates
        # sta
        for ii in range(snippet-1,len(uniquestim)):
            # Calculate means of responses to specific frame with ii index
            responsechunk = centereddf[np.where(stimframesused==uniquestim[ii])[0]]
            responsemean = np.mean(responsechunk)
            # Create a tiled matrix for fast calculation
            response = np.tile(responsemean,(stimused.shape[2],stimused.shape[1],snippet)).T
            # Find the stimulus values in the window and get tiled matrix
            stimsnip = stimused[uniquestim[ii]-snippet:uniquestim[ii],:,:]
            
            # Fast calculation is actually is a multiplication
            avg += np.multiply(response,stimsnip)
        sta = avg/(len(uniquestim)-snippet)     # Average with the number of additions
        roi.sta = sta
        
    return rois

def plotRFs_WN(rois, f_w=None,number=None,cmap='coolwarm',fig_save_dir=None,fit_plot=False):
    import random
    plt.close('all')
    colors = pac.run_matplotlib_params()
    if (number == None) or (f_w==None):
        f_w = 5
        if len(rois)>10:
            number=10
        else:
            number = len(rois)
    elif number > len(rois):
        number = len(rois)
    f_w = f_w*2
    # Randomize ROIs
    copy_rois = copy.deepcopy(rois)
    random.shuffle(copy_rois)
    max_n=np.array(map(lambda roi : np.max(roi.sta), rois)).max()
    min_n=np.array(map(lambda roi : np.max(roi.sta), rois)).min()
        
    time_d = rois[0].sta.shape[0]
    sta_d1 = rois[0].sta.shape[1]
    sta_d2 = rois[0].sta.shape[2]
    
    if number <= f_w/2:
        dim1= number
        dim2 = 1
    elif number/float(f_w/2) > 1.0:
        dim1 = f_w
        dim2 = int(np.ceil(number/float(f_w)))
    fig, ax1 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1*1.5, dim2*1.5))
    ax = ax1.flatten()
    for idx, roi in enumerate(rois):
        if idx == number:
            break
        
        if sta_d1 == 1:
            curr_sta = roi.sta[:,0,:].T
        elif sta_d2 ==1:
            curr_sta = roi.sta[:,:,0].T
        else:
            max_t = np.where(np.abs(roi.sta)==np.abs(roi.sta).max())[0][0]
            curr_sta = roi.sta[max_t,:,:]
            

        sns.heatmap(curr_sta, cmap=cmap, ax=ax[idx], cbar=False,vmax=max_n,
                    center=0)
        if fit_plot:
            ax[idx].contour(roi.rf_fit,alpha=.3,cmap=cmap)
        if 'center_coords' in roi.__dict__.keys():
            ax[idx].set_title('X:{x}| Y:{y}| \nRsq:{rs}'.format(x=int(roi.center_coords[1]),
                y=int(roi.center_coords[0]),rs = np.round(roi.rf_fit_rsq,2)))

        ax[idx].axis('off')
    for axs in ax:
        axs.axis('off')

    if fig_save_dir is not None:
        if fit_plot:
            f_name = 'RFs_fit' 
        else :
            f_name = 'RFs' 
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f_name, bbox_inches='tight',transparent=False)
        plt.gcf()

    return fig

def plotRF_WNstripes(rois, fig_save_dir = None):

    
    for roi in rois:
        max_t = np.where(np.abs(roi.sta)==np.abs(roi.sta).max())[0][0]
        plot_sta = roi.sta[max_t,:,:]
        
        plt.subplot(121)
        plt.title(roi.uniq_id)
        roi.showRoiMask()
        plt.subplot(122)
        plt.title('Spatial RF t:%s' % str(max_t))
        plt.imshow(plot_sta)
        fig = plt.gcf()

        if fig_save_dir is not None:
            f_name = 'STA_%s' % (roi.uniq_id)
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% f_name, bbox_inches='tight',transparent=False)
            plt.gcf()

    return fig

def plotRF_WNsquare(rois, fig_save_dir = None):

    
    for roi in rois:
        max_t = np.where(np.abs(roi.sta)==np.abs(roi.sta).max())[0][0]
        plot_sta = roi.sta[max_t,:,:]
        
        plt.subplot(121)
        plt.title(roi.uniq_id)
        roi.showRoiMask()
        plt.subplot(122)
        plt.title('Spatial RF t:%s' % str(max_t))
        plt.imshow(plot_sta)
        fig = plt.gcf()

        if fig_save_dir is not None:
            f_name = 'STA_%s' % (roi.uniq_id)
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% f_name, bbox_inches='tight',transparent=False)
            plt.gcf()

    return fig

def analyzeLuminanceEdges(rois,int_rate = 10):
        
    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None

        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.epoch_luminances = {}
        roi.edge_step_responses = {}
        roi.edge_absolute_responses = {}
        roi.int_whole_trace = {}
        roi.int_resp_trace = {}

        interpolated_len = len(interpolate_signal(roi.resp_traces[used_epochs[0]],roi.imaging_info['frame_rate'],int_rate,int_time=int(round(len(roi.resp_traces[used_epochs[0]])/roi.imaging_info['frame_rate'])))) 
        int_resp_traces = np.zeros((len(used_epochs),interpolated_len))

        for idx, epoch in enumerate(used_epochs):
            roi.epoch_luminances[epoch] = epoch_infos[epoch]['pre_lum']

            # Find edge responses
            pre_frames = int(epoch_infos[epoch]['pre_dur_sec'] * roi.imaging_info['frame_rate'])
            pre_lum_mean = roi.resp_traces[epoch][pre_frames-pre_frames/2:pre_frames].mean() # Half of pre-lum dur 500ms
            edge_resp = roi.resp_traces[epoch][pre_frames:].max()

            roi.edge_absolute_responses[epoch] = edge_resp
            roi.edge_step_responses[epoch] = edge_resp - pre_lum_mean

            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate=int_rate,int_time=epoch_infos[epoch]['total_dur_sec']) 
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate,int_time=int(round(len(roi.whole_traces[epoch])/roi.imaging_info['frame_rate']))) 

            int_resp_traces[idx, :] = roi.int_resp_trace[epoch]
        
        roi.edge_resp_traces_interpolated = int_resp_traces
        fp = roi.imaging_info['frame_rate']
        fp = np.tile(fp,len(used_epochs))

        aligned_traces = np.array(map(lambda trace,fp : np.roll(trace,len(trace)/2 - \
                                            int(np.argmax(trace[10:])+10)),
                     roi.edge_resp_traces_interpolated.copy(),fp))

        roi.max_aligned_traces = aligned_traces.copy()
        roi.concatenated_lum_traces = np.concatenate(aligned_traces)
        X = [epoch_infos[epoch]['pre_lum'] for epoch in used_epochs]
        Y = roi.edge_step_responses.values()
        roi.slope = linregress(X, np.transpose(Y))[0]

        # Sorted values
        luminances = roi.epoch_luminances.values()
        roi.sorted_luminances = np.sort(luminances)
        resps = np.array(roi.edge_step_responses.values())
        roi.sorted_edge_step_responses = resps[np.argsort(luminances)]

        resps_abs = np.array(roi.edge_absolute_responses.values())
        roi.sorted_edge_absolute_responses = resps_abs[np.argsort(luminances)]

        
        roi.max_resp = np.max(resps)
        
    return rois

def analyzeLuminanceEdges_L1_ONOFF(rois,int_rate = 10):
        
    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None

        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.epoch_luminances = {}
        roi.edge_step_responses = {}
        roi.edge_absolute_responses = {}
        roi.int_whole_trace = {}
        roi.int_resp_trace = {}

        interpolated_len = len(interpolate_signal(roi.resp_traces[used_epochs[0]],roi.imaging_info['frame_rate'],int_rate,int_time=int(round(len(roi.resp_traces[used_epochs[0]])/roi.imaging_info['frame_rate'])))) 
        int_resp_traces = np.zeros((len(used_epochs),interpolated_len))

        for idx, epoch in enumerate(used_epochs):
            roi.epoch_luminances[epoch] = epoch_infos[epoch]['edge_lum']

            # Find edge responses
            pre_frames = int(epoch_infos[epoch]['pre_dur_sec'] * roi.imaging_info['frame_rate'])
            pre_lum_mean = roi.resp_traces[epoch][pre_frames-pre_frames/2:pre_frames].mean() # Half of pre-lum dur 500ms
            edge_resp = roi.resp_traces[epoch][pre_frames:].min() # L1 responds negatively

            roi.edge_absolute_responses[epoch] = edge_resp
            roi.edge_step_responses[epoch] = edge_resp - pre_lum_mean

            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate=int_rate,int_time=epoch_infos[epoch]['total_dur_sec']) 
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate,int_time=int(round(len(roi.whole_traces[epoch])/roi.imaging_info['frame_rate']))) 

            int_resp_traces[idx, :] = roi.int_resp_trace[epoch]
        
        roi.edge_resp_traces_interpolated = int_resp_traces
        fp = roi.imaging_info['frame_rate']
        fp = np.tile(fp,len(used_epochs))

        aligned_traces = np.array(map(lambda trace,fp : np.roll(trace,len(trace)/2 - \
                                            int(np.argmin(trace[10:])+10)),
                     roi.edge_resp_traces_interpolated.copy(),fp))

        roi.max_aligned_traces = aligned_traces.copy()
        roi.concatenated_lum_traces = np.concatenate(aligned_traces)
        lums = [epoch_infos[epoch]['edge_lum'] for epoch in used_epochs]
        X =np.log10(np.array([ 16293.79680931,  31622.8436683 ,  62280.93738627, 123597.12482221,
       184913.31225815, 246229.49969409])) # Real luminance values
        Y = np.abs(roi.edge_step_responses.values())/np.max(np.abs(roi.edge_step_responses.values())) # L1 negative responses require absolute to quantify their amplitude
        X = X[np.argsort(lums)]
        roi.slope = linregress(X, np.transpose(Y))[0]

        # Sorted values
        luminances = roi.epoch_luminances.values()
        roi.sorted_luminances = np.sort(luminances)
        resps = np.array(roi.edge_step_responses.values())
        roi.sorted_edge_step_responses = resps[np.argsort(luminances)]

        resps_abs = np.array(roi.edge_absolute_responses.values())
        roi.sorted_edge_absolute_responses = resps_abs[np.argsort(luminances)]

        
        roi.max_resp = np.min(resps)
        
    return rois

def analyzeCenterGratwithBGcircle(rois,int_rate =10):
    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None
        
        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.power_at_sineFreq = {}
        roi.int_resp_trace = {}
        roi.mean_resp = {}
        roi.int_whole_trace = {}
        roi.conc_trace = []
        roi.epoch_contrasts = {}

        roi.BGcircle_luminance = {}
        roi.BGcircle_diameter = {}
        for epoch in used_epochs:
            curr_freq = epoch_infos[epoch]['velocity']/epoch_infos[epoch]['spatial_wavelength']
            
            roi.mean_resp[epoch] = roi.resp_traces[epoch].mean() - roi.whole_traces[epoch][int(roi.imaging_info['frame_rate']*2):].mean()
            roi.epoch_contrasts[epoch] = epoch_infos[epoch]['michelson_contrast']

            roi.BGcircle_luminance[epoch] = epoch_infos[epoch]['circle_luminance']
            roi.BGcircle_diameter[epoch] = epoch_infos[epoch]['circle_diam_deg']

            # Interpolate traces 
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate=int_rate,int_time=epoch_infos[epoch]['total_dur_sec']) 
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate=int_rate,int_time = round(len(roi.whole_traces[epoch])/roi.imaging_info['frame_rate'])) 
            # Fourier analysis of sinusodial responses

            curr_resp = roi.resp_traces[epoch]

            # Take the responses to sinusoidal but 700ms after stimuli to capture the whole dynamics of response
            # curr_resp = roi.whole_traces[epoch]
            # curr_resp = curr_resp[int(roi.imaging_info['frame_rate']*3.2):int(roi.imaging_info['frame_rate']*6)]

            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_resp-curr_resp.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=epoch_infos[epoch]['mean_luminance']))
            # plt.legend()
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs((xf[1:N//2])-curr_freq))
            roi.power_at_sineFreq[epoch] = power[req_idx]
            

        # Sorted values
        luminances = roi.BGcircle_luminance.values()
        diameters = roi.BGcircle_diameter.values()
        roi.sorted_luminances = np.sort(luminances)
        roi.sorted_diameters = np.sort(diameters)
        powers = np.array(roi.power_at_sineFreq.values())
        roi.diameters_sorted_power_at_sineFreq = powers[np.argsort(diameters)]
        roi.sorted_power_at_sineFreq = powers[np.argsort(luminances)]
        mean_resps = np.array(roi.mean_resp.values())
        roi.sorted_mean_resps = mean_resps[np.argsort(luminances)]
        roi.diameters_sorted_mean_resps = mean_resps[np.argsort(diameters)]
        
        roi.max_power = np.max(powers)
    
    return rois
def analyzeSineGratings(rois,int_rate = 10):
    """ Analysis for sinusoidal gratings """

    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None
        
        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.power_at_sineFreq = {}
        roi.int_resp_trace = {}
        roi.mean_resp = {}
        roi.int_whole_trace = {}
        roi.conc_trace = []
        roi.epoch_luminances = {}
        roi.epoch_contrasts = {}
        for epoch in used_epochs:
            curr_freq = epoch_infos[epoch]['velocity']/epoch_infos[epoch]['spatial_wavelength']
            curr_resp = roi.resp_traces[epoch]
            roi.mean_resp[epoch] = curr_resp.mean()
            roi.epoch_luminances[epoch] = epoch_infos[epoch]['mean_luminance']
            roi.epoch_contrasts[epoch] = epoch_infos[epoch]['michelson_contrast']
            # Interpolate traces 
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate=int_rate,int_time=epoch_infos[epoch]['total_dur_sec']) 
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate=int_rate,int_time = round(len(roi.whole_traces[epoch])/roi.imaging_info['frame_rate'])) 
            # Fourier analysis of sinusodial responses
            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_resp-curr_resp.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=epoch_infos[epoch]['mean_luminance']))
            # plt.legend()
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs((xf[1:N//2])-curr_freq))
            roi.power_at_sineFreq[epoch] = power[req_idx]
            
        X = [epoch_infos[epoch]['mean_luminance'] for epoch in used_epochs]
        Y = roi.power_at_sineFreq.values()
        Z = roi.mean_resp.values()
        
        roi.power_lum_slope = linregress(X, np.transpose(Y))[0]
        roi.base_slope = linregress(X, np.transpose(Z))[0]

        # Sorted values
        luminances = roi.epoch_luminances.values()
        contrasts = roi.epoch_contrasts.values()
        roi.sorted_luminances = np.sort(luminances)
        roi.sorted_contrasts = np.sort(contrasts)
        powers = np.array(roi.power_at_sineFreq.values())
        roi.contrast_sorted_power_at_sineFreq = powers[np.argsort(contrasts)]
        roi.sorted_power_at_sineFreq = powers[np.argsort(luminances)]
        mean_resps = np.array(roi.mean_resp.values())
        roi.sorted_mean_resps = mean_resps[np.argsort(luminances)]
        roi.contrast_sorted_mean_resps = mean_resps[np.argsort(contrasts)]
        
        roi.max_power = np.max(powers)
    return rois

def analyzeCenteredSineGratings(rois,int_rate = 10):
    """ Analysis for sinusoidal gratings with different luminances and contrasts """

    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None
    
        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.power_at_sineFreq = {}
        roi.int_resp_trace = {}
        roi.mean_resp = {}
        roi.int_whole_trace = {}
        roi.conc_trace = []
        roi.epoch_luminances = {}
        roi.epoch_contrasts = {}
        roi.epoch_temp_frequencies = {}
        roi.epoch_spatial_wavelengths = {}
        roi.epoch_diam = {}
        for epoch in used_epochs:
            curr_freq = epoch_infos[epoch]['velocity']/epoch_infos[epoch]['spatial_wavelength']
            curr_resp = roi.resp_traces[epoch]
            roi.mean_resp[epoch] = curr_resp.mean()
            roi.epoch_luminances[epoch] = epoch_infos[epoch]['mean_luminance']
            roi.epoch_contrasts[epoch] = epoch_infos[epoch]['michelson_contrast']
            roi.epoch_temp_frequencies[epoch] = curr_freq
            roi.epoch_spatial_wavelengths[epoch] = epoch_infos[epoch]['spatial_wavelength']
            roi.epoch_diam[epoch] = epoch_infos[epoch]['diameter_deg']
            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation
            # Fourier analysis of sinusodial responses

            # curr_resp = roi.int_resp_trace[epoch]
            period = 1.0 / roi.imaging_info['frame_rate']
            # period = 1.0 / 10 # FR is the interpolation rate            
            N = len(curr_resp)
            
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_resp-curr_resp.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=epoch_infos[epoch]['mean_luminance']))
            # plt.legend()
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs((xf[1:N//2])-curr_freq))
            roi.power_at_sineFreq[epoch] = power[req_idx]
            
    
    return rois

def analyzeSineGratings_generalized(rois,int_rate = 10):
    """ Analysis for sinusoidal gratings with different luminances and contrasts """

    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None
    
        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.power_at_sineFreq = {}
        roi.int_resp_trace = {}
        roi.mean_resp = {}
        roi.int_whole_trace = {}
        roi.conc_trace = []
        roi.epoch_luminances = {}
        roi.epoch_contrasts = {}
        roi.epoch_temp_frequencies = {}
        roi.epoch_spatial_wavelengths = {}
        for epoch in used_epochs:
            curr_freq = epoch_infos[epoch]['velocity']/epoch_infos[epoch]['spatial_wavelength']
            curr_resp = roi.resp_traces[epoch]
            roi.mean_resp[epoch] = curr_resp.mean()
            roi.epoch_luminances[epoch] = epoch_infos[epoch]['mean_luminance']
            roi.epoch_contrasts[epoch] = epoch_infos[epoch]['michelson_contrast']
            roi.epoch_temp_frequencies[epoch] = curr_freq
            roi.epoch_spatial_wavelengths[epoch] = epoch_infos[epoch]['spatial_wavelength']
            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation
            # Fourier analysis of sinusodial responses
            N = len(curr_resp)
            period = 1.0 / roi.imaging_info['frame_rate']
            x = np.linspace(0.0, N*period, N)
            yf = fft(curr_resp)
            xf = np.linspace(0.0, 1.0/(2.0*period), N//2)
            # mitigate spectral leakage
            w = blackman(N)
            ywf = fft((curr_resp-curr_resp.mean())*w)
            # plt.plot(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]),
            #               label = '{l}'.format(l=epoch_infos[epoch]['mean_luminance']))
            # plt.legend()
            power = 2.0/N * np.abs(ywf[1:N//2])
            req_idx = np.argmin(np.abs((xf[1:N//2])-curr_freq))
            roi.power_at_sineFreq[epoch] = power[req_idx]
            
    
    return rois

def fffAnalyze(rois, interpolation = True, int_rate = 10):
    """
    Concatanates and interpolates traces.
    
    """
    for roi in rois:
        stim_info = roi.stim_info
        epoch_infos = stim_info['meta']['epoch_infos']

        conc_trace = []
        stim_trace = []
        for epoch in epoch_infos:
            curr_stim = np.zeros((1,len(roi.resp_traces[epoch])))[0]
            curr_stim = curr_stim + epoch_infos[epoch]['lum']
            stim_trace=np.append(stim_trace,curr_stim,axis=0)
            conc_trace=np.append(conc_trace,roi.whole_traces[epoch],axis=0)
        
        roi.conc_trace = conc_trace
        roi.stim_trace = stim_trace
        
        # Calculating correlation
        curr_coeff, pval = pearsonr(roi.conc_trace,roi.stim_trace)
        roi.corr_fff = curr_coeff
        roi.corr_pval = pval
        if interpolation:
            roi.int_con_trace = interpolate_signal(conc_trace, 
                                                   roi.imaging_info['frame_rate'], 
                                                   int_rate,int(round(len(conc_trace)/roi.imaging_info['frame_rate']))) 
            roi.int_stim_trace = interpolate_signal(stim_trace, 
                                                    roi.imaging_info['frame_rate'], 
                                                   int_rate,int(round(len(conc_trace)/roi.imaging_info['frame_rate']))) 
            roi.int_rate = int_rate
            
    return rois

def analyzeAB_steps_time(rois, int_rate= 10):
    
    for roi in rois:
        stim_info = roi.stim_info
        fr = roi.imaging_info['frame_rate']
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None

        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.int_resp_trace = {}
        roi.int_whole_trace = {}

        roi.initial_luminances = {}
        roi.initial_dur = {}
        roi.weber_c = {}
        roi.a_step_response = {}
        roi.b_step_response = {}
        roi.b_absolute_response = {}
    

        for epoch in used_epochs:

            roi.initial_luminances[epoch] = epoch_infos[epoch]['initial_lum']
            roi.initial_dur[epoch] = epoch_infos[epoch]['initial_dur_sec']
            roi.weber_c[epoch] = epoch_infos[epoch]['weber_c']

            curr_resp = roi.resp_traces[epoch]

            
            base_end = int(np.floor(base_dur*fr))
            base_start  = int(np.floor(base_end - (3*fr)))
            base_resp = np.mean(roi.whole_traces[epoch][base_start:base_end])

            a_end = int(np.floor(epoch_infos[epoch]['initial_dur_sec']*fr))
            roi.a_step_response[epoch] = np.max(curr_resp[:a_end]) - base_resp

            ms_500 = int(np.floor(0.5*fr))
            roi.b_step_response[epoch] = np.max(curr_resp[a_end:]) - np.mean(curr_resp[a_end-ms_500:a_end])
            roi.b_absolute_response[epoch] = np.max(curr_resp[a_end:]) - base_resp

            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation

        initial_durs = roi.initial_dur.values()
        roi.sorted_a_durs = np.sort(initial_durs)
        b_steps = np.array(roi.b_step_response.values())
        roi.sorted_b_steps = b_steps[np.argsort(initial_durs)]
        
        
        roi.sorted_b_steps_a_added = np.empty((roi.sorted_b_steps.shape[0]+1,))
        roi.sorted_b_steps_a_added[:-1] = roi.sorted_b_steps.copy()
        roi.sorted_b_steps_a_added[-1] = (np.mean(roi.a_step_response.values()))
    
    return rois
def analyze_contrast_flashes(rois,int_rate=10):

    for roi in rois:
        stim_info = roi.stim_info
        fr = roi.imaging_info['frame_rate']
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None

        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.int_resp_trace = {}
        roi.int_whole_trace = {}

        roi.epoch_luminance = {}
        roi.weber_c = {}
        roi.step_response_upwards = {}
        roi.step_response_downwards = {}
    

        for epoch in used_epochs:

            roi.epoch_luminance[epoch] = epoch_infos[epoch]['lum']
            roi.weber_c[epoch] = (epoch_infos[epoch]['lum']-epoch_infos[baseline_epoch]['lum'])/epoch_infos[baseline_epoch]['lum']

            curr_resp = roi.resp_traces[epoch]
            base_end = int(np.floor(base_dur*fr))
            base_start  = int(np.floor(base_end - (1*fr))) # last second of baseline
            base_resp = np.mean(roi.whole_traces[epoch][base_start:base_end])

            roi.step_response_upwards[epoch] = np.max(curr_resp) - base_resp
            roi.step_response_downwards[epoch] = np.abs(np.min(curr_resp) - base_resp)

            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate,epoch_infos[epoch]['total_dur_sec']) #TODO: time for interpolation
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate,epoch_infos[epoch]['total_dur_sec']+2*base_dur) #TODO: time for interpolation

    
    return rois
    
def sigmoid_c(C,k,a):
  """
  Sigmoid contrast function. Taken from Shuai's L2 model.

  Args:
    C (float): x
    k (float): determines the slope of sigmoid
    a (float): determines the scaling of sigmoid

  Returns:
    float: response F(x) for input x
  """

  # Define the sigmoidal transfer function f = F(x)
  f = a* ((1 + np.exp(k*C))**-1 - 1/2)

  return f

def twoDgaussian(height, center_x, center_y, width_x, width_y):
    """Returns a 2D gaussian function with the given parameters"""
    
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def moments_v2(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments. 

    Modified BG: x and y are now the argmax of the RF, while estimating the width negative values 
    are set to 0 in order to estimate it more robustly.
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    max_coords = np.array(np.where(np.abs(data)==np.abs(data).max())).astype(float)
    x = float(max_coords[0])
    y = float(max_coords[1])
    col = data[:, int(y)]
    col[col<0]=0 # get rid of the negative values
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    row[row<0] = 0 # get rid of the negative values
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitTwoDgaussian(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit. 
    
    Data needs to have a positive peak
    """
    params = moments_v2(data)
    
    errorfunction = lambda p: np.ravel(twoDgaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    fit = twoDgaussian(*p)(*np.indices(data.shape))

    residuals = data- fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data-np.mean(data))**2)
    r_squared = 1 - (ss_res / ss_tot)


    return p, success, fit ,r_squared


def plot2DGaussian(fit, data):
    
    plt.close('all')
    fig = plt.figure()
    plt.imshow(data)
    # plt.imshow(fit, alpha=.5)
    plt.contour(fit,alpha=.5)
    plt.show()
    return fig

def plotMasksROInums(rois,back_image,save_fig = False, save_dir = None,save_id =None):
    """ Makes an image of ROI masks and their corresponding numbers to facilitate
    ROI selection for the same layers so that ROI numbers match and data can be later on
    easily matched """
   
    plt.close('all')
    plt.style.use("dark_background")

    plt.imshow(back_image,cmap='gray')
    for roi in rois:
        curr_mask = np.array(roi.mask.copy()).astype(float)
        curr_mask[curr_mask==0] = np.nan

        x = np.where(roi.mask)[0][0]
        y = np.where(roi.mask)[1][0]
        plt.text(y,x,str(roi.number_id))
        plt.imshow(curr_mask,cmap='Accent', alpha=0.15)
    ax1=plt.gca()
    ax1.axis('off')
    ax1.set_title('ROIs n=%d' % len(rois))

    if save_fig:
        # Saving figure
        save_name = 'ROIs_num_%s' % (save_id)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('ROI images saved')
    plt.close('all')
    plt.imshow(back_image,cmap='gray')
    ax2=plt.gca()
    ax2.axis('off')
    ax2.set_title('Mean image')

    if save_fig:
        # Saving figure
        save_name = 'MeanIm_%s' % (save_id)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('Mean image saved')

def analyzeConsecutiveFlashes(rois, int_rate=10):
    for roi in rois:
        stim_info = roi.stim_info
        fr = roi.imaging_info['frame_rate']
        epoch_infos = stim_info['meta']['epoch_infos']
        randomization_c = stim_info['meta']['randomization_condition']

        if (randomization_c == 1) or (randomization_c == 3):
            baseline_epoch = 'epoch_1'
            used_epochs = epoch_infos.keys()
            used_epochs.remove(baseline_epoch)
            base_dur = epoch_infos[baseline_epoch]['total_dur_sec']
        else:
            baseline_epoch = None

        # Initialize ROI variables
        roi.int_rate = int_rate
        roi.int_resp_trace = {}
        roi.int_whole_trace = {}

        roi.initial_luminances = {}
        roi.initial_dur = {}
        roi.weber_c = {}
        roi.a_step_response = {}
        roi.b_step_response = {}
        roi.b_absolute_response = {}
    

        for epoch in used_epochs:

            roi.initial_luminances[epoch] = epoch_infos[epoch]['initial_lum']
            roi.initial_dur[epoch] = epoch_infos[epoch]['initial_dur_sec']
            roi.weber_c[epoch] = epoch_infos[epoch]['weber_c']

            curr_resp = roi.resp_traces[epoch]

            
            base_end = int(np.floor(base_dur*fr))
            base_start  = int(np.floor(base_end - (0.5*fr))) # 500ms before the end of baseline
            base_resp = np.mean(roi.whole_traces[epoch][base_start:base_end])

            a_end = int(np.floor(epoch_infos[epoch]['initial_dur_sec']*fr))
            roi.a_step_response[epoch] = np.max(curr_resp[:a_end]) - base_resp

            ms_500 = int(np.floor(0.5*fr))
            roi.b_step_response[epoch] = np.max(curr_resp[a_end:]) - np.mean(curr_resp[a_end-ms_500:a_end])
            roi.b_absolute_response[epoch] = np.max(curr_resp[a_end:]) - base_resp

            # Interpolate traces
            roi.int_resp_trace[epoch] = interpolate_signal(roi.resp_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation
            roi.int_whole_trace[epoch] = interpolate_signal(roi.whole_traces[epoch],roi.imaging_info['frame_rate'],int_rate) #TODO: time for interpolation

        initial_lums = roi.initial_luminances.values()
        roi.sorted_a_lums = np.sort(initial_lums)
        b_steps = np.array(roi.b_step_response.values())
        roi.sorted_b_steps = b_steps[np.argsort(initial_lums)]

        abs_b_steps = np.array(roi.b_absolute_response.values())
        roi.sorted_absolute_b_steps = abs_b_steps[np.argsort(initial_lums)]
    
    return rois

def mapRF_8D_stripes(rois,screen_props = {'45':113, '135':113,
                                                   '225':113,'315':113,
                                                   '0':80,'180':80,
                                                   '90':80,'270':80}):
    """
    Maps the receptive field with a method based on the Fiorani et al 2014 paper:
    "Automatic mapping of visual cortex receptive fields: A fast and precise algorithm"
    
    # Delay correction not good as currently implemented here. Non DS neurons 
    # do not need delay correction
        
    """
    from scipy.ndimage.interpolation import rotate

    for i, roi in enumerate(rois):
        
        
                
        dim = int(round(np.max(screen_props.values())*np.sqrt(2)))
        all_RFs = []
        
        for epoch in roi.stim_info['meta']['epoch_infos'].keys():
            if roi.stim_info['meta']['epoch_infos'][epoch]['stim_type'] != "movingStripe-v1":
                continue
            stripe_speed = roi.stim_info['meta']['epoch_infos'][epoch]['velocity']
            curr_direction = roi.stim_info['meta']['epoch_infos'][epoch]['direction_deg']
            try:
                degrees_covered = screen_props[str(int(curr_direction))]
                frames_needed = int(np.around((degrees_covered/float(stripe_speed))\
                    * roi.imaging_info['frame_rate'],0))
            except KeyError:
                raise KeyError('Stripe direction not found: %s degs' % str(int(curr_direction)))
            
            
            curr_RF = np.full((int(dim), int(dim)),np.nan)
            b, a = signal.butter(3, 0.2, 'low')
            whole_t = signal.filtfilt(b, a,roi.whole_traces[epoch])
            whole_t = whole_t -np.min(whole_t)
            resp_len = len(roi.resp_traces[epoch])
            base_len = (len(whole_t)-resp_len)/2
            base_t = whole_t[:base_len]
            base_activity = np.mean(base_t)
            
            raw_trace =whole_t[base_len:base_len+resp_len]
                
                
            # Standardize responses so that DS responses dominate less
            sd = np.sqrt(np.sum(np.square(raw_trace))/(len(raw_trace)-1))
            normalized = (raw_trace - base_activity)/sd
            resp_trace = normalized
        
            # Need to map to the screen
            diagonal_dir = str(int(np.mod(curr_direction+90,360)))
            
            degree_needed = degrees_covered
            diag_dir_covered = screen_props[diagonal_dir]
                
                
            screen_coords = np.linspace(0, degree_needed, 
                                        num=degree_needed, endpoint=True)
            roi_t_v = np.linspace(0, degree_needed, 
                                    num=len(resp_trace), endpoint=True)
            i_resp = np.interp(screen_coords, roi_t_v, resp_trace)
            diagonal_dir = str(int(np.mod(curr_direction+90,360)))
            back_projected = np.tile(i_resp, (int(diag_dir_covered),1))
            back_projected[np.isnan(back_projected)] = 0
            # 90 degrees are rightwards w.r. to the fly so it shouldn't be turned
            # 0 degrees is upwards so 90-curr_dir
            # 
            rotated = rotate(back_projected+1, 
                                angle=np.mod(90-curr_direction,360))
            rotated[rotated==0] = np.nan
            rotated = rotated-1
            idx1_1 = int((dim-rotated.shape[0])/2)
            idx1_2 = int((dim-rotated.shape[0])/2+rotated.shape[0])
            
            idx2_1 = int((dim-rotated.shape[1])/2)
            idx2_2 = int((dim-rotated.shape[1])/2+rotated.shape[1])
            
            
            curr_RF[idx1_1 : idx1_2,idx2_1 : idx2_2] = rotated
            all_RFs.append(curr_RF)
                
                
        roi.RF_maps = all_RFs
        roi.RF_map = np.mean(roi.RF_maps, axis=0)
        roi.RF_map_norm = (roi.RF_map - np.nanmin(roi.RF_map)) / \
                             (np.nanmax(roi.RF_map) - np.nanmin(roi.RF_map))
        roi.RF_center_coords = np.argwhere(roi.RF_map==np.nanmax(roi.RF_map))[0]
        
    return rois

def findMovingStripeDur(epoch_info,proj_params):
        if (epoch_info['stim_type'] == 'edges-v1') or (epoch_info['stim_type'] == 'movingStripe-v1'):
            diag = np.sqrt(proj_params['sizeY']**2+proj_params['sizeX']**2)
            angle_y = np.degrees(np.arcsin(proj_params['sizeY'] / diag))
            # Calculates the distance that the edge or the stripe needs to travel to cover the whole screen
            if epoch_info['direction_deg'] <= 90:
                distance_to_travel = diag * np.sin(np.deg2rad(angle_y+epoch_info['direction_deg']))
                distance_to_travel = np.abs(distance_to_travel)
            elif epoch_info['direction_deg'] <= 180:
                distance_to_travel = diag * np.cos(np.deg2rad(90-angle_y+epoch_info['direction_deg']))
                distance_to_travel = np.abs(distance_to_travel)
            elif epoch_info['direction_deg'] <= 270:
                distance_to_travel = diag * np.sin(np.deg2rad(angle_y+epoch_info['direction_deg']))
                distance_to_travel = np.abs(distance_to_travel)
            elif epoch_info['direction_deg'] <= 360:
                distance_to_travel = diag * np.cos(np.deg2rad(90-angle_y+epoch_info['direction_deg']))
                distance_to_travel = np.abs(distance_to_travel)
            else:
                raise NameError("Direction has to be in degrees")
            
        else:   
            raise NameError("Stim type not appropriate")
        
        distance_to_travel += epoch_info['width'] # so that the stripe disappears from the screen
        epoch_dur = distance_to_travel/epoch_info['velocity'] 

        return epoch_dur, distance_to_travel

def plot_RFs(rois, number=None, f_w =None,cmap='inferno',
             center_plot = False, center_val = 0.95):
    import random
    plt.close('all')
    colors = pac.run_matplotlib_params()
    if (number == None) or (f_w==None):
        f_w = 5
        if len(rois)>10:
            number=10
        else:
            number = len(rois)
    elif number > len(rois):
        number = len(rois)
    f_w = f_w*2
    # Randomize ROIs
    copy_rois = copy.deepcopy(rois)
    # random.shuffle(copy_rois)
        
        
    
    if number <= f_w/2:
        dim1= number
        dim2 = 1
    elif number/float(f_w/2) > 1.0:
        dim1 = f_w/2
        dim2 = int(np.ceil(number/float(f_w/2)))
    fig1, ax1 = plt.subplots(ncols=dim1, nrows=dim2, figsize=(dim1, dim2))
    ax = ax1.flatten()
    for idx, roi in enumerate(rois):
        if idx == number:
            break
        
        center_RF = copy.deepcopy(roi.RF_map_norm)
        center_RF[center_RF<center_val] =np.nan
        ax[idx].imshow(roi.RF_map_norm, cmap=cmap)
        if center_plot:
            ax[idx].imshow(center_RF,alpha=.5,
                        cmap='Greens')
        ax[idx].axis('off')
        ax[idx].set_xlim(((np.shape(roi.RF_map_norm)[1]-80)/2,(np.shape(roi.RF_map_norm)[0]-80)/2+80))
        ax[idx].set_ylim(((np.shape(roi.RF_map_norm)[0]-80)/2+80,(np.shape(roi.RF_map_norm)[0]-80)/2))
        try:
            ax[idx].set_title('PD: {pd}'.format(pd=int(roi.PD)),fontsize='xx-small')
        except AttributeError:
            a=0
    try:
        for ax_id in range(len(ax)-idx-1):
            ax[ax_id+idx].axis('off')
    except:
        a =1
    return fig1