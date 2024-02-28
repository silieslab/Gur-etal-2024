import os
from skimage import io
import glob
import pickle
import re
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
from matplotlib import pyplot as plt
from roipoly import RoiPoly

import PyROI
from xmlUtilities import getFramePeriod, getLayerPosition, getPixelSize, getMicRelativeTime
import summary_figures as sf

def preProcessMovie(data_path):
    

    # Generate necessary directories for figures
    current_t_series=os.path.basename(data_path)
    
    # Load movie, get stimulus and imaging information
    try:
        movie_path = os.path.join(data_path, 'motCorr.sima',
                                  '{t_name}_motCorr.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
    except IOError:
        movie_path = os.path.join(data_path, '{t_name}_motCorr.tif'.format(t_name=current_t_series))
        time_series = io.imread(movie_path)
    
    frames_imaged = time_series.shape[0]
    ## Get stimulus and xml information
    imaging_info = processXmlInfo(data_path)
    stim_info = processPyStimInfo(data_path,frames_imaged)
    
    
    return time_series, stim_info, imaging_info

def processXmlInfo(data_path):
    """
        Extracts the stimulus and imaging parameters. 
    """
    # Finding the xml file and retrieving relevant information
    xmlPath = os.path.join(data_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    imagetimes = getMicRelativeTime(xmlFile)
    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    frameRate = 1/framePeriod
    layerPosition = getLayerPosition(xmlFile=xmlFile)
    depth = layerPosition[2]
    
    # Pixel definitions
    x_size, y_size, pixelArea = getPixelSize(xmlFile)

    imaging_info = {'frame_rate' : frameRate, 'pixel_size': x_size, 
                             'depth' : depth, 'frame_timings':imagetimes}
    
    return imaging_info
    
def processPyStimInfo(data_path, frames_imaged):

    # Stimulus information
    paths = os.path.join(data_path, '*.pickle')
    stim_file_path = (glob.glob(paths))[0]

    load_var = open(stim_file_path, 'rb')
    stim_info = pickle.load(load_var)

    stim_info['processed'] = {}

    # Epoch coordinates
    (stim_info['processed']['trial_coordinates'], \
        stim_info['processed']['epoch_trace_frames']) = \
        getEpochTrialCoords(stim_info,frames_imaged)

    return stim_info

def getEpochTrialCoords(stim_info, frames_imaged):
    """
        Finds the epoch changing coordinates.

        Returns
        =======
        epoch coordinates : dict
            Each key is the epoch name. 
            Trial coordinates are given as frames: [[baseStart, trialStart, trialEnd]]
            Note: Frames are mapped onto 0 index (1st frame is -> 0) due to Python indexing. Movie will be mapped like this too.


    """

    epoch_coordinates = {}
    epoch_trace = np.array(stim_info['stimulus_epoch'])
    frame_trace = np.array(stim_info['imaging_frame'])-1 # Map frames to 0 beginning index

    frame_changes = np.where(np.insert(np.diff(frame_trace),0,1))[0]

    epoch_trace_frames = np.ones([frames_imaged])
    frame_trace_frames = range(frames_imaged)

    # Find the epochs corresponding to the imaged frames
    # If a frame contains more than 1 epoch than assign it to the longest appearing epoch
    for iFrame in range(frames_imaged):
        frame_mask = frame_trace == iFrame
        if frame_mask.any(): 
            epoch_trace_frames[iFrame] = np.bincount(epoch_trace[frame_mask].astype(int)).argmax()
        else: # Sometimes the first frame is skipped in the stim output (or some other frames due to delays)
            # Then we append the epoch of the frame with the nearest number
            frame_mask = frame_trace == frame_trace[np.argmin(np.abs(frame_trace-iFrame))]
            epoch_trace_frames[iFrame] = np.bincount(epoch_trace[frame_mask].astype(int)).argmax()


    epoch_trace_frames = epoch_trace_frames[:frames_imaged]

    # In case of randomization is 1 or 3, then we will have coordinates with
    # [[baseStart, trialStart, trialEnd, baseEnd]]
    randomization = stim_info['meta']['randomization_condition']
    if (randomization == 1) or (randomization == 3):
        base_epoch_n = 1
        base_coords_beg = np.where(np.diff(np.array(epoch_trace_frames==base_epoch_n).astype(int))==1)[0] + 1
        base_coords_beg = np.insert(base_coords_beg,0,0) # Add the start

        base_coords_end = np.where(np.diff(np.array(epoch_trace_frames==base_epoch_n).astype(int))==-1)[0]
        base_reps = np.min([len(base_coords_beg),len(base_coords_end)])
        base_coords_beg = base_coords_beg[:base_reps]
        base_coords_end = base_coords_end[:base_reps]
        base_length = np.min(base_coords_end - base_coords_beg) + 1
    else:
        base_epoch_n = None
        base_length = 0
    
    for curr_epoch in stim_info['meta']['epoch_infos'].keys():
        curr_epoch_n = int(re.search(r'\d+', curr_epoch).group())
        epoch_coordinates[curr_epoch] = []
        # Don't take the baseline epoch
        if curr_epoch_n == base_epoch_n:
            for baseTrial in range(base_reps):
                epoch_coordinates[curr_epoch].append([frame_trace_frames[base_coords_beg[baseTrial]],
                     frame_trace_frames[base_coords_end[baseTrial]]])
            continue

        # Find the trial start and ends
        epoch_beg = np.where(np.diff(np.array(epoch_trace_frames==curr_epoch_n).astype(int))==1)[0] + 1

        # If this is the first presented epoch then add a 0 to the epoch_beg
        # This condition will only work when the first epoch is not a baseline epoch 
        # meaning that randomization is not 1 or 3
        if epoch_trace_frames[0] == curr_epoch_n:
            epoch_beg = np.insert(epoch_beg,0,0) # Add the start
        epoch_end = np.where(np.diff(np.array(epoch_trace_frames==curr_epoch_n).astype(int))==-1)[0]
        completed_trial_n = np.min([len(epoch_beg),len(epoch_end)]) # Don't use if the trial ended prematurely
        
        
        # Arrange it in a list
        for iTrial in range(completed_trial_n):  
            base_beg = frame_trace_frames[epoch_beg[iTrial]-base_length]
            
            trial_beg = frame_trace_frames[epoch_beg[iTrial]]
            trial_end = frame_trace_frames[epoch_end[iTrial]]      
            
            # Don't use the trial if it is not followed by a full baseline
            if epoch_end[iTrial]+base_length >= len(frame_trace_frames):
                continue
            else:
                base_end = frame_trace_frames[epoch_end[iTrial]+base_length]

            epoch_coordinates[curr_epoch].append([base_beg, trial_beg,trial_end,base_end])
    
    return epoch_coordinates, epoch_trace_frames
            
def organizeExtractionParams(extraction_type,
                               current_t_series=None,current_exp_ID=None,
                               alignedDataDir=None,
                               stimInputDir=None,
                               use_other_series_roiExtraction = None,
                               use_avg_data_for_roi_extract = None,
                               roiExtraction_tseries=None,
                               transfer_data_n = None,
                               transfer_data_store_dir = None,
                               transfer_type = None,
                               imaging_info=None,
                               experiment_conditions=None):
    
    extraction_params = {}
    extraction_params['type'] = extraction_type
    if extraction_type == 'SIMA-STICA':
        if use_other_series_roiExtraction:
            series_used = roiExtraction_tseries
        else:
            series_used = current_t_series
        extraction_params['series_used'] = series_used
        extraction_params['series_path'] = \
            os.path.join(alignedDataDir, current_exp_ID, 
                                  series_used)
        extraction_params['area_max_micron'] = 4
        extraction_params['area_min_micron'] = 1
        extraction_params['cluster_max_1d_size_micron'] = 4
        extraction_params['cluster_min_1d_size_micron'] = 1
        extraction_params['extraction_reliability_threshold'] = 0.4
        extraction_params['use_trial_avg_video'] = \
            use_avg_data_for_roi_extract
    elif extraction_type == 'transfer':
        transfer_data_path = os.path.join(transfer_data_store_dir,
                                          transfer_data_n)
        extraction_params['transfer_data_path'] = transfer_data_path
        extraction_params['transfer_type']=transfer_type
        extraction_params['imaging_information']= imaging_info
        extraction_params['experiment_conditions'] = experiment_conditions
        
        
    return extraction_params

def selectManualROIs(image_to_select_from, image_cmap ="gray",pause_t=7,
                   ask_name=True):
    """ Enables user to select rois from a given image using roipoly module.

    Parameters
    ==========
    image_to_select_from : numpy.ndarray
        An image to select ROIs from
    
    Returns
    =======
    
    """
    import warnings 
    plt.close('all')
    stopsignal = 0
    roi_number = 0
    roi_masks = []
    mask_names = []
    
    im_xDim = np.shape(image_to_select_from)[0]
    im_yDim = np.shape(image_to_select_from)[1]
    mask_agg = np.zeros(shape=(im_xDim,im_yDim))
    iROI = 0
    plt.style.use("dark_background")
    while (stopsignal==0):

        
        # Show the image
        fig = plt.figure()
        plt.imshow(image_to_select_from, interpolation='nearest', cmap=image_cmap)
        plt.colorbar()
        curr_agg = mask_agg.copy()
        curr_agg[curr_agg==0] = np.nan
        plt.imshow(curr_agg, alpha=0.3,cmap = 'Accent')
        plt.title("Select ROI: ROI%d" % roi_number)
        plt.show(block=False)
       
        
        # Draw ROI
        curr_roi = RoiPoly(color='r', fig=fig)
        iROI = iROI + 1
        if ask_name:
            mask_name = raw_input("\nEnter the ROI name:\n>> ")
        else:
            mask_name = iROI
        curr_mask = curr_roi.get_mask(image_to_select_from)
        if len(np.where(curr_mask)[0]) ==0 :
            warnings.warn('ROI empty.. discarded.') 
            continue
        mask_names.append(mask_name)
        
        
        roi_masks.append(curr_mask)
        
        mask_agg[curr_mask] += 1
        
        
        
        roi_number += 1
        signal = raw_input("\nPress k for exiting program, otherwise press enter")
        if (signal == 'k'):
            stopsignal = 1
        
    
    return roi_masks, mask_names

def generateROIsImage(roi_masks,im_shape):
    # Generating an image with all clusters
    all_rois_image = np.zeros(shape=im_shape)
    all_rois_image[:] = np.nan
    for index, roi in enumerate(roi_masks):
        curr_mask = roi
        all_rois_image[curr_mask] = index + 1
    return all_rois_image
    
def transferROIs(transfer_data_path, transfer_type,experiment_info=None,
                     imaging_info=None):
    '''
    
    Updates:
        25/03/2020 - Removed transfer types of 11 steps and AB steps since they
        are redundant with minimal type
    '''
    load_path = open(transfer_data_path, 'rb')
    workspace = cPickle.load(load_path)
    rois = workspace['final_rois']
    
    if transfer_type == 'luminance_gratings' or \
        transfer_type == 'lum_con_gratings' :
        
        properties = ['CSI', 'CS','PD','DSI','category',
                      'analysis_params']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'stripes_OFF_delay_profile':
        
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='OFF')
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
        
    elif transfer_type == 'stripes_ON_delay_profile':
        properties = ['CSI', 'CS','PD','DSI','two_d_edge_profile','category',
                      'analysis_params','edge_start_loc','edge_speed']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS='ON')

        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))

    elif ((transfer_type == 'stripes_ON_vertRF_transfer') or \
          (transfer_type == 'stripes_ON_horRF_transfer') or \
          (transfer_type == 'stripes_OFF_vertRF_transfer') or \
          (transfer_type == 'stripes_OFF_horRF_transfer')):
        properties = ['corr_fff', 'max_response','category','analysis_params']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'ternaryWN_elavation_RF'):
        properties = ['corr_fff', 'max_response','category','analysis_params',
                      'reliability','SNR']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'gratings_transfer_rois_save'):
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        
    elif (transfer_type == 'luminance_edges_OFF' ):
        if (('R64G09' in rois[0].experiment_info['Genotype']) or \
         ('T5' in rois[0].experiment_info['Genotype'])):
            CS = 'OFF'
            warnings.warn('Transferring only T5 neurons')
        else:
            CS = None
            warnings.warn('NO CS selected since genotype is not found to be T4-5')
        
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    elif (transfer_type == 'luminance_edges_ON'):
        
        if (('R64G09' in rois[0].experiment_info['Genotype']) or \
         ('T4' in rois[0].experiment_info['Genotype'])):
            CS = 'ON'
            warnings.warn('Transferring only T4 neurons')
        else:
            CS = None
            warnings.warn('NO CS selected since genotype is not found to be T4-5')
        properties = ['CSI', 'CS','PD','DSI','category','RF_maps','RF_map',
                      'RF_center_coords','analysis_params','RF_map_norm']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info,CS=CS)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
   
    elif transfer_type == 'STF_1':
        properties = ['CSI', 'CS','PD','DSI','category','analysis_params']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
            
    elif transfer_type == 'minimal' :
        print('Transfer type is minimal... Transferring just masks, categories and if present RF maps...\n')
        properties = ['category','analysis_params','RF_maps','RF_map',
                      'RF_center_coords','RF_map_norm']
        transferred_rois = PyROI.transfer_masks(rois, properties,
                                          experiment_info = experiment_info, 
                                          imaging_info =imaging_info)
        
        print('{tra_n}/{all_n} ROIs transferred and analyzed'.format(all_n = \
                                                                     int(len(rois)),
                                                                     tra_n= int(len(transferred_rois))))
    else:
        raise NameError('Invalid ROI transfer type')
        
        
   
    return transferred_rois

def selectROIs(extraction_params, image_to_select=None):
    """

    """
    # Categories can be used to classify ROIs depending on their location
    # Backgroud mask (named "bg") will be used for background subtraction
    plt.close('all')
    plt.style.use("default")
    print('\n\nSelect categories and background')
    [cat_masks, cat_names] = selectManualROIs(image_to_select, 
                                            image_cmap="gray",
                                            pause_t=8)
    
    # have to do different actions depending on the extraction type
    if extraction_params['type'] == 'manual':
        print('\n\nSelect ROIs')
        [roi_masks, roi_names] = selectManualROIs(image_to_select, 
                                                image_cmap="gray",
                                                pause_t=4.5,
                                                ask_name=False)
        all_rois_image = generateROIsImage(roi_masks,
                                                  np.shape(image_to_select))
        
        return cat_masks, cat_names, roi_masks, all_rois_image, None, None
            
    elif extraction_params['type'] == 'SIMA-STICA': 
        # Could be copied from process_mov_core -> run_ROI_selection()
        raise NameError("ROI extraction for SIMA-STICA is not yet impletemented.")
    
    elif extraction_params['type'] == 'transfer':
        
        rois = transferROIs(extraction_params['transfer_data_path'],
                                extraction_params['transfer_type'],
                                experiment_info=extraction_params['experiment_conditions'],
                                imaging_info=extraction_params['imaging_information'])
        
        return cat_masks, cat_names, None, None, rois, None
    
    else:
       raise TypeError('ROI selection type not understood.') 

def analyzeTraces(rois, analysis_params,save_fig=True,fig_save_dir=None,summary_save_d=None):
    """ Each different stimulus type has its own analysis way. 
        This function will implement them accordingly.
    """
    plt.style.use('default')
    analysis_type = analysis_params['analysis_type']

    figtitle = 'Summary: %s Gen: %s | Age: %s' % \
           (rois[0].experiment_info['MovieID'].split('-')[0],
            rois[0].experiment_info['Genotype'], rois[0].experiment_info['Age'])

    if analysis_type == 'centered_uniform_lum':
        rois = PyROI.analyzeConsecutiveFlashes(rois,int_rate = 10)
        stim_info = rois[0].stim_info

        b_steps = []
        for roi in rois:
            b_steps.append(roi.sorted_b_steps)
            plt.scatter(roi.sorted_a_lums,roi.sorted_b_steps)

        mean_r = np.mean(b_steps,axis=0)
        plt.plot(roi.sorted_a_lums,mean_r,color='k')
        std_r = np.std(b_steps,axis=0)
        ub = mean_r + std_r
        lb = mean_r - std_r
        plt.fill_between(roi.sorted_a_lums, ub, lb, color='k',alpha=.3)

        plt.xlabel('1st flash luminance')
        plt.ylabel('2nd flash step response')
        plt.title('Consecutive flashes %d degrees'%(int(stim_info['meta']['epoch_infos']['epoch_2']['diameter_deg'])))

        fig = plt.gcf()
        f1_n = 'consFlashes_center_summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
    
    elif analysis_type == '8D-stripes-RFmap':
        rois = PyROI.mapRF_8D_stripes(rois,screen_props = {'45':113, '135':113,
                                                   '225':113,'315':113,
                                                   '0':80,'180':80,
                                                   '90':80,'270':80})
    
        # random.shuffle(rois_plot)
        
        fig1 = PyROI.plot_RFs(rois, number=len(rois), f_w =5,cmap='coolwarm',
                                center_plot = True, center_val = 0.95)
        
        if save_fig:
           # Saving figure 
           f1_n = 'RF_examples' 
           os.chdir(fig_save_dir)
           fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)


    
    elif analysis_type == 'luminance_gratings' or analysis_type == 'centered_gratings_5lum_1stExp':
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']) and \
            (not('1D' in rois[0].stim_name)):
            map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
            
        rois = PyROI.analyzeSineGratings(rois)
        luminances = rois[0].epoch_luminances.values()
        powers = np.array(map(lambda roi: roi.power_at_sineFreq.values(),rois)).T
        plt.scatter(np.tile(luminances,(powers.shape[1],1)).T,powers,color='k',alpha=0.7)
        plt.plot(np.sort(luminances),powers.mean(axis=1)[np.argsort(luminances)],
            '--k',linewidth=3)

        plt.xlabel('Luminance')
        plt.ylabel('Signal strength')
        fig = plt.gcf()
        f0_n = 'Summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False)

        if analysis_type == 'centered_gratings_5lum_1stExp':
            for roi in rois: # normally there should be a single ROI so a loop is not necessary (TODO: find a way with more clarity)
                roi.gratings_roi_id = '{flyID}-{roi_id}'.format(flyID=roi.experiment_info['FlyID'],
                                                                roi_id=analysis_params['center_grating_roiID'])
    
    elif analysis_type == 'centered_gratings_lum_size':

        rois = PyROI.analyzeCenteredSineGratings(rois)

        roi = rois[0]
        luminances = np.array(roi.epoch_luminances.values())
        diameters = np.array(roi.epoch_diam.values())
        powers = np.array(roi.power_at_sineFreq.values())

        for diam in np.unique(diameters):
            curr_lums = luminances[diam==diameters]
            curr_pows = powers[diam==diameters]

            sorted_pow = curr_pows[np.argsort(curr_lums)]

            plt.plot(np.sort(curr_lums),sorted_pow,color=[diam/diameters.max(), 0, 0],label='d:{s}'.format(s=str(diam)))
            
        
        plt.legend()
        plt.xlabel('luminance')
        plt.ylabel('Power at 1 Hz')

        fig1 = plt.gcf()
        f1_n = 'DataSummary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
                    transparent=False)
        
        dim1 = 15
        dim2 = 15
        fig2, axs = plt.subplots(nrows=len(np.unique(diameters)), ncols=len(np.unique(luminances)), figsize=(dim1, dim2),sharey=True,sharex=True)

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.8, hspace=0.2)

        uniq_lums = np.unique(luminances)

        uniq_diams = np.unique(diameters)
        epochs = np.array(roi.int_whole_trace.keys())
        
        t_axis = np.linspace(0,12,len(roi.int_whole_trace['epoch_2']))

        
        for i_diam,diam in enumerate(uniq_diams):
            for i_lum, lum in enumerate(uniq_lums):
            
                curr_epoch = epochs[(lum == luminances) & (diameters == diam)]
                axs[i_diam, i_lum].plot(t_axis,roi.int_whole_trace[curr_epoch[0]])
                
                axs[i_diam, i_lum].set_title('l:{s}\nd:{d}'.format(s=lum,d =diam))
                
                axs[i_diam, i_lum].axvline(x=4, color=[1,0,0,0.4], linestyle='-')
                axs[i_diam, i_lum].axvline(x=8, color=[1,0,0,0.4], linestyle='-')

                if i_diam == len(uniq_diams):
                    axs[i_diam, i_lum].set_xlabel('Time (s)')
        axs[i_diam, i_lum].set_xlim([0,12])

        f2_n = 'Traces_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig2.savefig('%s.pdf'% f2_n, bbox_inches='tight',
                    transparent=False)
        
        
    
    elif analysis_type == 'centered_gratings_expanding_bgCircle':
        
        rois = PyROI.analyzeCenterGratwithBGcircle(rois)

        roi = rois[0]
        luminances = np.array(roi.BGcircle_luminance.values())
        diameters = np.array(roi.BGcircle_diameter.values())
        powers = np.array(roi.power_at_sineFreq.values())

        for luminance in np.unique(luminances):
            if luminance == 0.25: #for power at mean
                continue
            curr_diams = diameters[luminance==luminances]
            curr_pows = powers[luminance==luminances]

            sorted_pow = curr_pows[np.argsort(curr_diams)]

            plt.plot(np.sort(curr_diams),sorted_pow,color=[2*luminance, 0, 0],label='lum:{s}'.format(s=str(luminance)))
            
        
        plt.axhline(y=powers[0.25==luminances], color='k', linestyle='--', label='lum:0.25')
        plt.legend()
        plt.xlabel('BG circle size')
        plt.ylabel('Power at 1 Hz')

        fig1 = plt.gcf()
        f1_n = 'DataSummary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
                    transparent=False)
        
        dim1 = 15
        dim2 = 10
        fig2, axs = plt.subplots(nrows=len(np.unique(luminances)), ncols=len(np.unique(diameters))-1, figsize=(dim1, dim2),sharey=True,sharex=True)

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.8, hspace=0.2)

        uniq_lums = np.unique(luminances)
        mask = uniq_lums != 0.25  
        uniq_lums = uniq_lums[mask] 

        uniq_diams = np.unique(diameters)
        uniq_diams = uniq_diams[1:] #get rid of 5deg
        epochs = np.array(roi.int_whole_trace.keys())
        
        f_epoch = epochs[(diameters == 5)]
        t_axis = np.linspace(0,9,len(roi.int_whole_trace[f_epoch[0]]))

        axs[0, 0].plot(t_axis,roi.int_whole_trace[f_epoch[0]])
        
        axs[0, 0].axvline(x=3, color=[1,0,0,0.4], linestyle='-')
        axs[0, 0].axvline(x=6, color=[1,0,0,0.4], linestyle='-')
        
        for i_lum, lum in enumerate(uniq_lums):
            for i_diam,diam in enumerate(uniq_diams):
                curr_epoch = epochs[(lum == luminances) & (diameters == diam)]
                axs[i_lum+1, i_diam].plot(t_axis,roi.int_whole_trace[curr_epoch[0]])
                
                axs[i_lum+1, i_diam].set_title('l:{s}\nd:{d}'.format(s=lum,d =diam))
                
                axs[i_lum+1, i_diam].axvline(x=3, color=[1,0,0,0.4], linestyle='-')
                axs[i_lum+1, i_diam].axvline(x=6, color=[1,0,0,0.4], linestyle='-')

                if i_lum+1 == len(uniq_lums):
                    axs[i_lum+1, i_diam].set_xlabel('Time (s)')
        axs[i_lum+1, i_diam].set_xlim([0,10])

        f2_n = 'Traces_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig2.savefig('%s.pdf'% f2_n, bbox_inches='tight',
                    transparent=False)
    elif analysis_type == 'center_grating_miri' :
        
            
        rois = PyROI.analyzeSineGratings(rois)
        roi = rois[0]
        plt.plot(roi.resp_traces['epoch_2'],label='centered')
        plt.plot(roi.resp_traces['epoch_3'],label='10deg away')

        

        plt.xlabel('Frames, fr={fr}FPS'.format(fr=roi.imaging_info['frame_rate']))
        plt.ylabel('dF/F')
        plt.legend()
        fig = plt.gcf()
        f0_n = 'Summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.pdf'% f0_n, bbox_inches='tight',
                    transparent=False)

    elif analysis_type == 'contrast_flashes_w_BG':
        rois = PyROI.analyze_contrast_flashes(rois,int_rate = 10)
        resps = np.array(map(lambda roi: roi.int_whole_trace.values(), rois))

        fig, ax = plt.subplots(2, 1, sharex='all', sharey='all')
        ax = ax.flatten()

        ax[0].plot(resps[:,0,:].T,alpha=.5,color='k')
        ax[0].plot(np.nanmean(resps[:,0,:],axis=0),color='r',linewidth=2)
        ax[0].set_xlabel('Time (decaseconds)')
        ax[0].set_ylabel('$\Delta F/F$')
        ax[0].set_title('Epoch weber c: {c}'.format(c=rois[0].weber_c.values()[0]))

        ax[1].plot(resps[:,1,:].T,alpha=.5,color='k')
        ax[1].plot(np.nanmean(resps[:,1,:],axis=0),color='r',linewidth=2)
        ax[1].set_xlabel('Time (decaseconds)')
        ax[1].set_ylabel('$\Delta F/F$')
        ax[1].set_title('Epoch weber c: {c}'.format(c=rois[0].weber_c.values()[1]))
        
        
        ax[0].vlines([40,60],ax[0].get_ylim()[0],ax[0].get_ylim()[1])
        ax[1].vlines([40,60],ax[1].get_ylim()[0],ax[1].get_ylim()[1])


        f_n = 'Summary' 
        os.chdir(fig_save_dir)
        fig.savefig('%s.pdf'% f_n, bbox_inches='tight',
                   transparent=False,dpi=300)
                   
    elif analysis_type == 'lum_con_gratings':
        
        rois = PyROI.analyzeSineGratings_generalized(rois,int_rate = 10)
        # run_matplotlib_params()
        
        for roi in rois:
            roi_dict = {}
            roi_dict['Luminance'] = roi.epoch_luminances.values()
            roi_dict['Contrast'] = roi.epoch_contrasts.values()
            roi_dict['Response'] = roi.power_at_sineFreq.values()
            
            df_roi = pd.DataFrame.from_dict(roi_dict)
            cl_map = df_roi.pivot(index='Contrast',columns='Luminance')
            roi.cl_map= cl_map
            roi.cl_map_norm=(cl_map-cl_map.mean())/cl_map.std()

        mean_CL = np.mean([np.array(roi.cl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_CL, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].cl_map.columns.levels[1]).astype(float),
                       yticklabels=np.array(rois[0].cl_map.index),
                       cbar_kws={'label': 'Response'})
        ax.invert_yaxis()
        plt.title('CL map')
        plt.xlabel('Luminance')
        plt.ylabel('Contrast')

        fig = plt.gcf()
        f0_n = 'Summary_CL_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False)
    
    elif analysis_type == 'lum_lambda_gratings':
        
        rois = PyROI.analyzeSineGratings_generalized(rois,int_rate = 10)
        # run_matplotlib_params()
        
        for roi in rois:
            roi_dict = {}
            roi_dict['Luminance'] = roi.epoch_luminances.values()
            roi_dict['Lambda'] = roi.epoch_spatial_wavelengths.values()
            roi_dict['Response'] = roi.power_at_sineFreq.values()
            
            df_roi = pd.DataFrame.from_dict(roi_dict)
            cl_map = df_roi.pivot(index='Lambda',columns='Luminance')
            roi.cl_map= cl_map
            roi.cl_map_norm=(cl_map-cl_map.mean())/cl_map.std()

        mean_CL = np.mean([np.array(roi.cl_map) for roi in rois],axis=0)
        fig = plt.figure(figsize = (5,5))
        
        ax=sns.heatmap(mean_CL, cmap='coolwarm',center=0,
                       xticklabels=np.array(rois[0].cl_map.columns.levels[1]).astype(float),
                       yticklabels=np.array(rois[0].cl_map.index),
                       cbar_kws={'label': 'Response'})
        ax.invert_yaxis()
        plt.title('LL map')
        plt.xlabel('Luminance')
        plt.ylabel('Lambda')

        fig = plt.gcf()
        f0_n = 'Summary_LL_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                    transparent=False)
        
        for roi_id , roi in enumerate(rois):
            luminances = np.array(roi.epoch_luminances.values())
            lambdas = np.array(roi.epoch_spatial_wavelengths.values())

            dim1 = 15
            dim2 = 15
            fig2, axs = plt.subplots(nrows=len(np.unique(lambdas)), ncols=len(np.unique(luminances)), figsize=(dim1, dim2),sharey=True,sharex=True)

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.8, hspace=0.2)

            uniq_lums = np.unique(luminances)

            uniq_lambdas = np.unique(lambdas)
            epochs = np.array(roi.int_whole_trace.keys())
            
            t_axis = np.linspace(0,12,len(roi.int_whole_trace['epoch_2']))

            
            for i_diam,diam in enumerate(uniq_lambdas):
                for i_lum, lum in enumerate(uniq_lums):
                
                    curr_epoch = epochs[(lum == luminances) & (lambdas == diam)]
                    axs[i_diam, i_lum].plot(t_axis,roi.int_whole_trace[curr_epoch[0]])
                    
                    axs[i_diam, i_lum].set_title('lum:{s}\nlambda:{d}'.format(s=lum,d =diam))
                    
                    axs[i_diam, i_lum].axvline(x=4, color=[1,0,0,0.4], linestyle='-')
                    axs[i_diam, i_lum].axvline(x=8, color=[1,0,0,0.4], linestyle='-')

                    if i_diam == len(uniq_lambdas):
                        axs[i_diam, i_lum].set_xlabel('Time (s)')
            axs[i_diam, i_lum].set_xlim([0,12])

            f2_n = 'Traces_roi%s_%s' % (roi_id,rois[0].experiment_info['MovieID'])
            os.chdir(fig_save_dir)
            fig2.savefig('%s.pdf'% f2_n, bbox_inches='tight',
                        transparent=False)
        
    elif analysis_type == 'TF_lum_gratings':

            rois = PyROI.analyzeSineGratings_generalized(rois,int_rate = 10)

            for roi in rois:
                roi_dict = {}
                roi_dict['Luminance'] = roi.epoch_luminances.values()
                roi_dict['TF'] = roi.epoch_temp_frequencies.values()
                roi_dict['Response'] = roi.power_at_sineFreq.values()
                
                df_roi = pd.DataFrame.from_dict(roi_dict)
                tf_map = df_roi.pivot(index='TF',columns='Luminance')
                roi.tf_map= tf_map
                roi.tf_map=(tf_map-tf_map.mean())/tf_map.std()

            mean_TF = np.mean([np.array(roi.tf_map) for roi in rois],axis=0)
            fig = plt.figure(figsize = (5,5))
            
            ax=sns.heatmap(mean_TF, cmap='coolwarm',center=0,
                        xticklabels=np.array(rois[0].tf_map.columns.levels[1]).astype(float),
                        yticklabels=np.array(rois[0].tf_map.index),
                        cbar_kws={'label': 'Response'})
            ax.invert_yaxis()
            plt.title('TF map')
            plt.xlabel('Luminance')
            plt.ylabel('TF')

            fig = plt.gcf()
            f0_n = 'Summary_TFL_%s' % (rois[0].experiment_info['MovieID'])
            os.chdir(fig_save_dir)
            fig.savefig('%s.png'% f0_n, bbox_inches='tight',
                        transparent=False)
    
    elif analysis_type == '5sFFF_analyze_save':

        rois = PyROI.fffAnalyze(rois, interpolation = True, int_rate = 10)
        roi_conc_traces = list(map(lambda roi: roi.conc_trace, rois))
        stim_trace  = rois[0].stim_trace
        fig = sf.fffSummary(figtitle,stim_trace, roi_conc_traces,
                    True,rois[0].experiment_info['MovieID'],summary_save_d)
        f1_n = '5sFFF_summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)
                       
    elif analysis_type == 'A_B_time_adaptation':
        # Fixed A and B contrast but A step times changing
        #TODO: Fix the plots make them nicer etc.
        rois = PyROI.analyzeAB_steps_time(rois)
        b_steps = []
        a_steps = []
        for roi in rois:
            b_steps.append(roi.sorted_b_steps)
            plt.scatter(roi.sorted_a_durs,roi.sorted_b_steps)
            plt.scatter(32,np.mean(roi.a_step_response.values()))
            a_steps.append(np.mean(roi.a_step_response.values()))
        mean_r = np.mean(b_steps,axis=0)
        plt.plot(roi.sorted_a_durs,mean_r,color='k')
        std_r = np.std(b_steps,axis=0)
        ub = mean_r + std_r
        lb = mean_r - std_r
        plt.fill_between(roi.sorted_a_durs, ub, lb, color='k',alpha=.3)

        plt.xlabel('A step time')
        plt.ylabel('B step response')

        fig = plt.gcf()
        f1_n = 'AB_time_summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)

    elif ((analysis_type == 'luminance_edges_OFF' ) or\
          (analysis_type == 'luminance_edges_ON' )) :
        
        if ('T4' in rois[0].experiment_info['Genotype'] or \
            'T5' in rois[0].experiment_info['Genotype']):
            map(lambda roi: roi.calculate_DSI_PD(method='PDND'), rois)
        rois = PyROI.analyzeLuminanceEdges(rois,int_rate = 10)
        roi_image = PyROI.generatePropertyMasks(rois, 'slope')
        fig = sf.summarizeLuminanceEdges(figtitle,rois,roi_image,
                                            rois[0].experiment_info['MovieID'],
                                            summary_save_d)
        
        
        slope_data = PyROI.dataToList(rois, ['slope'])['slope']
        rangecolor= np.max(np.abs([np.min(slope_data),np.max(slope_data)]))
        
        if 'RF_map' in rois[0].__dict__:
            fig2 = ROI_mod.plot_RF_centers_on_screen(rois,prop='slope',
                                                     cmap='PRGn',
                                                     ylab='Lum sensitivity',
                                                     lims=(-rangecolor,
                                                           rangecolor))
            f2_n = 'Slope_on_screen_%s' % (rois[0].experiment_info['MovieID'])
            os.chdir(fig_save_dir)
            fig2.savefig('%s.png'% f2_n, bbox_inches='tight',
                           transparent=False,dpi=300)
        else:
            print('No RF found for the ROI.')
        
        f1_n = 'Summary_%s' % (rois[0].experiment_info['MovieID'])
        os.chdir(fig_save_dir)
        fig.savefig('%s.png'% f1_n, bbox_inches='tight',
                       transparent=False,dpi=300)

    return rois

def plotAllMasks(roi_image, underlying_image,n_roi1,exp_ID,
                       save_fig = False, save_dir = None,alpha=0.5):
    """ 

    """

    plt.close('all')
    plt.style.use("dark_background")

    # All masks
    plt.imshow(underlying_image,cmap='gray')
    plt.imshow(roi_image,alpha=alpha,cmap = 'tab20b')
    
    ax1=plt.gca()
    ax1.axis('off')
    ax1.set_title('ROIs n=%d' % n_roi1)
    
    if save_fig:
        # Saving figure
        save_name = 'ROIs_%s' % (exp_ID)
        os.chdir(save_dir)
        plt.savefig('%s.png'% save_name, bbox_inches='tight')
        print('ROI images saved')
    return None