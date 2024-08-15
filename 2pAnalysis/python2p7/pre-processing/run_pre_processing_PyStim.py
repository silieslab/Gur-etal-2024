#%% 1st step of pre-processing: This code performs folder processing and SIMA package based motion correction of raw time series and stimulus outputs
# Requires specific folder structure: 
#     example T series path: raw_data>200122bgfly1>T-series-1
#     example stimulus path: stimuli>200122bgfly1>.... (has to include stimuli used for all T series so it matches)

#%% Importing required packages 
import os
import time
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/pre-processing')

from batch_analysis_functions import batchMotionAlignmentPyStim, preProcessDataFolders_PyStim

#%% User dependent directories, modify them according to your folder structure

# Directory where all the other directories located
initialDirectory = '/Users/burakgur/Desktop/Python_data'

# Directory containing the raw data and raw stimuli outputs
#initialDirectory = "/Users/burakgur/2p/Python_data"
rawDataDir = os.path.join(initialDirectory,'raw_data')
stimDir = os.path.join(initialDirectory,'stimuli')

# Directory where the motion corrected movies and datasets will be saved
alignedDataDir = os.path.join(initialDirectory,'motion_corrected')    

#%% Organizing the folders of images with stimuli for further processing
print('Organizing the folders of images with stimuli')
preProcessDataFolders_PyStim(rawDataDir, stimDir)


#%% Batch Motion correction (plane based Hidden Markov)
bleedthrough_correct = False
print('Performing plane batch motion alignment...\n')
plane_align_path = os.path.join(alignedDataDir,'plane_alignment')
start1 = time.time()
batchMotionAlignmentPyStim(rawDataDir, plane_align_path,granularity ='plane',bleedthrough_correct=bleedthrough_correct)
end1 = time.time()
time_passed = end1-start1
print('Batch motion alignment successfully done in %d minutes\n' % \
      round(time_passed/60) )
