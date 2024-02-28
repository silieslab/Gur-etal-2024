#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:23:20 2018

@author: burakgur / Necessary functions for doing batch analysis
"""
from __future__ import division
from shutil import copyfile
import os
import warnings
import glob
import numpy as np
import re

from core_functions import motionCorrection, getEpochCount
from core_functions import readStimOut, divideEpochs, readStimInformation
from core_functions import getROIinformation, subtractBg, dff, trialAverage
from core_functions import corrTrialAvg, interpolateTrialAvgROIs, saveWorkspace
from core_functions import extractRawSignal
from xmlUtilities import getFramePeriod, getLayerPosition, getMicRelativeTime


def sortStimulusFiles(stimulusPath):
    """Cleans the stimulus folder (deletes all files except stimulus output)
    and sorts the stimulus output files according to their time of last 
    modification.

    Parameters
    ==========
    stimulusPath: str
        Path of the stimulus folder containing all the corresponding stimuli
    
    Returns
    =======
    all_stim_files_clean: ordered stimulus output files
         
    """

    all_stim_files = os.listdir(stimulusPath)

    # Deleting the unncessary folders
    for file_name in all_stim_files:
        # If it's not a stimulus output file, delete it
        if ('stimulus_output' not in file_name):
            file_to_remove_path = os.path.join(stimulusPath, file_name)
            os.remove(file_to_remove_path)

        # If it is a search file, delete it
        # Determined by checking the stimulus description which is on the first
        # line of the .txt file
        else:
            f = open(os.path.join(stimulusPath, file_name), "r")
            stimulus_description = f.readline()
            f.close()
            if 'search' in stimulus_description.lower():
                file_to_remove_path = os.path.join(stimulusPath, file_name)
                os.remove(file_to_remove_path)

    all_stim_files_clean = os.listdir(stimulusPath)
    # Sort the remaining files according to their time of last modification   
    all_stim_files_clean.sort(key=lambda \
            stim_file: int(''.join(stim_file.split('.')[0].split('_')[-3:])))

    return all_stim_files_clean

def sortStimulusFiles_PyStim(stimulusPath):
    """Cleans the stimulus folder (deletes all files except stimulus output)
    and sorts the stimulus output files according to their time of last 
    modification.

    Parameters
    ==========
    stimulusPath: str
        Path of the stimulus folder containing all the corresponding stimuli
    
    Returns
    =======
    all_stim_files_clean: ordered stimulus output files
         
    """

    all_stim_files = os.listdir(stimulusPath)
    all_stim_output_files = []

    # Ignoring the unnecessary files
    for file_name in all_stim_files:
        # Check if it is a stimulus out file that is used in a real experiment via synchronizing with NIDAQ
        if ('stimulus_output' in file_name) and ('NIDAQ-True'in file_name):
            all_stim_output_files.append(file_name)

    
    # Sort the remaining files according to their time last 6 digits which is 
    # time of creation
    all_stim_output_files.sort(key=lambda \
            stim_file: int(stim_file.split('.')[0][-6:])) 

    return all_stim_output_files

def preProcessDataFolders(rawDataDir, stimDir):
    """Puts the stimulus output files automatically to the T-series folders. 
       Find the images and stimuli according to the flyIDs. Sorts them 
       according to time and matches them.
       
       Naming scheme:
       Experiment folder name: Has to include the string "fly" inside
       Stimulus folder name: Has to include the experiment folder name within

    Parameters
    ==========
    rawDataDir : str
        Path of the folder where the raw data is located.
        
    stimDir : str
        Path of the folder where the stimuli are located.



    Returns
    =======
    
    """
    print('Pre-processing the data folders...\n')

    all_data_folders = os.listdir(rawDataDir)
    all_stim_folders = os.listdir(stimDir)
    for data_folder_name in all_data_folders:
        if data_folder_name[0]=='.':
            continue
        # Enter if it is an image data folder determined by fly ID
        if 'fly' in data_folder_name.lower():  # Has to include 'fly' in its ID
            images_path = os.path.join(rawDataDir, data_folder_name)
            current_exp_ID = data_folder_name.lower()

            # Finding the T-series
            t_series_names = [file_n for file_n in os.listdir(images_path) \
                              if 'tseries' in file_n.lower() or \
                              't-series' in file_n.lower()]
                
            t_series_paths = [os.path.join(images_path,file_n) for file_n \
                              in t_series_names]
            # Ordering the T-series according to their timing
            t_series_paths.sort(key=lambda t_series_path: \
                os.path.getmtime(t_series_path))

            # Searching for the stimulus file
            stim_found = False
            for stim_folder_name in all_stim_folders:
                if 'stimuli' in stim_folder_name.lower():
                    # Searching for the correct stimulus folder
                    if current_exp_ID in stim_folder_name.lower():
                        stimulus_path = os.path.join(stimDir,
                                                     stim_folder_name)
                        stimuli_output_names = sortStimulusFiles(
                            stimulus_path)
                        stim_found = True
                        break
            if not stim_found:
                warn_string = "!!!!Stimulus folder not found for %s...\n" % \
                              (current_exp_ID)
                warnings.warn(warn_string)
                continue
            # Copying the output files to the corresponding T-series folders
            if len(stimuli_output_names) == len(t_series_paths):
                print("Image and stimuli numbers match for  %s...\n" % \
                      (current_exp_ID))
                for i, stimuli_output_file in enumerate(stimuli_output_names):
                    os.rename(os.path.join(stimulus_path, stimuli_output_file),
                              os.path.join(t_series_paths[i],
                                           stimuli_output_file))

                print("Folder processing of %s completed...\n" % \
                      (current_exp_ID))
            else:
                warn_string = "!!!!Image and stimuli numbers DO NOT match for  %s...\n" % \
                              (current_exp_ID)
                warnings.warn(warn_string)
    return None

def preProcessDataFolders_PyStim(rawDataDir, stimDir):
    """Puts the stimulus output files automatically to the T-series folders. 
       Find the images and stimuli according to the flyIDs. Sorts them 
       according to time and matches them.
       
       Naming scheme:
       Experiment folder name: Has to include the string "fly" inside
       Stimulus folder name: Has to include the experiment folder name within

    Parameters
    ==========
    rawDataDir : str
        Path of the folder where the raw data is located.
        
    stimDir : str
        Path of the folder where the stimuli are located.



    Returns
    =======
    
    """
    print('Pre-processing the data folders...\n')

    all_data_folders = os.listdir(rawDataDir)
    all_stim_folders = os.listdir(stimDir)
    for data_folder_name in all_data_folders:
        if data_folder_name[0]=='.':
            continue
        # Enter if it is an image data folder determined by fly ID
        if 'fly' in data_folder_name.lower():  # Has to include 'fly' in its ID
            images_path = os.path.join(rawDataDir, data_folder_name)
            current_exp_ID = data_folder_name.lower()

            # Finding the T-series
            t_series_names = [file_n for file_n in os.listdir(images_path) \
                              if 'tseries' in file_n.lower() or \
                              't-series' in file_n.lower()]
                
            t_series_paths = [os.path.join(images_path,file_n) for file_n \
                              in t_series_names]
            # Ordering the T-series according to their timing
            key=lambda \
            stim_file: int(stim_file.split('.')[0][-6:])
            
            t_series_paths.sort(key=lambda t_series_path: \
                os.path.getmtime(t_series_path))

            # Searching for the stimulus file
            stim_found = False
            for stim_folder_name in all_stim_folders:
                if 'stim' in stim_folder_name.lower():
                    # Searching for the correct stimulus folder
                    if current_exp_ID in stim_folder_name.lower():
                        stimulus_path = os.path.join(stimDir,
                                                     stim_folder_name)
                        stimuli_output_names = sortStimulusFiles_PyStim(
                            stimulus_path)
                        stim_found = True
                        break
            if not stim_found:
                warn_string = "!!!!Stimulus folder not found for %s...\n" % \
                              (current_exp_ID)
                warnings.warn(warn_string)
                continue
            # Copying the output files to the corresponding T-series folders
            # Since we have .mat .pickle and .txt files we advance T series counter every 3 steps
            if len(stimuli_output_names)/3.0 == len(t_series_paths):
                print("Image and stimuli numbers match for  %s...\n" % \
                      (current_exp_ID))
                t_counter = -1
                for i, stimuli_output_file in enumerate(stimuli_output_names):
                    if np.mod(i,3) == 0:
                        t_counter += 1
                    os.rename(os.path.join(stimulus_path, stimuli_output_file),
                              os.path.join(t_series_paths[t_counter],
                                           stimuli_output_file))

                print("Folder processing of %s completed...\n" % \
                      (current_exp_ID))
            else:
                warn_string = "!!!!Image and stimuli numbers DO NOT match for  %s...\n" % \
                              (current_exp_ID)
                warnings.warn(warn_string)
    return None

def batchMotionAlignmentPyStim(rawDataDir, outputDir, granularity='plane', combined=False,bleedthrough_correct=False):
    """ Does batch motion alignment of experiments. Experiments has to include 
    image data within folders named "t-series" or "tseries" (case insensitive)

    Parameters
    ==========
    initialDirectory : str, optional
        Default: root directory, i.e. '/'.

        Path into which the directory selection GUI opens.



    Returns
    =======
    
    """
    print('Running batch motion alignment...')

    all_folders = os.listdir(rawDataDir)
    for folder_name in all_folders:
        # Enter if it is an experiment folder
        if 'fly' in folder_name.lower():
            experiment_path = os.path.join(rawDataDir, folder_name)
            current_exp_ID = folder_name.lower()
            print("\n\n-Aligning %s...\n" % current_exp_ID)
            
            # Create an ouput experiment directory, if exists use that one
            try:
                # instead of os.makedirs use os.mkdir
                # the former creates also intermediate dirs
                os.mkdir(os.path.join(outputDir, current_exp_ID))
            except OSError as err:
                if not 'File exists' in err:
                  raise OSError
                
            # Finding T-series for aligning
            if combined:
                t_series_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'combined' in file_n.lower()]
                ts = None

            else:
                # Do not align the combined ones individually
                combined_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'combined' in file_n.lower()]
                ts = []
                if combined_names: 
                    
                    for combined_n in combined_names:
                        ts.append(np.array(re.findall(r'\d+',combined_n)).astype(int))
                    np.array(ts).flatten()
                    np.concatenate(ts)
                    
                t_series_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'tseries' in file_n.lower() or \
                                  't-series' in file_n.lower()]
                    
            for iSeries, serie_name in enumerate(t_series_names):
                serie_num = np.array(re.findall(r'\d+',serie_name)).astype(int)[-1]
                
                if ts is not None :
                    try:
                        ts = np.concatenate(ts)
                    except:
                        ts = ts
                    
                if (not(combined)) and (np.isin(serie_num,ts)):
                    print('--Skipping {s} since it has a combined version.\n'.format(s=serie_name))
                    continue
                
                    
                t_series_path = os.path.join(experiment_path, serie_name)
                t_series_files = t_series_path + '/' + '*.tif'
                output_path = os.path.join(outputDir, current_exp_ID)

                # Align T-series
                unique_t_series = motionCorrection(serie_name, output_path, t_series_files,
                                                   maxDisplacement=[40, 40],
                                                   granularity=granularity,
                                                   exportFrames=True,bleedthrough_correct=bleedthrough_correct)
                if not(unique_t_series):
                    print('--Skipping {s} since the target directory already exists.\n'.format(s=serie_name))
                    continue
                for t_name in unique_t_series:
                    if len(unique_t_series) > 1:
                        # Directory shouldn't exist otherwise break
                        try:
                            # instead of os.makedirs use os.mkdir
                            # the former creates also intermediate dirs
                            os.mkdir(os.path.join(outputDir, current_exp_ID,t_name))
                        except OSError:
                            print('Either directory exists or \
                                  the intermedite directories do not exist!')
                            raise
                    t_s_path = os.path.join(experiment_path, t_name)
                    copyStimXmlFilesPyStim(output_path, t_s_path, t_name)

    print("---Batch motion alignment done---")

def batchMotionAlignment(rawDataDir, outputDir, granularity='plane', combined=False,bleedthrough_correct=False):
    """ Does batch motion alignment of experiments. Experiments has to include 
    image data within folders named "t-series" or "tseries" (case insensitive)

    Parameters
    ==========
    initialDirectory : str, optional
        Default: root directory, i.e. '/'.

        Path into which the directory selection GUI opens.



    Returns
    =======
    
    """
    print('Running batch motion alignment...')

    all_folders = os.listdir(rawDataDir)
    for folder_name in all_folders:
        # Enter if it is an experiment folder
        if 'fly' in folder_name.lower():
            experiment_path = os.path.join(rawDataDir, folder_name)
            current_exp_ID = folder_name.lower()
            print("\n\n-Aligning %s...\n" % current_exp_ID)
            
            # Create an ouput experiment directory, if exists use that one
            try:
                # instead of os.makedirs use os.mkdir
                # the former creates also intermediate dirs
                os.mkdir(os.path.join(outputDir, current_exp_ID))
            except OSError as err:
                if not 'File exists' in err:
                  raise OSError
                
            # Finding T-series for aligning
            if combined:
                t_series_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'combined' in file_n.lower()]
                ts = None

            else:
                # Do not align the combined ones individually
                combined_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'combined' in file_n.lower()]
                ts = []
                if combined_names: 
                    
                    for combined_n in combined_names:
                        ts.append(np.array(re.findall(r'\d+',combined_n)).astype(int))
                    np.array(ts).flatten()
                    np.concatenate(ts)
                    
                t_series_names = [file_n for file_n in os.listdir(experiment_path) \
                                  if 'tseries' in file_n.lower() or \
                                  't-series' in file_n.lower()]
                    
            for iSeries, serie_name in enumerate(t_series_names):
                serie_num = np.array(re.findall(r'\d+',serie_name)).astype(int)[-1]
                
                if ts is not None :
                    try:
                        ts = np.concatenate(ts)
                    except:
                        ts = ts
                    
                if (not(combined)) and (np.isin(serie_num,ts)):
                    print('--Skipping {s} since it has a combined version.\n'.format(s=serie_name))
                    continue
                
                    
                t_series_path = os.path.join(experiment_path, serie_name)
                t_series_files = t_series_path + '/' + '*.tif'
                output_path = os.path.join(outputDir, current_exp_ID)

                # Align T-series
                unique_t_series = motionCorrection(serie_name, output_path, t_series_files,
                                                   maxDisplacement=[40, 40],
                                                   granularity=granularity,
                                                   exportFrames=True,bleedthrough_correct=bleedthrough_correct)
                if not(unique_t_series):
                    print('--Skipping {s} since the target directory already exists.\n'.format(s=serie_name))
                    continue
                for t_name in unique_t_series:
                    if len(unique_t_series) > 1:
                        # Directory shouldn't exist otherwise break
                        try:
                            # instead of os.makedirs use os.mkdir
                            # the former creates also intermediate dirs
                            os.mkdir(os.path.join(outputDir, current_exp_ID,t_name))
                        except OSError:
                            print('Either directory exists or \
                                  the intermedite directories do not exist!')
                            raise
                    t_s_path = os.path.join(experiment_path, t_name)
                    copyStimXmlFiles(output_path, t_s_path, t_name)

    print("---Batch motion alignment done---")


def copyStimXmlFiles(output_path, t_series_path, t_name):
    """
    :return:
    """
    # Copying the stimulus output and xml file, for convenience in
    # further processes.
    stim_out_path = os.path.join(t_series_path, '_stimulus_output_*')
    stim_out_file_path = (glob.glob(stim_out_path))
    if not stim_out_file_path:
        print('!!Stimulus file not found -> %s!!' % t_series_path)
    else:
        stim_out_file = stim_out_file_path[0]
        stim_out_name = os.path.basename(stim_out_file)
        copyfile(stim_out_file_path[0], os.path.join(output_path, t_name,
                                                     stim_out_name))

    # Copying all xml files
    xml_path = os.path.join(t_series_path, '*.xml')
    xml_file_path = (glob.glob(xml_path))

    for xml_path_name in xml_file_path:
        xml_name = os.path.basename(xml_path_name)
        copyfile(xml_path_name, os.path.join(output_path, t_name,
                                             xml_name))
def copyStimXmlFilesPyStim(output_path, t_series_path, t_name):
    """
    :return:
    """
    # Copying the stimulus output and xml file, for convenience in
    # further processes.
    stim_out_path = os.path.join(t_series_path, 'stimulus_output_*')
    stim_out_file_path = (glob.glob(stim_out_path))
    if not stim_out_file_path:
        print('!!Stimulus file not found -> %s!!' % t_series_path)
    else:
        # There are multiple stim output files (different extensions)
        for stim_out_file in stim_out_file_path:
            stim_out_name = os.path.basename(stim_out_file)
            copyfile(stim_out_file, os.path.join(output_path, t_name,
                                                        stim_out_name))

    # Copying all xml files
    xml_path = os.path.join(t_series_path, '*.xml')
    xml_file_path = (glob.glob(xml_path))

    for xml_path_name in xml_file_path:
        xml_name = os.path.basename(xml_path_name)
        copyfile(xml_path_name, os.path.join(output_path, t_name,
                                             xml_name))

def batchSignalPlotSave(alignedDataDir, initialDirectory=
"/Users/burakgur/2p/Python_data", use_aligned=True,
                        mode='Stimulus'):
    """ Batch extracts signals from a selected ROI set and saves the plots of 
    already selected ROI signals.

    Parameters
    ==========
    alignedDataDir: str
        Path into the directory where the motion corrected data will be plotted
    
    initialDirectory : str, optional
        Default: root directory, i.e. '/'.

        Path into which the directory selection GUI opens.
     
    use_aligned: bool, optional
        Default: True
        
        Defines if aligned 'motCorr.sima' or non-aligned 'TIFFs.sima' will be used.

    Returns
    =======
    
    """
    print('Extracting all signals and saving figures...\n')

    all_folders = os.listdir(alignedDataDir)
    for folder_name in all_folders:
        # Enter if it is an experiment folder
        if 'fly' in folder_name.lower():
            experiment_path = os.path.join(alignedDataDir, folder_name)
            current_exp_ID = folder_name.lower()

            print(">Processing %s...\n" % (current_exp_ID))

            # Finding T-series for aligning
            t_series_names = [file_n for file_n in os.listdir(experiment_path) \
                              if 'tseries' in file_n.lower() or \
                              't-series' in file_n.lower()]
            for iTseries, t_name in enumerate(t_series_names):
                t_series_path = os.path.join(experiment_path, t_name)
                if use_aligned:
                    print('Using the aligned sequences for extraction')
                    dataDir = os.path.join(t_series_path, 'motCorr.sima')
                else:
                    print('Using the non-aligned sequences for extraction')
                    dataDir = os.path.join(t_series_path, 'TIFFs.sima')

                plotSaveROISignals(dataDir, saveFig=True, mode=mode)
                print(">Figure successfully saved for: %s...\n" % (t_name))

    print("---Figures are saved, check the log for errors---")
    return


def dataProcessSave(t_series_path, stimInputDir, saveOutputDir, imageID,
                    current_exp_ID, use_aligned=True, intRate=10):
    """ Processes the data and saves the necessary variables

    Parameters
    ==========
    t_series_path : str

        Path of the T-series that includes the motion correction directory
        along with stimulus output and xml file.
        
    stimInputDir : str

        Path of the folder where stimulus input files are located. These files
        contain information about all the stimuli used in the experiments.
        
    saveOutputDir : str

        Path of the folder where the data output files will be saved
        
    imageID : str

        The unique ID of the image data to be saved
        
    current_exp_ID : str

        The experiment ID of the image data to be saved
        
    use_aligned: bool, optional
        Default: True
        
        Defines if aligned 'motCorr.sima' or non-aligned 'TIFFs.sima' will be used.
    
    intRate: int, optional
        Default: 10
        
        The rate which data will be interpolated.
        
  
    Returns
    =======
    
    """
    if use_aligned:
        print('Using the aligned sequences for extraction')
        dataDir = os.path.join(t_series_path, 'motCorr.sima')
    else:
        print('Using the non-aligned sequences for extraction')
        dataDir = os.path.join(t_series_path, 'TIFFs.sima')

    t_series_name = os.path.basename(t_series_path)
    # Finding the xml file and retrieving relevant information
    xmlPath = os.path.join(t_series_path, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]
    micRelTimes = getMicRelativeTime(xmlFile)

    #  Finding the frame period (1/FPS) and layer position
    framePeriod = getFramePeriod(xmlFile=xmlFile)
    layerPosition = getLayerPosition(xmlFile=xmlFile)

    # Finding and reading the stimulus output file, extracting relevant info
    stimOutPath = os.path.join(t_series_path, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    (stimType, rawStimData) = readStimOut(stimOutFile=stimOutFile,
                                          skipHeader=1)
    (stimInputFile, stimInputData) = readStimInformation(stimType=stimType,
                                                         stimInputDir=stimInputDir)
    stimName = os.path.basename(stimInputFile)
    isRandom = int(stimInputData['Stimulus.randomize'][0])
    epochDur = stimInputData['Stimulus.duration']
    epochDur = [float(sec) for sec in epochDur]

    # Finding epoch coordinates and number of trials                                        
    epochCount = getEpochCount(rawStimData=rawStimData, epochColumn=3)
    (trialCoor, trialCount, isRandom) = divideEpochs(rawStimData=rawStimData,
                                                     epochCount=epochCount,
                                                     isRandom=isRandom,
                                                     framePeriod=framePeriod,
                                                     trialDiff=0.20,
                                                     overlappingFrames=0,
                                                     firstEpochIdx=0,
                                                     epochColumn=3,
                                                     imgFrameColumn=7,
                                                     incNextEpoch=True,
                                                     checkLastTrialLen=True)

    # Signal extraction and background subtraction
    print('Signal extraction...')
    (signalFile, chNames, usedChannel,
     roiKeys, usedRoiKey, usedExtLabel) = extractRawSignal(motCorrDir
                                                           =dataDir)

    # ROI information, header includes the ROI numbers
    # tags include the types of ROIs e.g. Layer1
    (header, bgIndex, tags) = getROIinformation(signalFile=signalFile,
                                                bgLabel=['bg', 0])
    (bgSub, rawTrace) = subtractBg(signalFile=signalFile, bgIndex=bgIndex,
                                   skipHeader=3)

    # Calculating dF/F according to the baseline type
    if isRandom == 1:  # There is an epoch used for baseline
        baselineEpochPresent = True
        baselineDurationBeforeEpoch = 1.5  # In seconds
        baseDur = int(baselineDurationBeforeEpoch / framePeriod)  # In frames
    else:  # Presumably no epoch present for baseline, taking the mean of trial
        baselineEpochPresent = False
        baselineDurationBeforeEpoch = np.nan
        baseDur = np.nan

    (dffTraceAllRoi,
     baselineStdAllRoi,
     baselineMeanAllRoi
     ) = dff(trialCoor=trialCoor, header=header,
             bgIndex=bgIndex, bgSub=bgSub,
             baselineEpochPresent=baselineEpochPresent,
             baseDur=baseDur)
    # Trial averaging
    trialAvgAllRoi = trialAverage(dffTraceAllRoi=dffTraceAllRoi,
                                  bgIndex=bgIndex)
    # Correlation with stimulus
    (corrHeader, pvalHeader) = corrTrialAvg(trialAvgAllRoi=trialAvgAllRoi,
                                            epochDur=epochDur, bgIndex=bgIndex,
                                            framePeriod=framePeriod)

    # Interpolation of responses to a certain frequency
    print('Interpolating to %d Hz', intRate)
    interpolationRate = intRate;  # Interpolation rate in Hz
    interpolatedAllRoi = interpolateTrialAvgROIs(trialAvgAllRoi=trialAvgAllRoi,
                                                 framePeriod=framePeriod,
                                                 intRate=interpolationRate)

    # locals() needs to be called within the script that
    # generates the variables to be saved
    varDict = locals()
    savePath = saveWorkspace(outDir=saveOutputDir, baseName=imageID,
                             varDict=varDict, varFile='variablesToSave.txt',
                             extension='.pickle')


def batchDataSave(stimInputDir, saveOutputDir, alignedDataDir, use_aligned=True,
                  intRate=10):
    """ Batch processes the data and saves the necessary variables

    Parameters
    ==========
    stimInputDir : str

        Path of the folder containing the stimulus input files.
    
    saveOutputDir : str

        Path of the folder for saving the output file.
        
    alignedDataDir : str
        
        Path of the folder that contains the motion aligned data.
    
    use_aligned: bool, optional
        Default: True
   
    intRate: int, optional
        Default: 10
        
        The rate which data will be interpolated.

    Returns
    =======
    
    """
    print('Processing data and saving the data output...\n')

    all_folders = os.listdir(alignedDataDir)
    for folder_name in all_folders:
        # Enter if it is an experiment folder
        if 'fly' in folder_name.lower():
            experiment_path = os.path.join(alignedDataDir, folder_name)
            current_exp_ID = folder_name.lower()

            print(">Processing %s...\n" % (current_exp_ID))

            # Finding T-series for aligning
            t_series_names = [file_n for file_n in os.listdir(experiment_path) \
                              if 'tseries' in file_n.lower() or \
                              't-series' in file_n.lower()]
            for iTseries, t_name in enumerate(t_series_names):
                t_series_path = os.path.join(experiment_path, t_name)
                imageID = current_exp_ID + '-' + t_name
                print(">>Processing %s...\n" % (imageID))
                # Process and save the data
                dataProcessSave(t_series_path, stimInputDir, saveOutputDir,
                                imageID, current_exp_ID, use_aligned=use_aligned,
                                intRate=intRate)
                print("%s saved." % imageID)

    print("---All data saved---")
    return None


def moveProcessedData(processedDataDir=
                      "/Users/burakgur/2p/Python_data/analyzed_data",
                      dataBaseDir=
                      "/Users/burakgur/2p/Python_data/database",
                      saveProcessedDataDir=
                      "/Users/burakgur/2p/Python_data/database/processed_data"):
    """ Batch processes the data and saves the necessary variables

    Parameters
    ==========
    saveOutputDir : str

        Path of the folder containing the processed data. The metadata should
        be extracted to the database before using this function.


    Returns
    =======
    
    """
    all_files = os.listdir(processedDataDir)
    for file_name in all_files:
        # Take if it's a data file
        if 'fly' in file_name.lower():
            file_path = os.path.join(processedDataDir, file_name)
            os.rename(file_path, os.path.join(saveProcessedDataDir,
                                              file_name))

            print(">Moved %s...\n" % (file_name))

            # Finding T-series for aligning

    print("---All data files are moved---")

    return None
