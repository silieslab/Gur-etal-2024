#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 00:38:51 2018

@author: burakgur/catay aydin / Functions for 2 photon image and data 
    processing. Most functions are created by Catay Aydin.
"""

from __future__ import division
from tkFileDialog import askdirectory, askopenfilename
from itertools import islice
from scipy.stats.stats import pearsonr
from collections import Counter
import re
import os
import numpy
import sima
import Tkinter
import glob
import cPickle
import multiprocessing
import sima.segment
import pandas as pd
import numpy as np
import warnings
from PIL import Image
from scipy.fftpack import fft2,fftshift,ifftshift,ifft2  
from scipy.signal.windows import gaussian


def getDataDirectory(initialDirectory='~'):
    """Gets the raw data directory and analysis output directory paths.

    Parameters
    ==========

    initialDirectory : str, optional
        Default: HOME directory, i.e. '~'.

        Path into which the directory selection GUI opens.

    Returns
    =======
    rawDataDir : str
        Raw data path.

    rawDataFile : str
        Raw tif files' path.

    xmlFile : str
        XML file path.

    outDir : str
        Output diectory path.

    baseName : str
        Name of the time series folder.

    stimOutFile : str
        Stimulus output file path.
    """
    root = Tkinter.Tk()
    root.withdraw()

    rawDataDir = askdirectory(parent=root,
                              initialdir=initialDirectory,
                              title='Raw Data Directory')
    outDir = askdirectory(parent=root,
                          initialdir=initialDirectory,
                          title='Output Directory')

    rawDataFile = os.path.join(rawDataDir, '*.tif')
    stimOutPath = os.path.join(rawDataDir, '_stimulus_output_*')
    stimOutFile = (glob.glob(stimOutPath))[0]
    baseName = os.path.basename(rawDataDir)
    # there might be other xml file in the directory
    # e.g. when you use markpoints
    # so use glob style regex to get only the main xml
    xmlPath = os.path.join(rawDataDir, '*-???.xml')
    xmlFile = (glob.glob(xmlPath))[0]

    return rawDataDir, rawDataFile, xmlFile, outDir, baseName, stimOutFile

def selectMotCorrApproach(approachName, maxDisplacement=[40, 40],
                     nCpu=(multiprocessing.cpu_count() - 1)):
    """ Selects the user defined motion correction approach from the sima package.

    Parameters
    ==========
    approachName : str
        Name of the time series folder.

    maxDisplacement : list, optional
        Default: [40,40]

        Defines the maximum amount of movement (in *pixels*) that
        the motion correction should consider.

    nCpu : int, optional
        Default: (Maximum number of CPUs) - 1

        Number of CPUs to be used when estimating model parameters for
        motion correction.
        
    Returns
    =======
    motCorrApproach : sima object encapsulation the motion correction approach.
     
    """
    if approachName == 'HiddenMarkov2D':
        mc_approach = sima.motion.HiddenMarkov2D(granularity='plane',
                                                 max_displacement=maxDisplacement,
                                                 verbose=True,
                                                 n_processes=nCpu)
        
    return mc_approach

def motionCorrection(baseName, outDir, rawDataFile, granularity ='plane',
                     maxDisplacement=[40, 40],
                     nCpu=(multiprocessing.cpu_count() - 1), 
                     exportFrames=True, bleedthrough_correct=False):
    """ Does motion correction of time series. Also produces a non-aligned dataset.

    Parameters
    ==========
    baseName : str
        Name of the time series folder.

    outDir : str
        Output diectory path.

    rawDataFile : str
        Raw tif files' path.

    maxDisplacement : list, optional
        Default: [40,40]

        Defines the maximum amount of movement (in *pixels*) that
        the motion correction should consider.

    nCpu : int, optional
        Default: (Maximum number of CPUs) - 1

        Number of CPUs to be used when estimating model parameters for
        motion correction.

    exportFrames : bool, optional
        Default: True

        Whether to export the motion corrected video.

    Returns
    =======
    motCorrDir : str
        Path of the motion correction directory.
        
    nonAlignedDatasetDir: str
        Path of the non-aligned data directory.
    """
    try:
        # instead of os.makedirs use os.mkdir
        # the former creates also intermediate dirs
        os.mkdir(outDir + '/' + baseName)
    except OSError:
        print('Either directory exists or \
              the intermedite directories do not exist!')
        print("--Skipping %s..." % baseName)
        return 0
    print("--Motion correction started with %s..." % baseName)
    # Checking if there are two cycles of images. Important to know otherwise 
    # you get merged motion corrected images.
    allImages = (glob.glob(rawDataFile))
    tseries_names = [os.path.basename(imageName).split('_')[0] for imageName in allImages]
    uniqueTSeries = Counter(tseries_names).keys()
    tseries_lengths = Counter(tseries_names).values()
                
    if len(uniqueTSeries) > 1:
        print('---More than 1 T-Series found. Aligning them together...')

    match = [int(re.search(r'Cycle(.*?)_',imageName).group(1)) for imageName in allImages]
    uniqueCycles = numpy.unique(numpy.asarray(match))
    
    if len(uniqueCycles) > 1:
        warnstr='More than 1 image cycle detected. Aborting alignment.'
        warnings.warn(warnstr)
        return 

    
    if bleedthrough_correct:
        print("Running bleedthrough elimination.")
        # Ultima recordings have a specific bleedthrough which can be eliminated by a FFT method (adapted from Juan Felipe)
        FFT_2d = calculate_spatial_FFT(allImages)
        filtered_data_FFT = filter_bleedthrough_gaussianstripe(FFT_2d)
        backtransformed = backtransform(filtered_data_FFT)
        # Transform the array to fit sima organization
        organized_array = backtransformed[:,np.newaxis,:,:,np.newaxis]
        organized_array = np.transpose(organized_array,(3,1,0,2,4))
        sequence=[sima.Sequence.create('ndarray', organized_array)]
    else:
        sequence = [sima.Sequence.create('TIFFs', [[rawDataFile]])]


    print("Creating sima dataset of non-aligned images.")
    nonAlignedDatasetDir = outDir + '/' + baseName + '/' + 'TIFFs.sima'
    sima.ImagingDataset(sequence, nonAlignedDatasetDir)

    print("Running motion correction.")
    
    mc_approach = sima.motion.HiddenMarkov2D(granularity=granularity,
                                             max_displacement=maxDisplacement,
                                             verbose=True,
                                             n_processes=nCpu)
    
    motCorrDir = outDir + '/' + baseName + '/' + 'motCorr.sima'
    dataset = mc_approach.correct(sequence, motCorrDir)
    print("Creating sima dataset of aligned images")

    if exportFrames:
        print("Exporting motion-corrected movies.")
        
        for iTSeries, curr_T_series in enumerate(uniqueTSeries):
            
            start_frame = tseries_names.index(curr_T_series)
            end_frame = start_frame+tseries_lengths[iTSeries]
            dataset[0,start_frame:end_frame].export_frames([[[os.path.join(motCorrDir,
                                                                             '{t}_motCorr.tif'.format(t = curr_T_series))]]],
                                  fill_gaps=True)
            

    print("--Motion correction done with %s..." % baseName)

    return uniqueTSeries

    

def extractRawSignal(motCorrDir, ch=0, roiKeyNo=0, extLabel='GC6f',
                     nCpu=(multiprocessing.cpu_count() - 1)):
    """ Extracts the raw signal from the time series.

    Parameters
    ==========
    motCorrDir : str

        Path of the motion correction directory.
    ch : int, optional

        Default: 0
        Index of the channel in sima.ImagingDataset object.
        For single channel recordings, one *should not* change this value.
        Multichannel recordings are not supported yet, but will be implemented
        if needed, which can be done with minor modifications.

    roiKeyNo : int, optional
        Default: 0

        Index of the roi key in dataset.ROIs.keys() which returns a list of
        roi keys. Assuming you have only one roi set per dataset,
        this value should be 0.

    nCpu : int, optional
        Default: (Maximum number of CPUs) - 1

        Number of CPUs to be used for the signal extraction.
        
    extLabel : str, optional

        Default: 'GC6f'
        Text label to describe this extraction, if None defaults to a 
        timestamp.

    Returns
    =======
    signalFile : str
        Path of the file which contains extracted signals.

    chNames : list
        List of all channel names present in the dataset.

    usedChannel : str
        Name of the channel used for signal extraction.

    roiKeys : list
        All the ROI key names saved when selecting ROIs.

    usedRoiKey : str
        ROI key that is used for signal extraction.
        
    usedExtLabel : str
        Extraction label that is used for signal extraction.
    """
    # in case there is only one cpu in total and using the default value
    if nCpu == 0:
        nCpu = 1

    dataset = sima.ImagingDataset.load(motCorrDir)
    roiKeys = dataset.ROIs.keys()
    rois = dataset.ROIs[roiKeys[roiKeyNo]]
    chNames = dataset.channel_names
    # extracted signals are permanently stored in the ImagingDataset object
    print('Extracting signals.')
    dataset.extract(rois, signal_channel=chNames[ch], remove_overlap=True,
                    n_processes=nCpu,label =extLabel)
    # load it again
    # export it to csv file
    # append the channel name into file name
    dataset = sima.ImagingDataset.load(motCorrDir)
    signalFile = os.path.join(motCorrDir, 'no_bg_signals_ch' + str(chNames[ch])
                              + '.csv')
    dataset.export_signals(signalFile, channel=str(ch))

    if len(roiKeys) > 1:
        print('WARNING: More than one ROI key present! Used key:%s' % roiKeyNo)
    if len(chNames) > 1:
        print('WARNING: More than one channel name present!')

    print('Roi Keys in the input file ' + str(roiKeys))
    print('Channels in the input file ' + str(chNames))
    print('Extraction label: %s' % extLabel)

    usedChannel = chNames[ch]
    usedRoiKey = roiKeys[roiKeyNo]
    usedExtLabel = extLabel
    return signalFile, chNames, usedChannel, roiKeys, usedRoiKey, usedExtLabel


def autoSegmentROIs(dataDir, strategy, channel = 0):
    """ Does segmentation of time series.

    Parameters
    ==========
   dataDir : str
        Path of the motion correction directory.
        
   strategy : str
        Strategy of the segmentation
        
   channel : int, optional
       Default: 0
       The index of the channel to be used.     

    Returns
    =======
    outputFile : str
        Path of the ROI list.
    """
    # Load the dataset which is located in the motCorr.sima
    dataset = sima.ImagingDataset.load(dataDir)
        
    # Determine the segmentation approach
    if strategy == 'STICA':
        segmentation_approach = sima.segment.STICA(channel = channel,components=20)
        segmentation_approach.append(sima.segment.SparseROIsFromMasks(min_size=20))
        segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.85))
        segmentation_approach.append(sima.segment.SmoothROIBoundaries())
        size_filter = sima.segment.ROIFilter(lambda roi: roi.size >= 20 and roi.size <= 50)
        segmentation_approach.append(size_filter)
        
        
    elif strategy == 'BAM':
        segmentation_approach = sima.segment.BasicAffinityMatrix(channel=channel, max_dist=(3,3), 
                        spatial_decay=(4,4), num_pcs=75, verbose=True)
            
    
    
     
    
#    segmentation_approach.append(sima.segment.MergeOverlapping(threshold=0.5))
    
   
    
    # returns a sima.ROI.ROIList
    outputFile = os.path.join(dataDir,'ROIs')
    rois = dataset.segment(segmentation_approach, 'auto_ROIs')
    rois.save(outputFile)
    print('Saved segmentation results...')
    return outputFile

def getROIinformation(signalFile, bgLabel=['bg', 0]):
    """Get the background ROI index. Also get the ROI numbers and ROI tags. 
    ROI tags correspond to the ROI types. e.g. Layer1, MedullaDendrite ...

    Parameters
    ==========
    signalFile : str
        Path of the file which contains extracted signals.

    bgLabel : list, optional
        Default: ['bg', 0]

        Label of the background ROI. In Roibuddy, if the background ROI is
        selected last, then it will a label 0. That is why 0 is among the
        default values. One can establish his/her own way to do it,
        e.g. labelling it as 'bg' (which is also in the default values).
        The parameter should contain two values, which can be either a string
        or an integer, and the second term has higher priority than the first
        one.

    Returns
    =======
    header : list
        It is a list of ROI labels, in the order it appears in the extracted
        signal file. Labels are numbers of single ROIs.

    bgIndex : int
        Index of the background in the header.
    
    tags : list 
        It is a list of ROI tags, in the order it appears in the extracted
        signal file. Tags are the types of ROIs. e.g. Layer1 Layer2...
    """
    # bg should have one of these labels
    # use islice so that only a part of the file is loaded
    # first line of the header is useless
    # in the second line, first two terms are useless
    # clean the header
    # leave only labels
    header = []
    tags = []
    with open(signalFile, 'r') as infile:
        lines_gen = islice(infile, 2)
        lines_genT = islice(infile, 1)
        for lines in lines_gen:
            lines = re.sub('\n', '', lines)
            lines = re.sub('\r', '', lines)
            lines = re.sub('', '', lines)
            lines = lines.split('\t')
            header.append(lines)
        for linesT in lines_genT:
            linesT = re.sub('\n', '', linesT)
            linesT = re.sub('\r', '', linesT)
            linesT = re.sub('', '', linesT)
            linesT = linesT.split('\t')
            tags.append(linesT)
    header = header[1][2:]  # list of ROIs
    tags = tags[0][2:]
    # pay attention that returned header does not contain the first two columns
    # get the index of the background intensity column
    # background ROI should be labelled as 'bg' or '0'
    try:
        bgIndex = header.index(str(bgLabel[1]))
    except ValueError:
        try:
            bgIndex = header.index(bgLabel[0])
        except ValueError:
            bgIndex = header.index(bgLabel[1])
    # pay attention that bgIndex is actaully pythonic counting

    return header, bgIndex, tags


def subtractBg(signalFile, bgIndex, skipHeader=3):
    """Subtracts the average intensity of the background ROI from all the ROIs
    of the corresponding imaging frame.

    Parameters
    ==========
    signalFile : str
        Path of the file which contains extracted signals.

    bgIndex : int
        Index of the background in the header.

    skipHeader : int, optional
        Default: 3

        Number of lines to be skipped from the beginning of the extracted
        signal file.

    Returns
    =======
    bgSub : ndarray
        Contains the background-subracted intensities for all ROIs,
        including the background ROI.

    rawTrace : ndarray
        Contains the raw intensities for all ROIs, including the background
        ROI.
    """
    # read the intensity values
    # skip the header so that array type can be float
    # each row is a list of roi intensities
    # substract the background columnwise
    # skip the first 3 lines by default
    # otherwise array type would not be float
    # skip the first two columns
    # first two columns are 'sequence' and 'frame'
    # discard them
    data = numpy.genfromtxt(signalFile, dtype='float', skip_header=skipHeader)
    # each column in rawTrace is an roi, inc. bg
    # each row is an img frame
    rawTrace = data[:, 2:len(data[0])]
    bgSub = rawTrace - rawTrace[:, bgIndex, None]

    return bgSub, rawTrace


def readStimOut(stimOutFile, skipHeader=1):
    """Read and get the stimulus output data.

    Parameters
    ==========
    stimOutFile : str
        Stimulus output file path.

    skipHeader : int, optional
        Default: 1

        Number of lines to be skipped from the beginning of the stimulus
        output file.

    Returns
    =======
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file.

    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.
    """
    # skip the first line since it is a file path
    rawStimData = numpy.genfromtxt(stimOutFile, dtype='float',
                                   skip_header=skipHeader)
    # also get the file path
    # do not mix it with numpy
    # only load and read the first line
    stimType = "stimType"
    with open(stimOutFile, 'r') as infile:
        lines_gen = islice(infile, 1)
        for line in lines_gen:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            line = re.sub(' ', '', line)
            stimType = line
            break

    return stimType, rawStimData

def readStimInput(stimType='', stimInputDir='', gui=True,
                  initialDirectory='~'): #NOT USED by BURAK
    """
    Parameters
    ==========
    stimType : str, optional
        Default: ''
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str, optional
        Default: ''

        Path of the directory to look for the stimulus input.

    gui : bool, optional
        Default: False

        Whether to use GUI for directory selection or not.

    initialDirectory : str, optional
        Default: HOME directory, i.e. '~'.

        Path into which the directory selection GUI opens.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).

    Notes
    =====
    This function does not return the values below anymore:

    - epochDur : list
        Duration of the epochs as in the stimulus input file.

    - isRandom : int

    - stimTransAmp : list

    """

    if stimInputDir == '' and not gui:
        print("Give a value to stimInputDir XOR gui")
        return None, None, None, None, None

    elif stimInputDir != '' and gui:
        print("Cannot give a non-default value to both gui and stimInputDir"
              "at the time ")
        return None, None, None, None, None

    else:
        if stimInputDir == '' and gui:
            initialDirectory = makePath(initialDirectory)
            root = Tkinter.Tk()
            root.withdraw()

            stimInputFile = askopenfilename(parent=root,
                                            initialdir=initialDirectory,
                                            filetypes=[('Text', '*.txt')],
                                            title='Open stimInputFile')

        elif stimInputDir != '' and not gui:
            stimInputDir = makePath(stimInputDir)
            stimType = stimType.split('\\')[-1]
            stimInputFile = glob.glob(os.path.join(stimInputDir, stimType))[0]
        # @TODO: Recognize if there is a slash or not after stimInputDir

        flHandle = open(stimInputFile, 'r')
        stimInputData = {}
        for line in flHandle:
            line = re.sub('\n', '', line)
            line = re.sub('\r', '', line)
            line = line.split('\t')
            stimInputData[line[0]] = line[1:]

#        epochDur = stimInputData['Stimulus.duration']
#        isRandom = int(stimInputData['Stimulus.randomize'][0])
#        stimTransAmp = stimInputData['Stimulus.stimtrans.amp']
#
#        epochDur = [float(sec) for sec in epochDur]
#        stimTransAmp = [float(deg) for deg in stimTransAmp]

        return stimInputFile, stimInputData


def getEpochCount(rawStimData, epochColumn=3):
    """Get the total epoch number.

    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    Returns
    =======
    epochCount : int
        Total number of epochs.
    """
    # get the max epoch count from the rawStimData
    # 4th column is the epoch number
    # add plus 1 since the min epoch no is zero
    
    # BG edit: Changed the previous epoch extraction, which uses the maximum 
    # number + 1 as the epoch number, to a one finding the unique values and 
    # taking the length of it
    epochCount = np.shape(np.unique(rawStimData[:, epochColumn]))[0]
    print("Number of epochs = " + str(epochCount))

    return epochCount


def readStimInformation(stimType, stimInputDir):
    """
    Parameters
    ==========
    stimType : str
        Path of the executed stimulus file as it appears in the header of the
        stimulus output file. *Required* if `gui` is FALSE and `stimInputDir`
        is given a non-default value.

    stimInputDir : str
        Path of the directory to look for the stimulus input.

    Returns
    =======
    stimInputFile : str
        Path to the stimulus input file (which contains stimulus parameters,
        not the output file).

    stimInputData : dict
        Lines of the stimulus generator file. Keys are the first terms of the
        line (the parameter names), values of the key is the rest of the line
        (a list).
    
    isRandom : int
        A that changes how epochs are randomized 

    Notes
    =====
    This function does not return the values below anymore:

    - epochDur : list
        Duration of the epochs as in the stimulus input file.


    - stimTransAmp : list

    """

    stimType = stimType.split('\\')[-1]
    stimInputFile = glob.glob(os.path.join(stimInputDir, stimType))[0]

    flHandle = open(stimInputFile, 'r')
    stimInputData = {}
    for line in flHandle:
        line = re.sub('\n', '', line)
        line = re.sub('\r', '', line)
        line = line.split('\t')
        stimInputData[line[0]] = line[1:]

#        epochDur = stimInputData['Stimulus.duration']
    
#        stimTransAmp = stimInputData['Stimulus.stimtrans.amp']
#
#        epochDur = [float(sec) for sec in epochDur]
#        stimTransAmp = [float(deg) for deg in stimTransAmp]

    return stimInputFile, stimInputData

def divideEpochs(rawStimData, epochCount, isRandom, framePeriod,
                 trialDiff=0.20, overlappingFrames=0, firstEpochIdx=0,
                 epochColumn=3, imgFrameColumn=7, incNextEpoch=True,
                 checkLastTrialLen=True):
    """
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    isRandom : int

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    overlappingFrames : int

    firstEpochIdx : int
        Default: 0

        Index of the first epoch.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    incNextEpoch :
    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [Z, D]] where first term is the trial beginning
        (first of first) and end (second of first), and second term is the
        baseline start (first of second) and end (second of second) for that
        trial.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).

    isRandom :
    """
    trialDiff = float(trialDiff)
    firstEpochIdx = int(firstEpochIdx)
    overlappingFrames = int(overlappingFrames)
    trialCoor = {}
    fullEpochSeq = []

    if isRandom == 0:
        fullEpochSeq = range(epochCount)
        # if the key is zero, that means isRandom is 0
        # this is for compatibitibility and
        # to make unified workflow with isRandom == 1
        trialCoor[0] = []

    elif isRandom == 1:
        # in this case fullEpochSeq is just a list of dummy values
        # important thing is its length
        # it's set to 3 since trials will be sth like: 0, X
        # if incNextEpoch is True, then it will be like : 0,X,0
        fullEpochSeq = range(2)
        for epoch in range(1, epochCount):
            # add epoch numbers to the dictionary
            # do not add the first epoch there
            # since it is not the exp epoch
            # instead it is used for baseline and inc coordinates
            trialCoor[epoch] = []

    if incNextEpoch:
        # add the first epoch
        fullEpochSeq.append(firstEpochIdx)
    elif not incNextEpoch:
        pass

    # min and max img frame numbers for each and every trial
    # first terms in frameBaselineCoor are the trial beginning and end
    # second terms are the baseline start and end for that trial
    currentEpochSeq = []
    frameBaselineCoor = [[0, 0], [0, 0]]
    nextMin = 0
    baselineMax = 0

    for line in rawStimData:
        if (len(currentEpochSeq) == 0 and
                len(currentEpochSeq) < len(fullEpochSeq)):
            # it means it is the very beginning of a trial block.
            # in the very first trial,
            # min frame coordinate cannot be set by nextMin.
            # this condition satisfies this purpose.
            currentEpochSeq.append(int(line[epochColumn]))
            if frameBaselineCoor[0][0] == 0:
                frameBaselineCoor[0][0] = int(line[imgFrameColumn])
                frameBaselineCoor[1][0] = int(line[imgFrameColumn])

        elif (len(currentEpochSeq) != 0 and
              len(currentEpochSeq) < len(fullEpochSeq)):
            # only update the current epoch list
            # already got the min coordinate of the trial
            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for the very first trial
                    frameBaselineCoor[1][1] = (int(line[imgFrameColumn])
                                               - overlappingFrames)

        elif len(currentEpochSeq) == len(fullEpochSeq):
            if nextMin == 0:
                nextMin = int(line[imgFrameColumn]) + overlappingFrames

            if int(line[epochColumn]) != currentEpochSeq[-1]:
                currentEpochSeq.append(int(line[epochColumn]))

            elif int(line[epochColumn]) == currentEpochSeq[-1]:
                if int(line[epochColumn]) == 0 and currentEpochSeq[-1] == 0:
                    # set the maximum endpoint of the baseline
                    # for all the trials except the very first trial
                    baselineMax = int(line[imgFrameColumn]) - overlappingFrames

        else:
            frameBaselineCoor[0][1] = (int(line[imgFrameColumn])
                                       - overlappingFrames)

            if frameBaselineCoor[0][1] > 0:
                if isRandom == 0:
                    # if the key is zero, that means isRandom is 0
                    # this is for compatibitibility and
                    # to make unified workflow isRandom == 1
                    trialCoor[0].append(frameBaselineCoor)
                elif isRandom == 1:
                    # get the epoch number
                    # epoch no should be the 2nd term in currentEpochSeq
                    expEpoch = currentEpochSeq[1]
                    trialCoor[expEpoch].append(frameBaselineCoor)
            # this is just a safety check
            # towards the end of the file, the number of epochs might
            # not be enough to form a trial block
            # so if the max img frame coordinate is still 0
            # it means this not-so-complete trial will be discarded
            # only complete trials are appended to trial coordinates
            # if it has a max frame coord, it is safe to say
            # it had nextMin in frameBaselineCoor
            # print(currentEpochSeq)
            currentEpochSeq = []
            currentEpochSeq.append(firstEpochIdx)
            # each time currentEpochSeq resets means that
            # one trial block is complete
            # adding firstEpochIdx is necessary
            # otherwise currentEpochSeq will shift by 1
            # after every trial cycle
            # now that the frame coordinates are stored
            # can reset it
            # and add min coordinate for the next trial
            # then add the max baseline coordinate for the next trial
            frameBaselineCoor = [[0, 0], [0, 0]]
            frameBaselineCoor[0][0] = nextMin
            frameBaselineCoor[1][0] = nextMin
            frameBaselineCoor[1][1] = baselineMax
            nextMin = 0
            baselineMax = 0

    # @TODO: no need to separate isRandoms, make a unified for loop
    if checkLastTrialLen:
        if isRandom == 0:
            lenFirstTrial = trialCoor[0][0][0][1] - trialCoor[0][0][0][0]
            lenLastTrial = trialCoor[0][-1][0][1] - trialCoor[0][-1][0][0]
            if ((lenFirstTrial - lenLastTrial) * framePeriod) >= trialDiff:
                trialCoor[0].pop(-1)
                print("Last trial is discarded since the length was too short")

        elif isRandom == 1:
            for epoch in trialCoor:
                delSwitch = False
                lenFirstTrial = (trialCoor[epoch][0][0][1]
                                 - trialCoor[epoch][0][0][0])
                lenLastTrial = (trialCoor[epoch][-1][0][1]
                                - trialCoor[epoch][-1][0][0])

                if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                    delSwitch = True

                if delSwitch:
                    print("Last trial of epoch " + str(epoch)
                          + " is discarded since the length was too short")
                    trialCoor[epoch].pop(-1)

    trialCount = []
    if isRandom == 0:
        # there is only a single key in the trialCoor dict in this case
        trialCount.append(len(trialCoor[0]))
    elif isRandom == 1:
        epochTrial = []
        for epoch in trialCoor:
            epochTrial.append(len(trialCoor[epoch]))
        # in this case first element in trialCount is min no of trials
        # second element is the max no of trials
        trialCount.append(min(epochTrial))
        trialCount.append(max(epochTrial))

    return trialCoor, trialCount, isRandom

def divide_all_epochs(rawStimData, epochCount, framePeriod, trialDiff=0.20,
                      epochColumn=3, imgFrameColumn=7,checkLastTrialLen=True):
    """
    
    Finds all trial and epoch beginning and end frames
    
    Parameters
    ==========
    rawStimData : ndarray
        Numeric data of the stimulus output file, e.g. stimulus frame number,
        imaging frame number, epoch number... Rows and columns are organized
        in the same fashion as they appear in the stimulus output file.

    epochCount : int
        Total number of epochs.

    framePeriod : float
        Time it takes to image a single frame.

    trialDiff : float
        Default: 0.20

        A safety measure to prevent last trial of an epoch being shorter than
        the intended trial duration, which can arise if the number of frames
        was miscalculated for the t-series while imaging. *Effective if and
        only if checkLastTrialLen is True*. The value is used in this way
        (see the corresponding line in the code):

        *(lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff*

        If the case above is satisfied, then this last trial is not taken into
        account for further calculations.

    epochColumn : int, optional
        Default: 3

        The index of epoch column in the stimulus output file
        (start counting from 0).

    imgFrameColumn : int, optional
        Default: 7

        The index of imaging frame column in the stimulus output file
        (start counting from 0).

    checkLastTrialLen :

    Returns
    =======
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list. Each term
        in this list is a trial of the epoch. These terms have the following
        structure: [[X, Y], [X, Y]] where X is the trial beginning and Y 
        is the trial end.

    trialCount : list
        Min (first term in the list) and Max (second term in the list) number
        of trials. Ideally, they are equal, but if the last trial is somehow
        discarded, e.g. because it ran for a shorter time period, min will be
        (max-1).
    """
    trialDiff = float(trialDiff)
    trialCoor = {}
    
    for epoch in range(0, epochCount):
        
        trialCoor[epoch] = []

    previous_epoch = []
    for line in rawStimData:
        
        current_epoch = int(line[epochColumn])
        
        if (not(previous_epoch == current_epoch )): # Beginning of a new epoch trial
            
            
            if (not(previous_epoch==[])): # If this is after stim start (which is normal case)
                epoch_trial_end_frame = previous_frame
                trialCoor[previous_epoch].append([[epoch_trial_start_frame, epoch_trial_end_frame], 
                                            [epoch_trial_start_frame, epoch_trial_end_frame]])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                previous_epoch = int(line[epochColumn])
                
            else:
                previous_epoch = int(line[epochColumn])
                epoch_trial_start_frame = int(line[imgFrameColumn])
                
        previous_frame = int(line[imgFrameColumn])
        
    if checkLastTrialLen:
        for epoch in trialCoor:
            delSwitch = False
            lenFirstTrial = (trialCoor[epoch][0][0][1]
                             - trialCoor[epoch][0][0][0])
            lenLastTrial = (trialCoor[epoch][-1][0][1]
                            - trialCoor[epoch][-1][0][0])
    
            if (lenFirstTrial - lenLastTrial) * framePeriod >= trialDiff:
                delSwitch = True
    
            if delSwitch:
                print("Last trial of epoch " + str(epoch)
                      + " is discarded since the length was too short")
                trialCoor[epoch].pop(-1)
                
    trialCount = []
    epochTrial = []
    for epoch in trialCoor:
        epochTrial.append(len(trialCoor[epoch]))
    # in this case first element in trialCount is min no of trials
    # second element is the max no of trials
    trialCount.append(min(epochTrial))
    trialCount.append(max(epochTrial))
       

    return trialCoor, trialCount

def dff(trialCoor, header, bgIndex, bgSub, baselineEpochPresent, baseDur):
    """ Calculate df/f. Modified by burak to handle stimuli without a baseline
    epoch. This function calculates dF/F for each epoch and each trial within 
    the epoch seperately.

    Parameters
    ==========
    trialCoor : dict
        Each key is an epoch number. Corresponding value is a list.
        Each term in this list is a trial of the epoch.
        These terms have the following str: [[X, Y], [Z, D]] where
        first term is the trial beginning (first of first) and end
        (second of first), and second term is the baseline start
        (first of second) and end (second of second) for that trial.

    header : list
        It is a list of ROI labels, in the order it appears in the extracted
        signal file.

    bgIndex : int
        Index of the background in the header.

    bgSub : ndarray
        Contains the background-subracted intensities for all ROIs, including
        the background ROI.
        
    baselineEpochPresent: bool
        If the baseline epoch present or not. If yes, this epoch will be used
        for dF/F calculations.
    
    baseDur : int
        Frame numbers before the actual epoch to take as baseline

    Returns
    =======
    dffTraceAllRoi : dict
        df/f trace of all the ROIs. Each key is an epoch number. Corresponding
        value is a list of lists. Every element in the outer list (i.e. the
        inner list elements) corresponds to ROIs, in the same order as in the
        header. Every 'ROI list' is a list of numpy arrays, where each array
        correponds to a trial.. Note that the background ROI has NaN values
        for every trial.

    baselineStdAllRoi : dict
        Standard deviation of the baseline trace. Each key is an epoch number.
        Corresponding value is a list of lists. Every element in the outer
        list (i.e. the inner list elements) corresponds to ROIs, in the same
        order as in the header. 'ROI lists' have floats which correpond to
        standard deviation of each trial.

    baselineMeanAllRoi : dict
        Mean of the baseline trace. Each key is an epoch number.
        Corresponding value is a list of lists. Every element in the outer
        list (i.e. the inner list elements) corresponds to ROIs, in the same
        order as in the header. 'ROI lists' have floats which correpond to
        mean of each trial.
    """

    dffTraceAllRoi = {}
    baselineStdAllRoi = {}
    baselineMeanAllRoi = {}
    # in the first iteration, each element in dffTraceAllRoi,
    # is an roi; in each roi, single elements are trials
    # IMPORTANT NOTE: Calculations are fast numpy operations,
    # but eventually calculated values values go into a list,
    # which is then converted to a numpy object-not an array.
    # the reason is that not all trial blocks have the same length.
    # right now it benefits from fast numpy calc, but in the future
    # you might want to do it more elegantly.
    # bg roi is filled by NaNs, and there is only one NaN per trial
    for epochNo in trialCoor:
        epochNo = int(epochNo)  # just to be safe
        dffTraceAllRoi[epochNo] = []  # init an empty list with epochNo key
        baselineStdAllRoi[epochNo] = []
        baselineMeanAllRoi[epochNo] = []
        for roiIdx in range(len(header)):
            for trial in trialCoor[epochNo]:
                # take this many rows of a particular column
                # -1 is bcs img frame indices start from 1 in trialCoor
                # run away from the division by zero problem for background
                normTrace = numpy.float(0)
                if roiIdx == bgIndex:
                    normTrace = numpy.nan
                    normBaseline = numpy.nan
                    normBaselineStd = numpy.nan
                else:
                    if baselineEpochPresent: # Take the baseline as F0
                        trace = bgSub[trial[0][0]-1:trial[0][1], roiIdx]
                        baseTrace = bgSub[trial[1][0]-1:trial[1][1], roiIdx]
                        baseline = numpy.average(baseTrace[-baseDur:])
                        normTrace = (trace - baseline) / baseline
                        normBaseTrace = (baseTrace - baseline) / baseline
                        # calculate baseline stdev
                        # might need for thresholding in the future
                        normBaseline = numpy.average(normBaseTrace)
                        normBaselineStd = numpy.std(normBaseTrace)
                    else: # Taking the mean of all trial as F0
                        trace = bgSub[trial[0][0]-1:trial[0][1], roiIdx]
                        baseline = numpy.average(trace)
                        normTrace = (trace - baseline) / baseline
                        normBaseline = numpy.nan # Returns NaN
                        normBaselineStd = numpy.nan # Returns NaN
                try:
                    dffTraceAllRoi[epochNo][roiIdx].append(normTrace)
                    baselineStdAllRoi[epochNo][roiIdx].append(normBaselineStd)
                    baselineMeanAllRoi[epochNo][roiIdx].append(normBaseline)
                except IndexError:
                    dffTraceAllRoi[epochNo].append([normTrace])
                    baselineStdAllRoi[epochNo].append([normBaselineStd])
                    baselineMeanAllRoi[epochNo].append([normBaseline])

    return dffTraceAllRoi, baselineStdAllRoi, baselineMeanAllRoi
def trialAverage(dffTraceAllRoi, bgIndex):
    """ Take the average of df/f traces across trials.

    Paremeters
    ==========
    dffTraceAllRoi : dict
        df/f trace of all the ROIs. Each key is an epoch number. Corresponding
        value is a list of lists. Every element in the outer list (i.e. the
        inner list elements) corresponds to ROIs, in the same order as in the
        header. Every 'ROI list' is a list of numpy arrays, where each array
        correponds to a trial.. Note that the background ROI has NaN values
        for every trial.

    bgIndex : int
        Index of the background in the header.

    Returns
    =======
    trialAvgAllRoi : dict
        Average of df/f traces across trials. Each key is an epoch number.
        Corresponding value is a list of numpy arrays. Each numpy array is an
        ROI and they are ordered in the same way as in the header. Background
        ROI has a single NaN value.
    """
    # each element in trialAvgAllRoi is an epoch
    # then each epoch has a single list
    # this list contains arrays of trial averages for every roi
    # if bg, instead of an array, it has NaN
    trialAvgAllRoi = {}
    for epoch in dffTraceAllRoi:
        trialAvgAllRoi[epoch] = []
        for roi in range(len(dffTraceAllRoi[epoch])):
            trialLengths = []
            # if Bg, append NaN to trialLengths
            # this way you dont distrupt bgIndex in the future
            if roi == bgIndex:
                trialLengths.append(numpy.nan)
            else:
                for trial in dffTraceAllRoi[epoch][roi]:
                    trialLengths.append(len(trial))

            if trialLengths[0] is not numpy.nan:
                # real ROI case, not bg
                minTrialLen = min(trialLengths)
                trialFit = 0
                for trial in dffTraceAllRoi[epoch][roi]:
                    trialFit += trial[:minTrialLen]
                # calculate the trial average for an roi
                trialAvg = trialFit/len(dffTraceAllRoi[epoch][roi])

                trialAvgAllRoi[epoch].append(trialAvg)

            elif trialLengths[0] is numpy.nan:
                # bgRoi case
                trialAvgAllRoi[epoch].append(numpy.nan)

    return trialAvgAllRoi
def corrTrialAvg(trialAvgAllRoi, epochDur, bgIndex, framePeriod):
    """Calculate the Pearson correlation coefficient and p-value for trial
    averaged signal of every ROI.

    Parameters
    ==========
    trialAvgAllRoi : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which consists of numpy arrays and each array is the trial
        averaged trace of an ROI. ROIs are ordered in the same way as in the
        header, e.g. the first term in trialAvgAllRoi corrsponds to the first
        ROI label in the header.

    epochDur : list
        Duration of the epochs as in the stimulus input file.

    bgIndex : int
        Index of the background in the header.

    framePeriod : float
        Time it takes to image a single frame.

    Returns
    =======
    corrHeader : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which contains Pearson correlation coefficients for every ROI.
        ROIs are ordered in the same way as in the header, e.g. the first term
        in corrHeader corrsponds to the first ROI label in the header.

    pvalHeader : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which contains p-values of Pearson correlation for every ROI.
        ROIs are ordered in the same way as in the header, e.g. the first term
        in pvalHeader corrsponds to the first ROI label in the header.
    """
    corrHeader = {}
    pvalHeader = {}
    for epochNo in trialAvgAllRoi:
        epochNo = int(epochNo)  # safety measure
        corrHeader[epochNo] = []
        pvalHeader[epochNo] = []
        epochFrames = [int(sec / framePeriod) for sec in epochDur]
        corrSignal = ([0.0] * epochFrames[0] + [1.0] * epochFrames[epochNo]
                      + [0.0] * epochFrames[0])

        for roi in range(len(trialAvgAllRoi[epochNo])):
            if roi == bgIndex:
                corrHeader[epochNo].append(numpy.nan)
                pvalHeader[epochNo].append(numpy.nan)
            else:
                shortLen = min(len(corrSignal),
                               len(trialAvgAllRoi[epochNo][roi]))
                coeff, pval = pearsonr(corrSignal[:shortLen],
                                       trialAvgAllRoi[epochNo][roi][:shortLen])
                corrHeader[epochNo].append(coeff)
                pvalHeader[epochNo].append(pval)

    return corrHeader, pvalHeader

def roiAverage(trialAvgAllRoi, bgIndex, corrHeader, pvalHeader,
               corrCutOff=0.35, pvalCutOff=0.01, singleCorrEpoch=False):
    """ Filters the ROIs based on a correlation threshold and takes their
    average.

    Parameters
    ==========
    trialAvgAllRoi : dict
        Average of df/f traces across trials. Each key is an epoch number.
        Corresponding value is a list of numpy arrays. Each numpy array is an
        ROI and they are ordered in the same way as in the header. Background
        ROI has a single NaN value.

    bgIndex : int
        Index of the background in the header.

    corrHeader : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which contains Pearson correlation coefficients for every ROI.
        ROIs are ordered in the same way as in the header, e.g. the first term
        in corrHeader corrsponds to the first ROI label in the header.

    pvalHeader : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which contains p-values of Pearson correlation for every ROI.
        ROIs are ordered in the same way as in the header, e.g. the first term
        in pvalHeader corrsponds to the first ROI label in the header.

    corrCutOff : float, optional
        Default: 0.35

        Correlation coefficient value that is used to filter ROIs.
        Example: If corrCutOff=0.5 this means that any ROI having the value
        in the interval [-0.5, 0.5] is not taken into account for averaging.

    pvalCutOff : float, optional
        Default: 0.01

        P-value that is used to filter ROIs. Example: If pvalCutOff=0.01 this
        means that any ROI having the value greater than 0.01 is not taken
        into account for averaging.

    singleCorrEpoch : bool, optional
        Default: False
    """

    pvalCutOff = float(pvalCutOff)
    corrCutOff = abs(float(corrCutOff))

    roiAverageNoCorr = {}
    roiStdNoCorr = {}
    # epoch key has a single array, avg of all rois
    roiAverageCorr = {}
    roiStdCorr = {}
    roiStdNoCorr = {}
    roiCountCorr = {}
    roiIdxCorr = {}
    # epoch key has a list, which has two arrays,(+) and (-) polarity

    for epoch in trialAvgAllRoi:
        roiLengths = []

        for roi in range(len(trialAvgAllRoi[epoch])):
            if roi == bgIndex:
                pass
            else:
                roiLengths.append(len(trialAvgAllRoi[epoch][roi]))
        # probably all of them are equal already
        # there should not be +-1 frames, but just in case
        # not to run into broadcasting errors in the future steps
        # @TODO first try averaging by broadcasting and
        # if it does not work use this method as a fallback
        minLen = min(roiLengths)
        polarRoiList = [[], [], []]
        roiCount = [0, 0, 0]  # order: (+),(-) and discarded roi indexes
        roiIdx = [[], [], []]  # order: (+),(-) and discarded roi indexes
        # discarded ROIs do not include bg index

        for roi in range(len(trialAvgAllRoi[epoch])):
            if roi == bgIndex:
                pass

            # below is used if an external corrHeader provided and
            # it contains only a single epoch
            # example case: you have aligned ROIs from a stimulus
            # with multiple epochs to Full Field Flashes
            # but Full field flashes contain a single epoch
            # (not necessarily in the stim file, but in the code)

            # @TODO: FIX THIS RECURRENT BLOCK below
            else:
                avgSlice = trialAvgAllRoi[epoch][roi][:minLen]
                if not singleCorrEpoch:
                    if (corrHeader[epoch][roi] > corrCutOff and
                            pvalHeader[epoch][roi] < pvalCutOff):
                        roiCount[0] += 1
                        roiIdx[0].append(roi)
                        polarRoiList[0].append(avgSlice)
                    elif (corrHeader[epoch][roi] < -corrCutOff and
                          pvalHeader[epoch][roi] < pvalCutOff):
                        roiCount[1] += 1
                        roiIdx[1].append(roi)
                        polarRoiList[1].append(avgSlice)
                    else:
                        roiCount[2] += 1
                        roiIdx[2].append(roi)

                elif singleCorrEpoch:
                    if (corrHeader[0][roi] > corrCutOff and
                            pvalHeader[0][roi] < pvalCutOff):
                        roiCount[0] += 1
                        roiIdx[0].append(roi)
                        polarRoiList[0].append(avgSlice)
                    elif (corrHeader[0][roi] < -corrCutOff and
                          pvalHeader[0][roi] < pvalCutOff):
                        roiCount[1] += 1
                        roiIdx[1].append(roi)
                        polarRoiList[1].append(avgSlice)
                    else:
                        roiCount[2] += 1
                        roiIdx[2].append(roi)
                polarRoiList[2].append(avgSlice)

        posPolarRoiAvg = numpy.average(polarRoiList[0], axis=0)
        posPolarRoiStd = numpy.std(polarRoiList[0], axis=0)
        negPolarRoiAvg = numpy.average(polarRoiList[1], axis=0)
        negPolarRoiStd = numpy.std(polarRoiList[1], axis=0)
        allRoiAvg = numpy.average(polarRoiList[2], axis=0)
        allRoiStd = numpy.std(polarRoiList[2], axis=0)

        roiAverageNoCorr[epoch] = allRoiAvg
        roiStdNoCorr[epoch] = allRoiStd
        roiAverageCorr[epoch] = [posPolarRoiAvg, negPolarRoiAvg]
        roiStdCorr[epoch] = [posPolarRoiStd, negPolarRoiStd]

        roiCountCorr[epoch] = roiCount
        roiIdxCorr[epoch] = roiIdx

    return (roiAverageNoCorr, roiStdNoCorr, roiAverageCorr, roiStdCorr,
            roiCountCorr, roiIdxCorr, corrCutOff, pvalCutOff)
    
def interpolateTrialAvgROIs(trialAvgAllRoi, framePeriod, intRate):
    """ Interpolates the responses of ROIs that are trial averaged and sorted
    into epochs. 

    Parameters
    ==========
    trialAvgAllRoi : dict
        Keys in the dictionary are epoch numbers. Corresponding value is a
        list, which consists of numpy arrays and each array is the trial
        averaged trace of an ROI. ROIs are ordered in the same way as in the
        header, e.g. the first term in trialAvgAllRoi corrsponds to the first
        ROI label in the header.

    framePeriod : float
        Time it takes to image a single frame.

    intRate : int
        Interpolation rate in Hz.

    Returns
    =======
    interpolatedAllRoi : dict
        Same format as trialAvgAllRoi. Contains the interpolated arrays to the 
        desired frequency.

    """
    
    interpolatedAllRoi ={}
    for index, epoch in enumerate(trialAvgAllRoi): # For all epochs
        epochResponses = trialAvgAllRoi[epoch]
        interpolatedAllRoi[epoch] = []
        timeV = np.linspace(0,len(epochResponses[0]),len(epochResponses[0]))
        # Create an interpolated time vector in the desired interpolation rate
        timeVI = np.linspace(0,len(epochResponses[0]),
                             (len(epochResponses[0])*framePeriod*intRate+1))
        for iROI in range(len(epochResponses)): # For all ROIs
            currROIResponse = epochResponses[iROI]
            if currROIResponse is not np.nan:
                newCurrResponses = np.interp(timeVI, timeV, currROIResponse)
                interpolatedAllRoi[epoch].append(newCurrResponses)
            else:
                interpolatedAllRoi[epoch].append(np.nan)
    return interpolatedAllRoi

def makePath(path):
    """Make Windows and POSIX compatible absolute paths automatically.

    Parameters
    ==========
    path : str

    Path to be converted into Windows or POSIX style path.

    Returns
    =======
    compatPath : str
    """

    compatPath = os.path.abspath(os.path.expanduser(path))

    return compatPath

def getVarNames(varFile='variablesToSave.txt'):
    """ Read the variable names from a plain-text document. Then it is used to
    save and load the variables by conserving the variable names, in other
    functions. Whenever a new function is added, one should also add the stuff
    it returns (assuming returned values are stored in the same variable names
    as in the function definition) to the varFile.

    Parameters
    ==========
    varFile : str, optional
        Default: 'variablesToSave.txt'

        Plain-text file from where variable names are read.

    Returns
    =======
    varNames : list
        List of variable names
    """
    # get the variable names
    varFile = makePath(varFile)
    workspaceVar = open(varFile, 'r')
    varNames = []

    for line in workspaceVar:
        if line.startswith('#'):
            pass
        else:
            line = re.sub('\n', '', line)
            line = re.sub('', '', line)
            line = re.sub(' ', '', line)
            if line == '':
                pass
            else:
                varNames.append(line)
    workspaceVar.close()

    return varNames

def saveWorkspace(outDir, baseName, varDict, varFile='workspaceVar.txt',
                  extension='.pickle'):
    """ Save the variables that are present in the varFile. The file format is
    Pickle, which is a mainstream python format.

    Parameters
    ==========
    outDir : str
        Output diectory path.

    baseName : str
        Name of the time series folder.

    varDict : dict

    varFile : str, optional
        Default: 'workspaceVar.txt'

        Plain-text file from where variable names are read.

    extension : str, optional
        Default: '.pickle'

        Extension of the file to be saved.

    Returns
    =======
    savePath : str
        Path (inc. the filename) where the analysis output is saved.
    """

    # it is safer to get the variables from a txt
    # otherwise the actual session might have some variables
    # @TODO make workspaceFl path not-hardcoded
    print(varFile)
    varFile = makePath(varFile)
    varNames = getVarNames(varFile=varFile)
    workspaceDict = {}

    for variable in varNames:
        try:
            # only get the wanted var names from globals
            workspaceDict[variable] = varDict[variable]
        except KeyError:
            pass

    # open in binary mode and use highest cPickle protocol
    # negative protocol means highest protocol: faster
    # use cPickle instead of pickle: faster
    # C implementation of pickle
    savePath = os.path.join(outDir, baseName + extension)
    saveVar = open(savePath, "wb")
    cPickle.dump(workspaceDict, saveVar, protocol=-1)
    saveVar.close()

    return savePath

def extractMetadata(varDict={}, batch=False, batchRegEx='', IdName='ID',
                    batchDir='', metaDb='', getDate=False, termSep='_',
                    metaVars='metadataVar.txt'):
    """ Fills in a database of experiments. It reads the variable names from
    *metaVars*. If only the *varDict* argument is given (i.e. not the default
    value), it reads variables with the same name from the memory, and writes
    the values into the database. If varDict is not given (i.e. default value)
    but batch is TRUE, then according to the regular expression given, it will
    find the corresponding analysis files in batchDir, extract the data, and
    save the variables with the same name into the database. Even if the
    metadata database file does not exist, you can still give a name, as it
    will create a new file. If the file already exists, it will append to the
    existing file and will not overwrite it.

    Parameters
    ==========
    varDict : dict, optional
        Default: {}

    batch : bool, optional
        Default: False

        Whether or not to use batch processing.

    batchRegEx : str, optional
        Default: ''

        Name of the pickle analysis output file, regular expressions possible.
        See *glob* module documentation for how to enter regEx. Applicable if
        batch is TRUE.

    batchDir : str, optional
        Default: ''

        Path to the pickle analysis output file directory. Omit the trailing
        '/' (slash) when entering the path. Applicable if batch is TRUE.

    IdName : str, optional
        Default: 'ID'

        Name of the field for the fly ID. Should be the same name as the one
        in *metadataVar.txt*. Highly recommended *NOT* to change it. Also
        *NOT* recommended to change the name and the place in
        *metadataVar.txt*.

    metaDb : str, optional
        Default: ''

        Path of the metadata database file, including the filename and
        extension (if wanted). Applicable if batch is TRUE.

    getDate : bool, optional
        Default: False

        If the baseName of the experiment startswith the date, automatically
        retrieves it when set to TRUE. Terms in the `baseName` are separated
        based on the termSep argument.

    termSep : str, optional
        Default: '_'

        String to be used when separating baseName so that the date can be
        retrieved. Applicable only if `getDate` is TRUE.

    metaVars : str, optional
        Default: 'metadataVar.txt'

        Path to the metadata variable name file. The default file already
        contains pre-defined variable names which are compatible with variable
        names in 'workspaceVar.txt'. So it is suggested that you don't change
        metaVars path and filename.

    Returns
    =======
    None : NoneType

    """
    if varDict == {} and not batch:
        print("Either varDict XOR batch argument should be given.")
        return None

    elif varDict != {} and batch:
        print("Either varDict XOR batch argument should be the default value.")
        return None

    else:
        # start of @TODO
        # @TODO Integrate the block below into getVarNames, otherwise redundant
        metaVars = makePath(metaVars)
        metaVariables = open(metaVars, 'r')
        metaNames = []

        for line in metaVariables:
            if line.startswith('#'):
                pass
            else:
                line = re.sub('\n', '', line)
                line = re.sub('', '', line)
                line = re.sub(' ', '', line)
                if line == '':
                    pass
                else:
                    metaNames.append(line)
        metaVariables.close()
        # end of @TODO
        metaHeader = '\t'.join(metaNames)

        lastID = 0  # ID of the experiment
        checkPoint = False
        metaDb = makePath(metaDb)
        if os.path.isfile(metaDb):
            # means metaDb already exists
            # no need to append the metaHeader in this case
            print('metaDb file already exists. Appending to that file.')

            with open(metaDb, 'r') as infile:
                lines = infile.readlines()
            lines[0] = re.sub('\r\n', '', lines[0])  # Mac specific addition
            lines[-1] = re.sub('\r\n', '', lines[-1])

            if metaHeader != lines[0]: #Changed from != to in
                print("Header of the database is not the same as the "
                      "current header.\nAppending new lines will break"
                      "the structure.\nNothing will be written.")
                return None

            else:
                if lines[0] != lines[-1]:
                    # instead of getting the lastID from the last entry
                    # find the max id and increment the number
                    # more solid way to do it
                    idList = []
                    for line in lines:
                        if line.startswith(IdName):
                            pass
                        else:
                            idList.append(int(line.split('\t')[0]))
                    lastID = max(idList)
                else:
                    # in case it is manually created
                    lastID = 0

            checkPoint = True
            metaDbHandler = open(metaDb, 'a')

        else:
            metaDbHandler = open(metaDb, 'a')
            print('metaDb does not exist. Created a new one.')
            metaDbHandler.write(metaHeader + '\n')
            lastID = 0
            checkPoint = True

        if checkPoint:
            metaContent = []
            if batch:
                flList = glob.glob(os.path.join(batchDir, batchRegEx))
                
                # Trying to sort the data files according to their date, fly number
                # and image number. Implemented by BG using exponentials. 
                try:
                    print('Processed data files are sorted. Meta data will be \
                          appended in order of date, fly, image number.')
                    flList.sort(key=lambda processed_file:\
                                int(os.path.basename(processed_file)[:6])**3 + \
                                int(filter(str.isdigit, 
                                           re.search('fly\d+', processed_file).group(0)))\
                                           **5 + int(processed_file.split('.')[0][-3:]))
                                
                except:
                    print('Processed data files can''t be sorted. Meta data will be \
                          appended randomly.')
                                
                for fl in flList:
                    tempContent = []
                    workspace = loadWorkspace(workspaceFile=fl, gui=False)
                    for var in metaNames:
                        if var == IdName:
                            tempContent.append(str(lastID + 1))
                            lastID += 1
                            continue
                        try:
                            if var == 'date':
                                # @TODO: get rid of harcoded baseName
                                # & date str
                                if getDate:
                                    date = workspace['baseName'].split(termSep)
                                    date = str(date[0])
                                    tempContent.append(date)
                                else:
                                    tempContent.append(str(workspace[var]))
                            else:
                                tempContent.append(str(workspace[var]))
                        except KeyError:
                            tempContent.append('N/A')
                    metaContent.append(tempContent)

            elif not batch:
                tempContent = []
                for var in metaNames:
                    if var == IdName:
                        tempContent.append(str(lastID + 1))
                        continue
                    try:
                        # only get the wanted var names from globals
                        if var == 'date':
                            # @TODO: get rid of harcoded baseName
                            # & date str
                            if getDate:
                                date = varDict['baseName'].split(termSep)
                                date = str(date[0])
                                tempContent.append(date)
                            else:
                                tempContent.append(str(varDict[var]))
                        else:
                            tempContent.append(str(varDict[var]))
                    except KeyError:
                        # some attributes like comments will be typed by hand
                        # populate the corresponding fields with N/A
                        tempContent.append('N/A')
                metaContent.append(tempContent)

            for exp in metaContent:
                exp = str('\t'.join(exp))
                metaDbHandler.write('\n' + exp)
            metaDbHandler.close()

        else:
            print("Undefined problem in extractMetadata."
                  "\nNothing is written.")
            return None

        return None

def loadWorkspace(workspaceFile='', gui=True,
                  initialDirectory='~', extension='*.pickle'):
    """Loads a binary pickle file, which is the workspace of the analysis.

    Parameters
    ==========
    workspaceFile : str, optional
        Default: ''

        If the *gui* argument is False, this path is used to load the
        workspace. Do *not* omit the extension when entering the file name.

    gui : bool,optional
        Default: True

        Whether to use GUI to load a file.

    initialDirectory : str, optional
        Default: HOME directory, i.e. '~'.

        Path into which the directory selection GUI opens.

    extension : str, optional
        Default: '.pickle'

        Extension of the file to be loaded. Applicable when *gui* is *True*.

    Returns
    =======
    workspace : dict
        Whole workspace which is saved after the analysis. Keys are variable
        names.
    """
    if gui and workspaceFile != '':
        print("gui cannot be True when a value is given to workspaceFile"
              "(other than an empty string)")
        return None

    elif not gui and workspaceFile == '':
        print('Gui is disabled and nothing given to workspaceFile'
              'Change either one of them')
        return None

    else:
        if gui and workspaceFile == '':
            initialDirectory = makePath(initialDirectory)
            root = Tkinter.Tk()
            root.withdraw()
            workspaceFile = askopenfilename(parent=root,
                                            initialdir=initialDirectory,
                                            filetypes=[(extension, extension)],
                                            title='Open saved workspace')
        elif not gui and workspaceFile != '':
            workspaceFile = makePath(workspaceFile)

        workspaceFile = open(workspaceFile, 'rb')
        workspace = cPickle.load(workspaceFile)
        workspaceFile.close()

        return workspace
    
def searchDatabase(metaDataBaseFile, conditions,var_from_database_to_select):
    """ Reads the meta database and extracts the imageIDs with the conditions of 
    interest. Also returns the flyIDs together with the imageIDs in a pandas 
    data frame.

    ==========
    metaDataBaseFile : str
        Full path of the meta database file including .txt
        
    conditions : dict
        A dictionary with keys (conditions) that are the columns of meta database 
        file and values that are the conditions of interest.
        
     var_from_database_to_select : list
    
        A list of strings that indicate the data variables from the meta database
        to be kept for further analysis.

    Returns
    =======
    data_to_select : pandas dataframe
        A pandas data frame which keeps the flyIDs and can be used to 
        select data since it also includes the data file names.
    """
    # Read meta data and find the indices of interest
    # The indices will be those that are in the dataframe so that the user can index
    # the data frame to extract current_exp_ID in order to load the dataset
    metaData_frame = pd.read_csv(metaDataBaseFile,sep='\t',header=0)
    positive_indices = {}
    for condition_indentifier, condition in conditions.items():
        if condition.startswith('#'):
            pass
        else:
            currIndices = metaData_frame\
            [metaData_frame[condition_indentifier]==condition].index.values
            positive_indices[condition_indentifier] = set(currIndices.flatten())
        
    common_indices = list(set.intersection(*positive_indices.values()))
    
    data_to_select = pd.DataFrame()
    for variable in var_from_database_to_select:
        data_to_select[variable] = metaData_frame[variable].iloc[common_indices]
        
    
    
    
    return data_to_select




def retrieveData(processedDataStoreDir, data_to_select, var_from_database_to_select,
                 data_ext ='*.pickle'):
    """ Retrives the datasets, according to a specified format, that are 
    identified by the 'imageID' column in the data_to_select' pandas data frame. 
    Additionally appends the flyID inside the dataset. 
    Dataset is returned as a dictionary with each key being a unique dataset.

    ==========
    processedDataStoreDir : str
        Full path of the data storage directories.
        
    data_to_select : pandas dataframe
        A pandas data frame which keeps the flyIDs and can be used to 
        select data since it also includes the data file names.
        
    var_from_database_to_select : list
    
        A list of strings that indicate the data variables from the meta database
        to be kept for further analysis.
        
    data_ext = str, optional
        Default: '*.pickle'
        
        Format of the stored datasets.
        
    Returns
    =======
    all_data: dict
        A dictionary containing each dataset with unique keys.
    """
    
    dataPathString = os.path.join(processedDataStoreDir, data_ext)
    all_data_files = glob.glob(dataPathString)
    
    imageIDs_to_analyze = data_to_select['imageID'].values.tolist()
    
    
    all_data = {}
    for iData, data_file in enumerate(all_data_files):
        file_name = os.path.basename(data_file).split('.')[0]
        if any(file_name == imageID for imageID in imageIDs_to_analyze):
            imageIdx = imageIDs_to_analyze.index(file_name)
            data_file = open(data_file, 'rb')
            if data_ext == '*.pickle':
                data = cPickle.load(data_file)
            else:
                print('The extension type is not compatible with current reading \
                      procedure')
                break
            # Keeping desired variables from the meta data
            for variable in var_from_database_to_select:
                if 'imageID' in variable:
                    pass
                else:
                    data[variable] = data_to_select[variable].values.tolist()[imageIdx]
                
                
            
             
                 
            data_file.close()
            all_data[iData] = data
            
            
        else:
            pass
    return all_data

def selectData(all_data, data_var_to_select):
    """ Selects a subset of data for convenient further analysis.

    ==========
    all_data: dict
        A dictionary containing each dataset with unique keys.
        
    data_var_to_select : list
        A list of strings that indicate the data variables to be selected for 
        further analysis.

    Returns
    =======
     selected_data: dict
        A dictionary containing the selected variables of each dataset with 
        unique keys.
    """
    selected_data = {}
    for key, value in all_data.iteritems():
        current_dataset = all_data[key]
        data_to_select = {}
        
        for variable in data_var_to_select:
            data_to_select[variable] = current_dataset[variable]
        selected_data[key] = data_to_select
    
    return selected_data


# Bleedthrough correction functions for 2p Ultima stimulus bleedthrough (modified from Juan Felipe)

def calculate_spatial_FFT(image_paths):
    #find image size
    im=Image.open(image_paths[0])
    size = np.array(im.size)
    tif_array=np.zeros((size[1],size[0],len(image_paths))) 
    for n_tif,tif in enumerate(image_paths): 
        im=Image.open(image_paths[n_tif])
        im=np.array(im)
        #im=im/np.max(im) #take this out when testing is done
        tif_array[:,:,n_tif]=im

    FFT_2d = fftshift(fft2(tif_array,axes=(0,1)),axes=(0,1)) #fftshift shifts the dc component to the center of the image
    return FFT_2d #this should be 2d numpy arrays in the dict 

def filter_bleedthrough_gaussianstripe(FFT_2d):

    height= FFT_2d.shape[0]
    width= FFT_2d.shape[1]
    middle=[height//2,width//2]
    rectangle_filter=np.ones((height,width,1))

    gaussian_filter=gaussian(width,1)
    gaussian_filter=(gaussian_filter*-1)+1
    rectangle_filter[:,:,0]=np.broadcast_to(gaussian_filter,(height,width))
    center=gaussian(6,5)
    rectangle_filter[middle[0]-3:middle[0]+3,:]=1

    filtered_data_FFT = FFT_2d*rectangle_filter
    return filtered_data_FFT

def backtransform(filtered_data_FFT):

    iFFT =np.abs(ifft2(ifftshift(filtered_data_FFT,axes=(0,1)),axes=(0,1)))
    backtransformed = iFFT.astype(np.int16)

    return backtransformed 