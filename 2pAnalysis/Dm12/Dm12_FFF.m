%% Load the combined processed data table of Luis Giordano Ramos Traslosheros' Dm12 imaging data
clear all
close all

cd("/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/Luis_data/Dm12 imaging")
load('processedTableWithoutBackgroundOnOff.mat')

%% Find the Dm12 FFF flyIDs
tSeriesNames = ProcessedTable.timeSeriesPath;

correctIndices = find(contains(ProcessedTable.timeSeriesPath, "Dm12") .* contains(ProcessedTable.stimParamFileName,'LocalCircle_ONOFF_2s_BG_4s_120deg_0degAz_0degEl_100_Con_0.5Lum_NonRand.txt'));
correctFiles = ProcessedTable.timeSeriesPath(correctIndices);

flyIDs = {};

for iFile = 1:length(correctFiles)
    currSeriesName = correctFiles{iFile};
    aa = split(currSeriesName,'\' );
    flyID = aa{5};
    flyIDs{iFile} = flyID;
    
end
uniqIDs =unique(flyIDs);
nFlies = length(uniqIDs);

%% Get the mean responses per fly

nEpochs = 2;
stimLen = 61; %2s BG 2s OFF (or ON) 2s BG = 6s
mean_fly_traces = zeros(stimLen, nFlies, nEpochs);
nROIs = 0;
for iFlyID = 1:length(uniqIDs)
    
    currSeriesIDs = correctIndices(strcmp(flyIDs,uniqIDs{iFlyID}));
    currFlyTraces = zeros(stimLen, length(currSeriesIDs), nEpochs);
    
    % Take all the Series for this fly ID
    for iID = 1:length(currSeriesIDs)
        currID = currSeriesIDs(iID);
        
        for iEpoch = 1:nEpochs
            % Mean for the Series
            allTraces = ProcessedTable.paddedEpochStacks{currID}{iEpoch};
            passedTraces = allTraces(:,ProcessedTable.responseIndex{currID}>0.5);
            currFlyTraces(:,iID,iEpoch) = nanmean(passedTraces,2);
        end
        nROIs = nROIs + size(passedTraces,2);
    end
    
    mean_fly_traces(:, iFlyID, :) = nanmean(currFlyTraces,2);
    
end

%% OFF
x = linspace(0,6,61)';
y = nanmean(squeeze(mean_fly_traces(:,:,1)),2);
err = nanstd(squeeze(mean_fly_traces(:,:,1)),[],2)/ sqrt(nFlies); 
fill([x;flipud(x)],[y-err;flipud(y+err)],[.9 .9 .9],'linestyle','none');
line(x,y)
title(sprintf("OFF Dm12 , N = %d ( %d )",nFlies,nROIs))
ylim([-0.1 0.3])
% plot(mean_fly_traces(:,:,1))

%% ON
x = linspace(0,6,61)';
y = nanmean(squeeze(mean_fly_traces(:,:,2)),2);
err = nanstd(squeeze(mean_fly_traces(:,:,2)),[],2)/ sqrt(nFlies); 
fill([x;flipud(x)],[y-err;flipud(y+err)],[.9 .9 .9],'linestyle','none');
line(x,y)
title(sprintf("ON Dm12 , N = %d ( %d )",nFlies,nROIs))
ylim([-0.1 0.3])
