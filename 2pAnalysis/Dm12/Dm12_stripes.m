%% Load the combined processed data table of Luis Giordano Ramos Traslosheros' Dm12 imaging data
clear all
close all

cd("/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/Luis_data/Dm12 imaging")
load('processedTableWithoutBackgroundOnOff.mat')

%% Find the Dm12 vertical OFF stripes flyIDs
tSeriesNames = ProcessedTable.timeSeriesPath;

correctIndices = find(contains(ProcessedTable.timeSeriesPath, "Dm12") .* contains(ProcessedTable.stimParamFileName,'StandingStripe_1s_XAxis_5degWide_2degSep_m1.0Con_rand_USEFRUSTUM.txt'));
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

nEpochs = 25;
stimLen = 21; %1s BG 1s stripe = 2s
mean_fly_traces = zeros(stimLen, nFlies, nEpochs);
nROIs = 0;

allTunings = {}; 
iFitStrct = 1;

for iFlyID = 1:nFlies
    
    currSeriesIDs = correctIndices(strcmp(flyIDs,uniqIDs{iFlyID}));
    currFlyTraces = zeros(stimLen, length(currSeriesIDs), nEpochs);
    
    roiIdx = 1;
    % Take all the Series for this fly ID
    for iID = 1:length(currSeriesIDs)
        currID = currSeriesIDs(iID);
        curr_ROIn = size(ProcessedTable.paddedEpochStacks{currID}{1},2);
        
        currROItunings = zeros(curr_ROIn, nEpochs);

        
        for iROI = 1:curr_ROIn
            
            fitStrct.fit{iFitStrct} = ProcessedTable.paramTuning{currID}(iROI).fit;
            fitStrct.rsq{iFitStrct} = ProcessedTable.paramTuning{currID}(iROI).gof.rsquare ;
            fitStrct.flyID{iFitStrct} = iFlyID;
            fitStrct.rqi{iFitStrct} = ProcessedTable.responseIndex{currID}(iROI);
            
            
            for iEpoch = 1:nEpochs
                % Calculate response for each stripe
                % mean of the 1st second (bg epoch) and mean of the second
                % (stimulation epoch)
                epochTrace = ProcessedTable.paddedEpochStacks{currID}{iEpoch}(:,iROI);
                
                epochResp = max(epochTrace(11:end)) - mean(epochTrace(5:10));

                currROItunings(iROI, iEpoch) = epochResp;
                
            end
            fitStrct.tuning{iFitStrct} = currROItunings(iROI, 1:end);
            iFitStrct = iFitStrct+1;
        end
        allTunings{iFlyID}{iID} = currROItunings; 
        
        
        nROIs = nROIs + curr_ROIn;
    end
    
    mean_fly_traces(:, iFlyID, :) = mean(currFlyTraces,2);
    
end
%% Organize the data into a big array
allRoiData = zeros(nROIs, nEpochs);
currId = 1;
for iFlyID = 1:nFlies
    currTunings = cell2mat(allTunings{iFlyID}');
    currROIn = size(currTunings,1);
    allRoiData(currId:currId+currROIn-1, 1:end) = currTunings;
    
    currId = currId+currROIn;
end
%% Plot each fly in a subplot
figure(1)
for iFlyID = 1:length(uniqIDs)
    
    currTunings = cell2mat(allTunings{iFlyID}');
    subplot(4,4,iFlyID)
    imagesc(currTunings)
    colormap(redblue)
    colorbar
    caxis([-0.5 0.5])
end
%% Plot a fly

figure(2)
imagesc(allTunings{4}{1})
colormap(redblue)
colorbar
caxis([-0.5 0.5])

%% Plot good fits
passedStrct = struct;
iPassed = 1;
close all
for iROI = 1:size(fitStrct.fit,2)
    if fitStrct.rsq{iROI} < 0.2 || fitStrct.rqi{iROI} <0.5 || (fitStrct.fit{iROI}.b1 < 2 || fitStrct.fit{iROI}.b1 > 23)
        continue
    end
    
    passedStrct.fit{iPassed} = fitStrct.fit{iROI};
    passedStrct.rsq{iPassed} = fitStrct.rsq{iROI};
    passedStrct.flyID{iPassed} = fitStrct.flyID{iROI};
    passedStrct.rqi{iPassed} = fitStrct.rqi{iROI};
    passedStrct.tuning{iPassed} = fitStrct.tuning{iROI};
    
    passedStrct.fwhm{iPassed} = 2*sqrt(log(2)) * fitStrct.fit{iROI}.c1*2;
    iPassed = iPassed+1;
        

end
boxplot(cell2mat(passedStrct.fwhm))
ylim([0,30])

%% Plot each fly in a subplot
figure(1)
currFlies = cell2mat(passedStrct.flyID);
uniqFlies = unique(currFlies);
allTunings = cell2mat(passedStrct.tuning');
for iFlyID = 1:length(uniqFlies)
    currFlyID = uniqFlies(iFlyID);
    
    currTunings = allTunings(cell2mat(passedStrct.flyID) == currFlyID,1:end);
    subplot(3,2,iFlyID)
    imagesc(currTunings)
    colormap(redblue)
    colorbar
    caxis([-0.5 0.5])
end
%% Show the data array
imagesc((allRoiData'./max(allRoiData'))')
imshow((allRoiData'./max(allRoiData'))')
imshow(allRoiData)
%% OFF
x = linspace(0,6,61)';
y = mean(squeeze(mean_fly_traces(:,:,1)),2);
err = std(squeeze(mean_fly_traces(:,:,1)),[],2)/ sqrt(nFlies); 
fill([x;flipud(x)],[y-err;flipud(y+err)],[.9 .9 .9],'linestyle','none');
line(x,y)
title(sprintf("OFF Dm12 , N = %d ( %d )",nFlies,nROIs))

% plot(mean_fly_traces(:,:,1))

%% ON
x = linspace(0,6,61)';
y = mean(squeeze(mean_fly_traces(:,:,2)),2);
err = std(squeeze(mean_fly_traces(:,:,2)),[],2)/ sqrt(nFlies); 
fill([x;flipud(x)],[y-err;flipud(y+err)],[.9 .9 .9],'linestyle','none');
line(x,y)
title(sprintf("ON Dm12 , N = %d ( %d )",nFlies,nROIs))

