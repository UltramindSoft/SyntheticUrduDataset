clear all
close all
clc
gpuDevice(1)
% return;
Yassers_Model='Yasser_HoriX_v1_Urdu_SSD_Trained_On_';
%%
% Load training data.
% data = load('Training_For_Horizontal_Regression_Network_Yasser.mat', 'RotatedCoordinates_plus_Angle', 'imageFileName');
data = load('Training_For_Horizontal_Regression_Network_Yasser_4228images.mat');
% stopSigns2 = struct2table(data.TrainingDataForRegression);
stopSigns2 = (data.TrainingDataForRegression);
% data2 = load('rcnnStopSigns.mat','stopSigns','fastRCNNLayers');
% fastRCNNLayers = data2.fastRCNNLayers;

% rng('default');
% Used_Model='_Custom_';
% Used_Model='squeezenet';
% Used_Model='vgg16';    % memory Error
Used_Model='resnet50';
% Used_Model='alexnet';
% Used_Model='googlenet';
% Used_Model='inceptionv3';
% Used_Model='vgg19';     % Error Nan-Values
% Used_Model='resnet18';
% Used_Model='inceptionresnetv2';  

% % % % net = load('ssdVehicleDetector.mat');
% % % % lgraph = net.lgraph

% %     'alexnet'
% %     'vgg16'
% %     'vgg19'
% %     'resnet50'
% %     'resnet101'
% %     'inceptionv3'
% %     'googlenet'
% %     'inceptionresnetv2'
% %     'squeezenet'

%             ---->>>      ------>   Trained on :::   4212-images  <<-----
%%
b=[];
% Add fullpath to image files.
for kYasser=1:size(stopSigns2,2)
    stopSigns2(kYasser).imageFileName = fullfile(pwd,(stopSigns2(kYasser).imageFileName));
    temp=stopSigns2(kYasser).RotatedCoordinates_plus_Angle;
    stopSigns2(kYasser).RotatedCoordinates_plus_Angle=temp;
end

s3=struct2table(stopSigns2);
for kYasser=1:size(s3,1)
    temp=cell2mat(s3.RotatedCoordinates_plus_Angle(kYasser));
    s3.RotatedCoordinates_plus_Angle{kYasser}=str2num(temp);
end


%% Testing Rectangles on Original images

% % % % for kYasser=1:size(s3,1)/100
% % % %     imshow(imread(s3.imageFileName{kYasser}))
% % % %     rectangle('Position',s3.RotatedCoordinates_plus_Angle{kYasser});
% % % %     pause(0.5);
% % % % end
% disp(s3);
% return

%%
% % % % % % %%
% % % % % % imageAugmenter = imageDataAugmenter( ...
% % % % % %     'RandRotation',[-20,20], ...
% % % % % %     'RandScale',[0.5 1.5])
% % % % % % 
% % % % % % imageSize = [320 240 3];
% % % % % % augimds = augmentedImageDatastore(imageSize,s3,'DataAugmentation',imageAugmenter);
% % % % % % 
% % % % % % minibatch = preview(augimds);
% % % % % % imshow(imtile(minibatch.input));

% s3=s3(1:200,:);
%%
% Set random seed to ensure example training reproducibility.
% rng(0);
rng('default');

% Randomly split data into a training and test set.
shuffledIndices = randperm(height(s3));
idx = floor(0.9 * length(shuffledIndices) );
trainingData = s3(shuffledIndices(1:idx),:);
testData = s3(shuffledIndices(idx+1:end),:);

trainingData2=trainingData;
imgRows=240;
imgCols=320;
ColExceeded=0
RowExceeded=0
BoundaryFlag=0
for LoopSize=1:size(trainingData2,1)/1
            imP=trainingData2.imageFileName{LoopSize};
% % % %             imG=imread(imP);
% % % %             hold on
            CoOrs=trainingData2.RotatedCoordinates_plus_Angle{LoopSize};
% % % %             imshow(imG);
                    if CoOrs(1)+CoOrs(4) > imgCols 
                        disp(' Col Range Exceeded ...');
                        maxRows=CoOrs(1)+CoOrs(4)
                        plot(maxRows,CoOrs(2),'r*');
                        disp(' Col-Ended-Range Exceeded ...');
                        ColExceeded=ColExceeded+1;
                        BoundaryFlag=1;
                        trainingData2.RotatedCoordinates_plus_Angle{LoopSize}=[];
                    end
                       if  CoOrs(2)+CoOrs(3) > imgRows
                        disp(' Row Range Exceeded ...');
                        maxCols=CoOrs(2)+CoOrs(3)
                        plot(maxRows,CoOrs(2),'b*');
                        disp(' Row-Ended-Range Exceeded ...');
                        RowExceeded=RowExceeded+1;
                        BoundaryFlag=1;
                        trainingData2.RotatedCoordinates_plus_Angle{LoopSize}=[];
                       end
                    if CoOrs(1) <1
                        disp(['Less than 1 row found at-->  ' num2str(LoopSize)]);
                        trainingData2.RotatedCoordinates_plus_Angle{LoopSize}=[];
                    end
                    if CoOrs(2) <1
                        disp(['Less than 1 col found at-->  ' num2str(LoopSize)]);
                        trainingData2.RotatedCoordinates_plus_Angle{LoopSize}=[];
                    end
% % % %                     rectangle('Position',CoOrs);
    %     imG = insertObjectAnnotation(imG,'Rectangle',CoOrs,'- -');
% % % %             drawnow
            if BoundaryFlag==1
                pause(0.1)
               BoundaryFlag=0;
            end
            LoopSize
% % % %             hold off
end
ColExceeded
RowExceeded


%Remove empty rows, acoording to 2nd empty column
idx=all(cellfun(@isempty,trainingData2{:,2}),2);
trainingData2(idx,:)=[];

% % % % % % % % % % % %% Showing cleaned annotations
% % % % % % % % % % % figure,
% % % % % % % % % % % for LoopSize=1:size(trainingData2,1)/1
% % % % % % % % % % %             imP=trainingData2.imageFileName{LoopSize};
% % % % % % % % % % %             CoOrs=trainingData2.RotatedCoordinates_plus_Angle{LoopSize};
% % % % % % % % % % %             imG=imread(imP);
% % % % % % % % % % %             imshow(imG);
% % % % % % % % % % %             hold on      
% % % % % % % % % % %             rectangle('Position',CoOrs);
% % % % % % % % % % %     %     imG = insertObjectAnnotation(imG,'Rectangle',CoOrs,'- -');
% % % % % % % % % % %             drawnow
% % % % % % % % % % %             pause(0.01)
% % % % % % % % % % %             LoopSize
% % % % % % % % % % %             hold off
% % % % % % % % % % % end
% % 
% % for LoopSize=1:size(trainingData2,1)/1
% %     CoOrs=trainingData2.RotatedCoordinates_plus_Angle{LoopSize};
% %     if CoOrs(1) <1
% %         disp(['Less than 1 row found-->  ' num2str(LoopSize)]);
% %     end
% %     if CoOrs(2) <1
% %         disp(['Less than 1 col found-->  ' num2str(LoopSize)]);
% %     end
% %     if CoOrs(3) <5
% %         disp(['Less than 1 Width found-->  ' num2str(LoopSize)]);
% %     end
% %      if CoOrs(4) <5
% %         disp(['Less than 1 Height found-->  ' num2str(LoopSize)]);
% %     end
% % end
    
    
% % % imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
% % % bldsTrain = boxLabelDatastore(trainingDataTbl(:,'Tip'));
% % % imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
% % % bldsTest = boxLabelDatastore(testDataTbl(:,'Tip'));
% % % trainingData = combine(imdsTrain,bldsTrain);
% % % testData = combine(imdsTest,bldsTest);
%%
% disp('Estimating Anchorboxes values from data..... (wait a little)');
% trainingData_estimate = boxLabelDatastore(trainingData(:,2:end));
% numAnchors = 5;
% [anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData_estimate,numAnchors);
% anchorBoxes
% % % % % % % % maxNumAnchors = 8;
% % % % % % % % meanIoU = zeros([maxNumAnchors,1]);
% % % % % % % % anchorBoxes = cell(maxNumAnchors, 1);
% % % % % % % % for k = 1:maxNumAnchors
% % % % % % % %     % Estimate anchors and mean IoU.
% % % % % % % %     [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(trainingData_estimate,k);    
% % % % % % % % end
% % % % % % % % figure
% % % % % % % % plot(1:maxNumAnchors,meanIoU,'-o');
% % % % % % % % ylabel("Mean IoU");
% % % % % % % % xlabel("Number of Anchors");
% % % % % % % % title("Number of Anchors vs. Mean IoU");

%%
% Set network training options:
%
% * Set the CheckpointPath to save detector checkpoints to a temporary
%   directory. Change this to another location if required.

% %   SelRange=20;
% %   trainingData=trainingData(1:SelRange,:);
% %   testData=testData(1:SelRange,:);

% % imds = imageDatastore(trainingData.imageFileName);
% % blds = boxLabelDatastore(trainingData(:,2:end));
% % ds = combine(imds, blds);


% % % % load('New_SSD_Model2.mat');
% % % % imds = imageDatastore(trainingData2.imageFileName);
% % % % blds = boxLabelDatastore(trainingData2(:,2:end));
% % % % ds = combine(imds, blds);
% 
% load('New_SSD_Model3.mat');


startI=1;
% imageRange=200;
imageRange=3766;
trainingData3=[];
trainingData3=trainingData2(startI:imageRange,:);
% % % % disp('Estimating Anchorboxes values from data..... (wait a little)');
% % % % trainingData3_estimate = boxLabelDatastore(trainingData3(:,2:end));
% % % % numAnchors = 12;
% % % % [anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData3_estimate,numAnchors);
% % % % anchorBoxes
imds = imageDatastore(trainingData3.imageFileName,'ReadFcn',@Yasser_customreader);
blds = boxLabelDatastore(trainingData3(startI:imageRange,2:end));
ds = combine(imds, blds);
% dat11 = preview(ds)
for i = 2:5
    img = readimage(imds,i);
    figure,imshow(img);
    hold on;
    rectangle('Position',ds.UnderlyingDatastores{1, 2}.LabelData{i, 1});
end
hold off;

% return;

inputSize = [320 320 3];
numClasses = width(trainingData3)-1;
% lgraph_1 = ssdLayers(inputSize, numClasses, 'resnet50');
lgraph_1 = ssdLayers(inputSize, numClasses, Used_Model);
% return;

% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',1, ...
%     'RandScale',[0.5 1], ...
%     'RandYTranslation',[-3 3],...
%     'RandXShear',[1 1],...
%     'RandYShear', [1 1])
% augimds = augmentedImageDatastore(inputSize,ds,'DataAugmentation',imageAugmenter);

% anchorBoxes =
%     82    75
%     82    47
%     82    97
%     82    58
%     82    83
%     82    90
%     82    66
% data = load('vehicleTrainingData.mat');
% trainingData3 = data.vehicleTrainingData;
% dataDir = fullfile(toolboxdir('vision'),'visiondata');
% trainingData3.imageFilename = fullfile(dataDir,trainingData3.imageFilename);
% imds = imageDatastore(trainingData3.imageFilename);
% blds = boxLabelDatastore(trainingData3(:,2:end));
% ds = combine(imds, blds);
% net = load('ssdVehicleDetector.mat');
% lgraph_1 = net.lgraph
%  82,75;82,47;82,97;82,58;82,83;82,90;
% 82,70;82,58;82,87;82,47;
            % %   82,81;82,64;82,67;82,74;82,78;82,97;
            % resnet50_modified_v3_anchorboxes=frcnn.Network;
            % resnet50_modified_v3_anchorboxes=lgraph_4;
            % save('resnet50_modified_v3_anchorboxes.mat','resnet50_modified_v3_anchorboxes');
% LD=load('Yasser_HoriX_v1_Urdu_SSD_Trained_On_3766_Tested_On_423-images_n_Model-Name_resnet50_Ep20_Tr_ap-0.0031402_Ts_ap-0.00028445_29759.6833_.mat');
% frcnn=LD.frcnn;


% % % LD=load('YDataset_best_SSD_model_7July2020.mat');
% % % frcnn=LD.frcnn;
LD=load('Synthetic_Natural_Model_v3.mat');
frcnn=LD.Synthetic_Natural_Model_v3;
YasserEpochs=15;
for i=1:1   %//  4x5=20 epochs
   
%         options = trainingOptions('sgdm', ...
%             'MiniBatchSize', 2, ...
%             'ExecutionEnvironment','cpu', ...
%             'InitialLearnRate', 1e-3, ...
%             'MaxEpochs',YasserEpochs, ...
%             'CheckpointPath', tempdir);
        %  return;
        options = trainingOptions('sgdm',...
          'InitialLearnRate',1e-3,...
          'MiniBatchSize',8,...
          'Verbose',true,...
          'MaxEpochs',YasserEpochs,...
          'Shuffle','every-epoch',...
          'VerboseFrequency',10,...
          'CheckpointPath',[pwd '\ssd_Training']);


        %%
        if i==1
            Used_Model_U=Used_Model; 
        else
            load(YsrModel_Name);
            Used_Model_U=frcnn;
        end
        
             
        tic;
        % Train the Fast R-CNN detector. Training can take a few minutes to complete.
%         [detector,info] = trainSSDObjectDetector(ds,lgraph,options);
%         [frcnn,info] = trainSSDObjectDetector(ds,lgraph_1,options);

%           load('Yasser_HoriX_v1_Urdu_SSD_Trained_On_3766_Tested_On_423-images_n_Model-Name__Custom__Ep100_TrainTime_92311.9878seconds_.mat');
%           [frcnn,info] = trainSSDObjectDetector(ds,frcnn,options);
%                load('resnet50_Anchorboxes_model.mat');
% % % % %             [frcnn,info] = trainSSDObjectDetector(ds,frcnn,options);
%  [frcnn,info] = trainSSDObjectDetector(ds,lgraph_4,options);
 [frcnn,info] = trainSSDObjectDetector(ds,frcnn,options);
%  [frcnn,info] = trainSSDObjectDetector(ds,lgraph_4,options);
%             [frcnn,info] = trainSSDObjectDetector(ds,resnet50_modified_v4_anchorboxes,options);
          %         [frcnn,info] = trainSSDObjectDetector(ds,lgraph,options);
%         [frcnn,info]= trainFasterRCNNObjectDetector(trainingData, Used_Model_U , options, ...
%             'NegativeOverlapRange', [0 0.1], ...
%             'PositiveOverlapRange', [0.5 1], ...
%             'SmallestImageDimension', 300);
        Y_endTime=toc;
        Y_TrainTime=Y_endTime;

        old_Epochs=YasserEpochs*i;
        YsrModel_Name=[Yassers_Model num2str(size(trainingData3,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(old_Epochs) '_TrainTime_' num2str(Y_TrainTime) 'seconds_.mat'];
        save(YsrModel_Name,'frcnn','info','Y_TrainTime');

        
        figure,
        plot(info.TrainingLoss);
        grid on;
        xlabel('Number of Iterations');
        ylabel('Training Loss for Each Iteration');
        figure
                % % % % % % % 
                % % % % % % % %%

                %% Retraining of Detector
                % load('fast_rcnn_checkpoint__14520__2019_04_26__02_02_34.mat')
                % % % % % %  load('faster_rcnn_stage_3_checkpoint__7582__2019_05_30__11_59_38.mat')
                % % % % % tic;
                % % % % % frcnn = trainFasterRCNNObjectDetector(trainingData, detector , options, ...
                            % % % % % % YasserEpochs=25;
                            % % % % tic
                            % % % % 
                            % % % % 
                            % % % % %//////////////////////////////////////////////////////////////////
                            % % % % %//////////////////////  Training Accuracy ////////////////////////////////////////////
                            % % % % %////////////frcnn = trainFasterRCNNObjectDetector(trainingData, frcnn , options, ...
                            % % % %     'NegativeOverlapRange', [0 0.1], ...
                            % % % %     'PositiveOverlapRange', [0.5 1], ...
                            % % % %     'SmallestImageDimension', 300);
                            % % % % Y_endTime=toc;
                            % % % % Y_TrainTime=Y_endTime;/////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                results=[];
                numImages = size(trainingData3,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((trainingData3.RotatedCoordinates_plus_Angle));
                BlackZerosImg=uint8(zeros(320,320,3));
                hold on,
                for i = 1:numImages
                %                 I = (imread(stopSigns2(i).imageFileName));
                                I = imread(trainingData3.imageFileName{i});
                                BlackZerosImg(1:240,1:320,:)=I;
                            %     RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
                            %     Following function 'YsrNetCopiedCode_RatioPreserve' is only necessary for
                            %     InceptionV3. Others Alexnet+Googlenet+Squeeznet automatically adjusts for
                            %     the image input size.

                %///////////////////////////////////////////////////////////////////////////            
                % % % % %                 I=YsrNetCopiedCode_RatioPreserve(I,299);
                % % % % %                 GroundTruthCoords=cell2mat(GroundTruth.Var1(i));
                % % % % %                 GroundTruth.Var1{i,1}(1)=GroundTruth.Var1{i,1}(1)-10;   
                %///////////////////////////////////////////////////////////////////////////

                            %     imshow(I);
%                                 [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
%                                 detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');
                                [bboxes,scores] = detect(frcnn,BlackZerosImg,'ExecutionEnvironment','gpu');


                % % % %                 GroundTruthCoords(2)=GroundTruthCoords(2)-10;
                % % % %                 GroundTruthCoords(1)=GroundTruthCoords(1)-11;   % changing column value
                % %                 detectedImg = insertShape(detectedImg, 'Rectangle',GroundTruthCoords ,'Color','green');
                %                 imshow(detectedImg)
                %                 drawnow 
                %                 pause(0.01);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Tr-' num2str(i)]);
                end
                results = struct2table(results);

                % GroundTruth=table((s3.RotatedCoordinates_plus_Angle));
                % [ap,recall,precision] = evaluateDetectionPrecision(results(1:20,:),GroundTruth(1:20,:));
                [ap_Train,Train_recall,Train_precision] = evaluateDetectionPrecision(results,GroundTruth);
                figure
                plot(Train_recall,Train_precision)
                grid on
                title(sprintf('Train-Set Average Precision = %.4f',ap_Train));
                TrResults={ap_Train,Train_recall,Train_precision};
                TrGroundTruth=GroundTruth;
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////


                %//////////////////////////////////////////////////////////////////
                %//////////////////////////// Testing Accuracy//////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                figure,
                results=[];
                numImages = size(testData,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((testData.RotatedCoordinates_plus_Angle));
                BlackZerosImg=uint8(zeros(320,320,3));
                hold on,
                for i = 1:numImages/1
                %                 I = (imread(stopSigns2(i).imageFileName));
                                I = imread(testData.imageFileName{i});
                            %     RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
                            %     Following function 'YsrNetCopiedCode_RatioPreserve' is only necessary for
                            %     InceptionV3. Others Alexnet+Googlenet+Squeeznet automatically adjusts for
                            %     the image input size.

                %///////////////////////////////////////////////////////////////////////////            
                % % % % %                 I=YsrNetCopiedCode_RatioPreserve(I,299);
                % % % % %                 GroundTruthCoords=cell2mat(GroundTruth.Var1(i));
                % % % % %                 GroundTruth.Var1{i,1}(1)=GroundTruth.Var1{i,1}(1)-10;   
                %///////////////////////////////////////////////////////////////////////////

                            %     imshow(I);
                                BlackZerosImg(1:240,1:320,:)=I;
                                [bboxes,scores] = detect(frcnn,BlackZerosImg,'ExecutionEnvironment','gpu');
                                detectedImg = insertShape(BlackZerosImg, 'Rectangle', bboxes,'Color','red');
%                                 [bboxes,scores] = detect(frcnn,I,'ExecutionEnvironment','gpu');
%                                 detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');


                % % % %                 GroundTruthCoords(2)=GroundTruthCoords(2)-10;
                % % % %                 GroundTruthCoords(1)=GroundTruthCoords(1)-11;   % changing column value
                % % % %                 detectedImg = insertShape(detectedImg, 'Rectangle',GroundTruthCoords ,'Color','green');
                                imshow(detectedImg)
                                drawnow 
                                pause(0.1);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Ts-' num2str(i)]);
% % %                                 spath=[fullfile(pwd,'SSD_Test_Detection_Results') '\SSD_Detections' num2str(i)];
% % % % %                                 savefig([fullfile(pwd,'SSD_Test_Detection_Results') '\SSD_Detections' num2str(i)])
% % %                                 f = gcf;
% % %                                 % Requires R2020a or later
% % %                                 exportgraphics(f,[spath '.png'],'Resolution',600);
                end
                results = struct2table(results);

                % GroundTruth=table((s3.RotatedCoordinates_plus_Angle));
                % [ap,recall,precision] = evaluateDetectionPrecision(results(1:20,:),GroundTruth(1:20,:));
                [ap_Test,Test_recall,Test_precision] = evaluateDetectionPrecision(results,GroundTruth);
                figure
                plot(Test_recall,Test_precision)
                grid on
                title(sprintf('Test-Set Average Precision = %.4f',ap_Test))
                xlabel('Recall');
                ylabel('Precision');
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                TsResults={ap_Test,Test_recall,Test_precision};
                TsGroundTruth=GroundTruth;
                YsrModel_Name=[Yassers_Model num2str(size(trainingData3,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(old_Epochs) '_Tr_ap-' num2str(ap_Train) '_Ts_ap-' num2str(ap_Test) '_' num2str(Y_TrainTime) '_.mat'];
                save(YsrModel_Name,'frcnn','info','TrResults','YasserEpochs','TsResults','TrGroundTruth','TsGroundTruth','ap_Train','ap_Test','Y_TrainTime','YasserEpochs');
                pause(1);
                gpuDevice(1);
                pause(4);
end
% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % 
% % % % % % % % % %
% % load(YsrModel_Name);
% % % % % % % % % %
% % % % % % % % % % % Test the Fast R-CNN detector on a test image.
% % img = imread('Ytest_Urdu_localization.jpg');
% % % img = imcomplement(imread('17.jpg'));
% % % img = (imread('17.jpg'));
% % img = (imread('18.jpg'));
% % % 
% % % % Run the detector.
% % [bbox, score, label] = detect(frcnn, img,'ExecutionEnvironment','cpu');
% % 
% % %
% % % % Display detection results.
% % % detectedImg = insertShape(img, 'Rectangle', bbox,'Color','green');
% % % figure
% % % imshow(detectedImg)


%%

% % function RatioPreservedImage=YsrNetCopiedCode_RatioPreserve(YourImage,EqualDimenstion)
% %             %figure out the pad value to pad to white
% %             if isinteger(YourImage)
% %                pad = intmax(class(YourImage));
% %             else
% %                pad = 1;   %white for floating point is 1.0
% %             end
% %             %figure out which dimension is longer and rescale that to be the 256
% %             %and pad the shorter one to 256
% % %             EqualDimenstion=256
% %             [r, c, ~] = size(YourImage)
% %             if r > c
% %               newImage = imresize(YourImage, EqualDimenstion / r);
% %               NewImage(:, end+1 : EqualDimenstion, :) = pad;
% %             elseif c > r
% %               newImage = imresize(YourImage, EqualDimenstion / c);
% %               NewImage(end+1 : EqualDimenstion, :, :) = pad;
% %             else
% %               newImage = imresize(YourImage, [EqualDimenstion, EqualDimenstion]);
% %             end
% %             RatioPreservedImage=newImage;
% % end
