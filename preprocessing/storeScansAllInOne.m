function storeScansAllInOne()
    clear ; close all; clc
    imageRootPath = 'images';
    annotationPath = 'annotations';
    savePrefix='1108';

    trainList = importdata('trainList.data','\n',1000);
    testList = importdata('testList.data','\n',1000);

    trDataPath = strcat('results/all_img_' , savePrefix);
    trLabelPath = strcat('results/all_label_' , savePrefix);
    teDataPath = strcat('results/all_testimg_' , savePrefix);
    teLabelPath = strcat('results/all_testlabel_' , savePrefix);

    
    trainData = storeAllImage(trainList);
    save(strcat(trDataPath,'.mat'),'trainData','-v7.3');
    fprintf('successfully save:%s \n',trDataPath);
    
    testData = storeAllImage(testList);
    save(strcat(teDataPath,'.mat'),'testData','-v7.3');
    fprintf('successfully saved:%s \n',teDataPath);
    
    trLabel = storeAllSegmenttion(trainList);
    save(strcat(trLabelPath,'.mat'),'trLabel','-v7.3');
    fprintf('seg successfully saved:%s \n',trLabelPath);
    
    teLabel = storeAllSegmenttion(testList);
    save(strcat(teLabelPath,'.mat'),'teLabel','-v7.3');
    fprintf('seg successfully saved:%s \n',teLabelPath);
    
    
    
    function[scanData] = storeAllImage(list)
        
    % max image number:56
%      maxv = 0;
%             if maxv <= size(images,1)
%                 maxv = size(images,1);
% 
%                 fprintf('max image number:%d %s \n',maxv,char(scanId(1)));
%             end

        scanNumber = size(list,1);
        scanData = uint8(zeros(scanNumber,56,512,512));
        for i=1:scanNumber
            scanId = list(i);
            scanPath = strcat(imageRootPath,'/',scanId(1),'/imgs/*.jpeg');
            images = dir(char(scanPath));
            fprintf('start load image:%s \n',char(scanId));
            for j=1:size(images,1)
                imageId = images(j).name;
                imagePath = strcat(imageRootPath,'/',scanId(1),'/imgs/',imageId);
                scanData(i,j,:,:) = imread(char(imagePath));
                
            end
            
            
        end
        

    end

    function[scanData]= storeAllSegmenttion(list)
        
        scanNumber = size(list,1);
        scanData = uint8(zeros(scanNumber,56,512,512));
        for i=1:scanNumber
            scanId = list(i);
            scanPath = strcat(annotationPath,'/',scanId(1),'/*.png');
            images = dir(char(scanPath));
            fprintf('start load segmentation:%s \n',char(scanId));
            for j=1:size(images,1)
                imageId = images(j).name;
                imagePath = strcat(annotationPath,'/',scanId(1),'/',imageId);
                
                segMap = segBitmap(char(imagePath));
                
                currendId = str2double(imageId(end-6:end-4));
                scanData(i,currendId,:,:) = segMap;
                
            end
            
            
        end
        
        
    end

    function[annotation]= segBitmap(link)
        X = imread(link);
        annotation = X(:, :, 1);
        annotation = bitor(annotation, bitshift(X(:, :, 2), 8));
        annotation = bitor(annotation, bitshift(X(:, :, 3), 16));
    end



end

