% for scan num5 generate all voxel patch-wised dataset
clear ; close all; clc
cropSizeX = 25;
cropSizeY = 25;
cropSizeZ = 25;

strideX = 10;
strideY = 10;
strideZ = 1;


testimgPath = 'results/all_testimg_1111_prepadding.mat';
testData = load(testimgPath);
testData = testData.testData;
testlabelPath = 'results/all_testlabel_1111_prepadding.mat';
testLabel = load(testlabelPath);
testLabel = testLabel.teLabel;

testData1 = squeeze(testData(5,:,:,:));
testLabel1 = squeeze(testLabel(5,:,:,:));


[imgZ,imgY,imgX] = size(testData1);
            
maxStepsX = floor((imgX-cropSizeX)/strideX)+1;
maxStepsY = floor((imgY-cropSizeY)/strideY)+1;
maxStepsZ = floor((imgZ-cropSizeZ)/strideZ)+1;

maxStep = maxStepsZ*maxStepsY*maxStepsX;

cropImagesPos = uint8(zeros(maxStep,cropSizeZ,cropSizeY,cropSizeX));
cropImagesNeg = uint8(zeros(maxStep,cropSizeZ,cropSizeY,cropSizeX));


posIdx = 1;
negIdx = 1;
for stepZ=1:maxStepsZ

    zStart = (stepZ-1)*strideZ+1;
    zEnd = (stepZ-1)*strideZ+cropSizeZ;
    zMiddle = floor((zStart+zEnd)/2);
            
    for stepY=1:maxStepsY
        for stepX=1:maxStepsX
            yStart = (stepY-1)*strideY+1;
            yEnd = (stepY-1)*strideY+cropSizeY;
            xStart = (stepX-1)*strideX+1;
            xEnd = (stepX-1)*strideX+cropSizeX;
            yMiddle = floor((yStart+yEnd)/2);
            xMiddle = floor((xStart+xEnd)/2);

            voxelImage = testData1(zStart:zEnd,yStart:yEnd,xStart:xEnd);
            voxelSeg = testLabel1(zStart:zEnd,...
                max(yStart,1):min(yEnd,512),...
                max(xStart,1):min(xEnd,512));
            if(sum(abs(voxelImage(:)))>0 )

                % if voxelImage is not black then add it to repos.
                % todo: if we need round the seg?
                isStroke = testLabel1(zMiddle,yMiddle,xMiddle);
                if(isStroke==2)
                    fprintf('******find STROKE at: z:%d,y:%d,x:%d*****\n',zMiddle,yMiddle,xMiddle);
                    cropImagesPos(posIdx,:,:,:) = reshape(voxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                    posIdx = posIdx+1;
                    
                else
                    cropImagesNeg(negIdx,:,:,:) = reshape(voxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                    negIdx = negIdx+1;
                end
            end

        end
        
    end
    
end


cropImagesPos(posIdx:end,:,:,:) = [];
cropImagesNeg(negIdx:end,:,:,:) = [];


save('results/test/img5TestDatasetPos.mat','cropImagesPos','-v7.3')
save('results/test/img5TestDatasetNeg.mat','cropImagesNeg','-v7.3')






