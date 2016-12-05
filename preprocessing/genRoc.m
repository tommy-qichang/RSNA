clear ; close all; clc
rootPath = '../training/25^3_fcnn_i12_r0.1_w50/rocData/';

startId = 7;

for id = startId:4:55
    resultid = id;
    
    result = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/result');
    target = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/target');

    result = permute(result,[3,2,1]);
    target = permute(target,[3,2,1]);
    target = target -1;

    bsize = size(result,1);
    finalResult = zeros(bsize,1);
    finalTarget = zeros(bsize,1);
    for i=1:bsize 
        img = squeeze(result(i,:,:));
        target1 = squeeze(target(i,:,:));
        img = medfilt2(img,[3,3]);
        maxV = max(img(:));
        isStroke = 0;
        if(sum(target1(:))>0)
            isStroke = 1;
        end
        finalResult(i) = maxV;
        finalTarget(i) = isStroke;
    end

    finalTarget = finalTarget+1;
    [X,Y,T,AUC] = perfcurve(finalTarget,finalResult,2);

    fprintf(strcat('iter:',num2str(id),' with AUC:%.4f \n'),AUC);
end






