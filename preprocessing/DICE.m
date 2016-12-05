clear ; close all; clc
rootPath = '../training/25^3_fcnn_i12_r0.1_w50/rocData/';

startId = 7;
threshold = 0.9;

for id = startId:4:55
    resultid = id;
    
    result = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/result');
    target = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/target');

    result = permute(result,[3,2,1]);
    target = permute(target,[3,2,1]);
    target = target -1;
    
    
    result(result>=threshold)=1;
    result(result<threshold)=0;

    andOp = (result&target)*2;
    orOp = result|target;
    dice = sum(andOp(:))/sum(orOp(:));
    fprintf(strcat('Iter:',num2str(id),',DICE: %.2f \n'),dice*100);
    
end

