--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 1:03 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'cunn'
require 'cudnn'
require './util'

if not opt then
    opt = {multiGPU = true}
end


-- building block
 function ConvBNReLU(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

 function DeConvBNReLU(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

 function ConvBNReLUStride2(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 2, 2, 2, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

 function DeConvBNReLUStride2(convNet,nInputPlane, nOutputPlane, adjz,adj)
    convNet:add(nn.VolumetricFullConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 2, 2, 2, 1, 1, 1,adjz,adj,adj))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

 function wrapGPU(model, inGPU, outGPU)
     return model;
end


 MaxPooling = nn.VolumetricMaxPooling
 UnPooling = nn.VolumetricMaxUnpooling

-- 2x1x56x512x512
 convGpu1 = nn.Sequential();
ConvBNReLUStride2(convGpu1,1,64):add(nn.Dropout(0.3));
convGpu1 = wrapGPU(convGpu1,1);
-- 2x64x28x256x256
 convGpu1_2 = nn.Sequential();
ConvBNReLUStride2(convGpu1_2,64,64);
convGpu1_2 = wrapGPU(convGpu1_2,3);
-- 2x64x14x128x128
 mp1 = MaxPooling(2, 2, 2, 2, 2, 2);
mp1 = wrapGPU(mp1,4);

-- 2x64x7x64x64
 convGpu2 = nn.Sequential();
ConvBNReLU(convGpu2,64, 128):add(nn.Dropout(0.4))
ConvBNReLU(convGpu2,128, 128)
convGpu2 = wrapGPU(convGpu2,3)
-- 2x128x7x64x64
 mp2 = MaxPooling(2, 2, 2, 2, 2, 2);
mp2 = wrapGPU(mp2,4)

-- 2x128x3x32x32

 convGpu2_2 = nn.Sequential();
ConvBNReLU(convGpu2_2, 128, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256):add(nn.Dropout(0.4));
ConvBNReLU(convGpu2_2, 256, 256);
convGpu2_2 = wrapGPU(convGpu2_2,3)

 mp3 = MaxPooling(1, 2, 2, 1, 2, 2);
mp3 = wrapGPU(mp3,4)

-- 2x256x3x16x16
 convGpu3 = nn.Sequential();
ConvBNReLU(convGpu3, 256, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512):add(nn.Dropout(0.4));
ConvBNReLU(convGpu3, 512, 512);
convGpu3 = wrapGPU(convGpu3,4);

 mp4 = MaxPooling(1, 2, 2, 1, 2, 2);
mp4 = wrapGPU(mp4,4)
-- 2x512x3x8x8

 convGpu3_2 = nn.Sequential();
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512):add(nn.Dropout(0.4))
ConvBNReLU(convGpu3_2, 512, 512)
convGpu3_2 = wrapGPU(convGpu3_2,4)
--
 mp5 = MaxPooling(1, 2, 2, 1, 2, 2);
mp5 = wrapGPU(mp5,4);


fcnn = nn.Sequential();
fcnn:add(convGpu1):add(convGpu1_2):add(mp1):add(convGpu2):add(mp2):add(convGpu2_2):add(mp3);
fcnn:add(convGpu3):add(mp4):add(convGpu3_2):add(mp5);


-- 2x512x3x4x4
----------------------
--MaxPooling = nn.VolumetricMaxPooling
--UnPooling = nn.VolumetricMaxUnpooling
--
--model = nn.Sequential();
--mp1 = MaxPooling(2, 2, 2, 2, 2, 2)
--
--cat= nn.ConcatTable();
--cat:add(mp1);
--cat:add(nn.Identity())
--model:add(cat);
--
--
--mp2 = MaxPooling(2,2,2,2,2,2);
--
--cat1 = nn.ConcatTable();
--cat1:add(mp2);
--cat1:add(nn.Identity());
--
--para1 = nn.ParallelTable();
--para1:add(cat1)
--para1:add(nn.Identity());
--model:add(para1)
--
--model:add(nn.FlattenTable());
--
--
--mp3 = MaxPooling(2,2,2,2,2,2);
--model:add(mp3);
--
--
--mp2 = MaxPooling(2,2,2,2,2,2);
--concatT1 = nn.ConcatTable();
--concatT1:add(mp2);
--concatT1:add(nn.Identity());
--model:add(concatT1);
--
--para1 = nn.ParallelTable()
--para1:add(UnPooling(mp2));
--para1:add(nn.Identity());
--model:add(para1);
--
--mp3 = MaxPooling(2,2,2,2,2,2);
--concatT3 = nn.ConcatTable();
--concatT3:add(mp3);
--concatT3:add(nn.Identity());
--model:add(concatT3);
--
--para1 = nn.ParallelTable()
--para1:add(UnPooling(mp3));
--para1:add(nn.Identity());
--para1:add(nn.Identity());
--model:add(para1);





--------------------

convGpu4 = nn.Sequential();
up1 = UnPooling(mp5);
convGpu4:add(up1);
cat1 = nn.ConcatTable();
cat1:add(convGpu4);
cat1:add(nn.Identity());

fcnn:add(cat1);

deGroup1 = nn.Sequential();
DeConvBNReLU(deGroup1,512,512);
up2 = UnPooling(mp4);
deGroup1:add(up2);
cat2 = nn.ConcatTable();
cat2:add(deGroup1);
cat2:add(nn.Identity());
convGpu4:add(cat2);

deGroup2 = nn.Sequential();
DeConvBNReLU(deGroup2,512,256);
up3 = UnPooling(mp3);
deGroup2:add(up3);
cat3 = nn.ConcatTable();
cat3:add(deGroup2);
cat3:add(nn.Identity());
deGroup1:add(cat3);

deGroup3 = nn.Sequential();
DeConvBNReLU(deGroup3,256,128);
deGroup3:add(UnPooling(mp2));
cat4 = nn.ConcatTable();
cat4:add(deGroup3);
cat4:add(nn.Identity())
deGroup2:add(cat4);

deGroup4 = nn.Sequential();
DeConvBNReLU(deGroup4,128,64);
deGroup4:add(UnPooling(mp1));
cat5 = nn.ConcatTable();
cat5:add(deGroup4);
cat5:add(nn.Identity());
deGroup3:add(cat5);

deGroup5 = nn.Sequential();
DeConvBNReLUStride2(deGroup5,64,64,0,0);
cat6 = nn.ConcatTable();
cat6:add(deGroup5);
cat6:add(nn.Identity());
deGroup4:add(cat6);

deGroup6 = nn.Sequential();
DeConvBNReLUStride2(deGroup6,64,2,1,1);
cat7 = nn.ConcatTable();
cat7:add(deGroup6);
cat7:add(nn.Identity());
deGroup5:add(cat7);

fcnn:add(nn.FlattenTable())

fcnn:add(nn.NarrowTable(4,5));

para1 = nn.ParallelTable();
fitTensor1 = nn.VolumetricReplicationPadding(0, 63-3, 0, 63-3, 0, 6-3);
fitTensor2 = nn.VolumetricReplicationPadding(0, 63-7, 0, 63-7, 0, 6-3);
fitTensor3 = nn.VolumetricReplicationPadding(0, 63-15, 0, 63-15, 0, 6-3);
fitTensor4 = nn.VolumetricReplicationPadding(0, 63-31, 0, 63-31, 0, 6-3);
fitTensor5 = nn.Identity();
para1:add(fitTensor5);
para1:add(fitTensor4);
para1:add(fitTensor3);
para1:add(fitTensor2);
para1:add(fitTensor1);

fcnn:add(para1);


--1 : DoubleTensor - size: 1x2x56x512x512
--2 : DoubleTensor - size: 1x64x27x255x255
--3 : DoubleTensor - size: 1x64x13x127x127
--4 : DoubleTensor - size: 1x128x6x63x63
--5 : DoubleTensor - size: 1x256x3x31x31
--6 : DoubleTensor - size: 1x512x3x15x15
--7 : DoubleTensor - size: 1x512x3x7x7
--8 : DoubleTensor - size: 1x512x3x3x3
--deep mask
-- juanji



------
--target = torch.CudaTensor(1,2,56,512,512):fill(0);
--fcnn:backward(m,target);
--collectgarbage(); getMemStats()


-- load model parameters from already trained model for the first cnn part
 modelPath = 'training/25^3_i9_r0.05/model_2.net'
 vggmodel = torch.load(modelPath);
vggmodel:remove(52);
vggmodel:remove(51);
 p,gp = vggmodel:parameters();


 fp,fgp = fcnn:parameters();

for i=1,#p do
     origParameters = p[i];
    fp[i]:copy(origParameters);
end


collectgarbage();
--fcnn = fcnn:cuda();
--m = fcnn:forward(torch.CudaTensor(1,1,56,512,512));
--target = torch.CudaTensor(1,2,56,512,512):fill(0);
--fcnn:backward(m,target);
collectgarbage(); getMemStats()


return fcnn