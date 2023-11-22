%%%%%%% PFA-Net   %%%%%%%%
function [lgraph, ModelName] = fn_PFA_Net(numClasses,params)
disp(strcat('PFA-Net is selected'));
ModelName = 'PFA-Net';
lgraph = layerGraph();

tempLayers = imageInputLayer([304 304 3],"Name","data");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","res2b_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a")
    reluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3b_branch2a.Bias,"Weights",params.res3b_branch2a.Weights)
    batchNormalizationLayer("Name","bn3b_branch2a","Offset",params.bn3b_branch2a.Offset,"Scale",params.bn3b_branch2a.Scale,"TrainedMean",params.bn3b_branch2a.TrainedMean,"TrainedVariance",params.bn3b_branch2a.TrainedVariance)
    reluLayer("Name","res3b_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3b_branch2b.Bias,"Weights",params.res3b_branch2b.Weights)
    batchNormalizationLayer("Name","bn3b_branch2b","Offset",params.bn3b_branch2b.Offset,"Scale",params.bn3b_branch2b.Scale,"TrainedMean",params.bn3b_branch2b.TrainedMean,"TrainedVariance",params.bn3b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3b")
    reluLayer("Name","res3b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a")
    reluLayer("Name","res4a_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res4a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4b_branch2a.Bias,"Weights",params.res4b_branch2a.Weights)
    batchNormalizationLayer("Name","bn4b_branch2a","Offset",params.bn4b_branch2a.Offset,"Scale",params.bn4b_branch2a.Scale,"TrainedMean",params.bn4b_branch2a.TrainedMean,"TrainedVariance",params.bn4b_branch2a.TrainedVariance)
    reluLayer("Name","res4b_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4b_branch2b.Bias,"Weights",params.res4b_branch2b.Weights)
    batchNormalizationLayer("Name","bn4b_branch2b","Offset",params.bn4b_branch2b.Offset,"Scale",params.bn4b_branch2b.Scale,"TrainedMean",params.bn4b_branch2b.TrainedMean,"TrainedVariance",params.bn4b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4b")
    reluLayer("Name","res4b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Padding","same","Bias",params.res5a_branch2a.Bias,"Weights",params.res5a_branch2a.Weights)
    batchNormalizationLayer("Name","bn5a_branch2a","Offset",params.bn5a_branch2a.Offset,"Scale",params.bn5a_branch2a.Scale,"TrainedMean",params.bn5a_branch2a.TrainedMean,"TrainedVariance",params.bn5a_branch2a.TrainedVariance)
    reluLayer("Name","res5a_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same","Bias",params.res5a_branch2b.Bias,"Weights",params.res5a_branch2b.Weights)
    batchNormalizationLayer("Name","bn5a_branch2b","Offset",params.bn5a_branch2b.Offset,"Scale",params.bn5a_branch2b.Scale,"TrainedMean",params.bn5a_branch2b.TrainedMean,"TrainedVariance",params.bn5a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch1","BiasLearnRateFactor",0,"Bias",params.res5a_branch1.Bias,"Weights",params.res5a_branch1.Weights)
    batchNormalizationLayer("Name","bn5a_branch1","Offset",params.bn5a_branch1.Offset,"Scale",params.bn5a_branch1.Scale,"TrainedMean",params.bn5a_branch1.TrainedMean,"TrainedVariance",params.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res5a")
    reluLayer("Name","res5a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same","Bias",params.res5b_branch2a.Bias,"Weights",params.res5b_branch2a.Weights)
    batchNormalizationLayer("Name","bn5b_branch2a","Offset",params.bn5b_branch2a.Offset,"Scale",params.bn5b_branch2a.Scale,"TrainedMean",params.bn5b_branch2a.TrainedMean,"TrainedVariance",params.bn5b_branch2a.TrainedVariance)
    reluLayer("Name","res5b_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"DilationFactor",[2 2],"Padding","same","Bias",params.res5b_branch2b.Bias,"Weights",params.res5b_branch2b.Weights)
    batchNormalizationLayer("Name","bn5b_branch2b","Offset",params.bn5b_branch2b.Offset,"Scale",params.bn5b_branch2b.Scale,"TrainedMean",params.bn5b_branch2b.TrainedMean,"TrainedVariance",params.bn5b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res5b")
    reluLayer("Name","res5b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","new_cat_3")
    convolution2dLayer([1 1],256,"Name","dec_c1","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn1","Offset",params.dec_bn1.Offset,"Scale",params.dec_bn1.Scale,"TrainedMean",params.dec_bn1.TrainedMean,"TrainedVariance",params.dec_bn1.TrainedVariance)
    reluLayer("Name","dec_relu1")
    transposedConv2dLayer([8 8],256,"Name","dec_upsample1","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0,"Bias",params.dec_upsample1.Bias,"Weights",params.dec_upsample1.Weights)];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    depthConcatenationLayer(3,"Name","new_cat_5") 
    convolution2dLayer([3 3],256,"Name","dec_c3","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn3")
    reluLayer("Name","dec_relu3")
    convolution2dLayer([3 3],256,"Name","dec_c4","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10,"Bias",params.dec_c4.Bias,"Weights",params.dec_c4.Weights)];
lgraph = addLayers(lgraph,tempLayers);  %%% connection is cutted here

tempLayers = [
    depthConcatenationLayer(4,"Name","new_cat_4") 
    convolution2dLayer([1 1],256,"Name","new_PConv_3")
    batchNormalizationLayer("Name","dec_bn4","Offset",params.dec_bn4.Offset,"Scale",params.dec_bn4.Scale,"TrainedMean",params.dec_bn4.TrainedMean,"TrainedVariance",params.dec_bn4.TrainedVariance)
    reluLayer("Name","dec_relu4")
    convolution2dLayer([1 1],numClasses,"Name","scorer","BiasLearnRateFactor",0,"WeightLearnRateFactor",10,"Bias",params.scorer.Bias,"Weights",params.scorer.Weights)
    transposedConv2dLayer([8 8],numClasses,"Name","dec_upsample2","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0,"Bias",params.dec_upsample2.Bias,"Weights",params.dec_upsample2.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    crop2dLayer("centercrop","Name","dec_crop2")
    softmaxLayer("Name","softmax-out")
    pixelClassificationLayer("Name","classification")];
lgraph = addLayers(lgraph,tempLayers);

%%%%%%%%%%%%%%%%%%%%%%%% Encoder

                             %%%% PFA-block 1
tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_1","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_1")
    batchNormalizationLayer("Name","new_BatchNorm_1")
    reluLayer("Name","new_Relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_2","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_2")
    batchNormalizationLayer("Name","new_BatchNorm_2")
    reluLayer("Name","new_Relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_3","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_3")
    batchNormalizationLayer("Name","new_BatchNorm_3")
    reluLayer("Name","new_Relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_4","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_4")
    batchNormalizationLayer("Name","new_BatchNorm_4")
    reluLayer("Name","new_Relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","new_cat_1")
    convolution2dLayer([1 1],128,"Name","new_PConv_1") ];
lgraph = addLayers(lgraph,tempLayers);

                             %% PFA block 2
tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_5","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_5")
    batchNormalizationLayer("Name","new_BatchNorm_5")
    reluLayer("Name","new_Relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_6","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_6")
    batchNormalizationLayer("Name","new_BatchNorm_6")
    reluLayer("Name","new_Relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_7","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_7")
    batchNormalizationLayer("Name","new_BatchNorm_7")
    reluLayer("Name","new_Relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_8","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_8")
    batchNormalizationLayer("Name","new_BatchNorm_8")
    reluLayer("Name","new_Relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","new_cat_2")
    convolution2dLayer([1 1],256,"Name","new_PConv_2") ];
lgraph = addLayers(lgraph,tempLayers);
                              %%% upsampling
tempLayers = [
convolution2dLayer([1 1],256,"Name","new_dec_c1")
    batchNormalizationLayer("Name","new_dec_bn1")
    reluLayer("Name","new_dec_relu1")
    transposedConv2dLayer([2 2],256,"Name","new_dec_upsample1","BiasLearnRateFactor",0,"Stride",[2 2],"WeightLearnRateFactor",0)];
lgraph = addLayers(lgraph,tempLayers);

                                %% PFA block 3
tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_9","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_9")
    batchNormalizationLayer("Name","new_BatchNorm_9")
    reluLayer("Name","new_Relu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_10","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_10")
    batchNormalizationLayer("Name","new_BatchNorm_10")
    reluLayer("Name","new_Relu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_11","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_11")
    batchNormalizationLayer("Name","new_BatchNorm_11")
    reluLayer("Name","new_Relu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_12","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_12")
    batchNormalizationLayer("Name","new_BatchNorm_12")
    reluLayer("Name","new_Relu_12")];
lgraph = addLayers(lgraph,tempLayers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Decoder

                          %%PFA block 4
tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_13","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_13")
    batchNormalizationLayer("Name","new_BatchNorm_13")
    reluLayer("Name","new_Relu_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_14","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_14")
    batchNormalizationLayer("Name","new_BatchNorm_14")
    reluLayer("Name","new_Relu_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_15","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_15")
    batchNormalizationLayer("Name","new_BatchNorm_15")
    reluLayer("Name","new_Relu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],320,"Name","new_Conv_16","Padding","same")
    convolution2dLayer([1 1],256,"Name","new_proj_Conv_16")
    batchNormalizationLayer("Name","new_BatchNorm_16")
    reluLayer("Name","new_Relu_16")];
lgraph = addLayers(lgraph,tempLayers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"data","conv1");
lgraph = connectLayers(lgraph,"data","dec_crop2/ref");
lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1","res2a/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2b","res2b/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
lgraph = connectLayers(lgraph,"res3a_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"res3a_relu","res3b/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2b","res3b/in1");
lgraph = connectLayers(lgraph,"bn4a_branch2b","res4a/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");
lgraph = connectLayers(lgraph,"res4a_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"res4a_relu","res4b/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2b","res4b/in1");
lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch2b","res5a/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","res5a/in2");
lgraph = connectLayers(lgraph,"res5a_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"res5a_relu","res5b/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2b","res5b/in1");

%%%%%   Encoder side
                            %% PFA block 1
lgraph = connectLayers(lgraph,"res2b_relu","new_Conv_1");
lgraph = connectLayers(lgraph,"res2b_relu","new_Conv_2");
lgraph = connectLayers(lgraph,"res2b_relu","new_Conv_3");
lgraph = connectLayers(lgraph,"res2b_relu","new_Conv_4");
lgraph = connectLayers(lgraph,"new_Relu_1","new_cat_1/in1");
lgraph = connectLayers(lgraph,"new_Relu_2","new_cat_1/in2");
lgraph = connectLayers(lgraph,"new_Relu_3","new_cat_1/in3");
lgraph = connectLayers(lgraph,"new_Relu_4","new_cat_1/in4");
lgraph = connectLayers(lgraph,"new_PConv_1","res3a_branch2a");
lgraph = connectLayers(lgraph,"new_PConv_1","res3a_branch1");
% lgraph = connectLayers(lgraph,"new_PConv_1","dec_c2");
                                 %% PFA Block 2
lgraph = connectLayers(lgraph,"res3b_relu","new_Conv_5");
lgraph = connectLayers(lgraph,"res3b_relu","new_Conv_6");
lgraph = connectLayers(lgraph,"res3b_relu","new_Conv_7");
lgraph = connectLayers(lgraph,"res3b_relu","new_Conv_8");
lgraph = connectLayers(lgraph,"new_Relu_5","new_cat_2/in1");
lgraph = connectLayers(lgraph,"new_Relu_6","new_cat_2/in2");
lgraph = connectLayers(lgraph,"new_Relu_7","new_cat_2/in3");
lgraph = connectLayers(lgraph,"new_Relu_8","new_cat_2/in4");
lgraph = connectLayers(lgraph,"new_PConv_2","res4a_branch2a");
lgraph = connectLayers(lgraph,"new_PConv_2","res4a_branch1");
lgraph = connectLayers(lgraph,"new_PConv_2","new_dec_c1");
                                   %% PFA Block 3
lgraph = connectLayers(lgraph,"res5b_relu","new_Conv_9");
lgraph = connectLayers(lgraph,"res5b_relu","new_Conv_10");
lgraph = connectLayers(lgraph,"res5b_relu","new_Conv_11");
lgraph = connectLayers(lgraph,"res5b_relu","new_Conv_12");
lgraph = connectLayers(lgraph,"new_Relu_9","new_cat_3/in1");
lgraph = connectLayers(lgraph,"new_Relu_10","new_cat_3/in2");
lgraph = connectLayers(lgraph,"new_Relu_11","new_cat_3/in3");
lgraph = connectLayers(lgraph,"new_Relu_12","new_cat_3/in4");

%%%%%   Decoder side
                                %% PFA Block 4

lgraph = connectLayers(lgraph,"dec_c4","new_Conv_13");
lgraph = connectLayers(lgraph,"dec_c4","new_Conv_14");
lgraph = connectLayers(lgraph,"dec_c4","new_Conv_15");
lgraph = connectLayers(lgraph,"dec_c4","new_Conv_16");
lgraph = connectLayers(lgraph,"new_Relu_13","new_cat_4/in1");
lgraph = connectLayers(lgraph,"new_Relu_14","new_cat_4/in2");
lgraph = connectLayers(lgraph,"new_Relu_15","new_cat_4/in3");
lgraph = connectLayers(lgraph,"new_Relu_16","new_cat_4/in4");

lgraph = connectLayers(lgraph,"dec_upsample1","new_cat_5/in1");
lgraph = connectLayers(lgraph,"new_dec_upsample1","new_cat_5/in2");
lgraph = connectLayers(lgraph,"new_PConv_1","new_cat_5/in3");
end
