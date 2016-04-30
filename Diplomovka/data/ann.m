 nHiddenNeurons = 15;

trainingData = dlmread('training-martin.data', ' ');
testingData = dlmread('testing-martin.data', ' ');
[nRows, nCols] = size(trainingData);

 mOutput = trainingData(:,(nCols - 1):end);
 mInput = trainingData(:, 1:end-2);
 
 mData = [mInput mOutput];
 
 # ~50% mTrain ~33%mTest ~17%mVali
 [mTrain, mTest, mVali] = subset(mData',1);
 mMinMaxElements = min_max(mTrain);

 
 nOutputNeurons = 2;
 
 MLPnet = newff(mMinMaxElements, [nHiddenNeurons nOutputNeurons]);

 # define validation data
 VV.P = mVali(1:end-2,:);
 VV.T = mVali((end-1):end,:);
 
  mTrainOutput = mTrain((nCols - 1):end,:);
 mTrainInput = mTrain(1:end-2,:);
 
[net] = train(MLPnet,  mTrainInput', mTrainOutput', [], [], VV);

