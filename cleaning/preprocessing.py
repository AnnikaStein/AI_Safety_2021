import numpy as np

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#from sys import getsizeof

#import time
import gc



NUM_DATASETS = 200

dataset_paths = ['/hpcwork/um106329/cleaned/biggerdataset_%d.npy' % k for k in range(0, NUM_DATASETS)]
DeepCSV_paths = ['/hpcwork/um106329/cleaned/biggerDeepCSV_dataset_%d.npy' % k for k in range(0, NUM_DATASETS)]
   
# preprocess the datasets, create train, val, test + DeepCSV
def preprocess(dataset, DeepCSV_dataset, s):    
    
    trainingset,testset,_,DeepCSV_testset = train_test_split(dataset, DeepCSV_dataset, test_size=0.2, random_state=1)
    trainset, valset = train_test_split(trainingset,test_size=0.1, random_state=1)
    
    
    test_inputs = torch.Tensor(testset[:,0:67])                                                
    test_targets = (torch.Tensor([testset[i][-1] for i in range(len(testset))])).long()        
    val_inputs = torch.Tensor(valset[:,0:67])
    val_targets = (torch.Tensor([valset[i][-1] for i in range(len(valset))]).long())
    train_inputs = torch.Tensor(trainset[:,0:67])
    train_targets = (torch.Tensor([trainset[i][-1] for i in range(len(trainset))])).long()
    
      
    
    norm_train_inputs,norm_val_inputs,norm_test_inputs = train_inputs.clone().detach(),val_inputs.clone().detach(),test_inputs.clone().detach()
    scalers = []
    
    for i in range(0,67):
        scaler = StandardScaler().fit(train_inputs[:,i].reshape(-1,1))
        norm_train_inputs[:,i] = torch.Tensor(scaler.transform(train_inputs[:,i].reshape(-1,1)).reshape(1,-1)).to(torch.float16)
        norm_val_inputs[:,i] = torch.Tensor(scaler.transform(val_inputs[:,i].reshape(-1,1)).reshape(1,-1)).to(torch.float16)
        norm_test_inputs[:,i] = torch.Tensor(scaler.transform(test_inputs[:,i].reshape(-1,1)).reshape(1,-1)).to(torch.float16)
        scalers.append(scaler)

    norm_trainset = [[norm_train_inputs[i].to(torch.float16),train_targets[i]] for i in range(len(trainset))]
    
    
    #use normalized values in following code
    #raw_train_inputs = train_inputs.clone().detach().to(torch.float16)
    #raw_val_inputs = val_inputs.clone().detach().to(torch.float16)
    #raw_test_inputs = test_inputs.clone().detach().to(torch.float16)
    #print(len(torch.cat((raw_train_inputs,raw_val_inputs,raw_test_inputs))))

    train_inputs = norm_train_inputs.clone().detach().to(torch.float16)
    val_inputs = norm_val_inputs.clone().detach().to(torch.float16)
    test_inputs = norm_test_inputs.clone().detach().to(torch.float16)
    trainset = norm_trainset.copy()
    
    valset = [[val_inputs[i].to(torch.float16),val_targets[i]] for k in range(len(val_inputs))]
    
    
    torch.save(train_inputs, '/work/um106329/MA/cleaned/preprocessed/train_inputs_%d.pt' % s)
    torch.save(val_inputs, '/work/um106329/MA/cleaned/preprocessed/val_inputs_%d.pt' % s)
    torch.save(test_inputs, '/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % s)
    torch.save(trainset, '/work/um106329/MA/cleaned/preprocessed/trainset_%d.pt' % s)
    torch.save(testset, '/work/um106329/MA/cleaned/preprocessed/testset_%d.pt' % s)
    torch.save(valset, '/work/um106329/MA/cleaned/preprocessed/valset_%d.pt' % s)
    torch.save(DeepCSV_testset, '/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % s)
    torch.save(train_targets, '/work/um106329/MA/cleaned/preprocessed/train_targets_%d.pt' % s)
    torch.save(val_targets, '/work/um106329/MA/cleaned/preprocessed/val_targets_%d.pt' % s)
    torch.save(test_targets, '/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % s)
    torch.save(scalers, '/work/um106329/MA/cleaned/preprocessed/scalers_%d.pt' % s)
    del train_inputs
    del val_inputs
    del test_inputs
    del trainset
    del trainingset
    del testset
    del valset
    del DeepCSV_testset
    del train_targets
    del val_targets
    del test_targets
    del scaler
    del scalers
    gc.collect()    
    
    
for s in range(188,NUM_DATASETS):
    preprocess(np.load(dataset_paths[s]), np.load(DeepCSV_paths[s]), s)

