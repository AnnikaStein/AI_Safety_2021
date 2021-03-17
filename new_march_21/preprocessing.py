import numpy as np

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import gc



# this will have all starts from 0 to (including) 11400
starts = np.arange(0,11450,50)
# this will have all ends from 49 to 11399 as well as 11407 (this was the number of original .root-files)
ends = np.concatenate((np.arange(49,11449,50), np.arange(11407,11408)))             

#print(starts)
#print(ends)
NUM_DATASETS = len(starts)
print(NUM_DATASETS)

dataset_paths = [f'/hpcwork/um106329/new_march_21/cleaned/inputs_{starts[k]}_to_{ends[k]}.npy' for k in range(0, NUM_DATASETS)]
DeepCSV_paths = [f'/hpcwork/um106329/new_march_21/cleaned/deepcsv_{starts[k]}_to_{ends[k]}.npy' for k in range(0, NUM_DATASETS)]




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
    
    for i in range(0,67): # do not apply scaling to default values, which were set to -999
        scaler = StandardScaler().fit(train_inputs[:,i][train_inputs[:,i]!=-999].reshape(-1,1))
        norm_train_inputs[:,i][train_inputs[:,i]!=-999]   = torch.Tensor(scaler.transform(train_inputs[:,i][train_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
        norm_val_inputs[:,i][val_inputs[:,i]!=-999]       = torch.Tensor(scaler.transform(val_inputs[:,i][val_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
        norm_test_inputs[:,i][test_inputs[:,i]!=-999]     = torch.Tensor(scaler.transform(test_inputs[:,i][test_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
        scalers.append(scaler)
    
    
    train_inputs = norm_train_inputs.clone().detach().to(torch.float16)
    val_inputs = norm_val_inputs.clone().detach().to(torch.float16)
    test_inputs = norm_test_inputs.clone().detach().to(torch.float16)
    
        
    torch.save(train_inputs, '/hpcwork/um106329/new_march_21/scaled/train_inputs_%d.pt' % s)
    torch.save(val_inputs, '/hpcwork/um106329/new_march_21/scaled/val_inputs_%d.pt' % s)
    torch.save(test_inputs, '/hpcwork/um106329/new_march_21/scaled/test_inputs_%d.pt' % s)
    torch.save(DeepCSV_testset, '/hpcwork/um106329/new_march_21/scaled/DeepCSV_testset_%d.pt' % s)
    torch.save(train_targets, '/hpcwork/um106329/new_march_21/scaled/train_targets_%d.pt' % s)
    torch.save(val_targets, '/hpcwork/um106329/new_march_21/scaled/val_targets_%d.pt' % s)
    torch.save(test_targets, '/hpcwork/um106329/new_march_21/scaled/test_targets_%d.pt' % s)
    torch.save(scalers, '/hpcwork/um106329/new_march_21/scaled/scalers_%d.pt' % s)
    
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
    
    
for s in range(1,121):
    preprocess(np.load(dataset_paths[s]), np.load(DeepCSV_paths[s]), s)

