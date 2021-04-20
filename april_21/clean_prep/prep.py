import numpy as np

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import gc



import argparse
#import ast

parser = argparse.ArgumentParser(description="Perform data preprocessing")
parser.add_argument("default", type=float, help="Default value relative to the minimum of the distribution, with positive sign")
args = parser.parse_args()

default = args.default
if int(default) == default:
    default = int(default)
minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')
defaults = minima - default



# this will have all starts from 0 to (including) 2400
starts = np.arange(0,2450,50)
# this will have all ends from 49 to 2399 as well as 2446 (this was the number of original .root-files)
ends = np.concatenate((np.arange(49,2449,50), np.arange(2446,2447)))             



#print(starts)
#print(ends)
NUM_DATASETS = len(starts)
print(NUM_DATASETS)


# TT to Semileptonic
dataset_paths = [f'/hpcwork/um106329/april_21/cleaned_TT/inputs_{starts[k]}_to_{ends[k]}_with_default_{default}.npy' for k in range(0, NUM_DATASETS)]
DeepCSV_paths = [f'/hpcwork/um106329/april_21/cleaned_TT/deepcsv_{starts[k]}_to_{ends[k]}_with_default_{default}.npy' for k in range(0, NUM_DATASETS)]




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
    
    if default == 999:
        for i in range(0,67): # do not apply scaling to default values, which were set to -999
            scaler = StandardScaler().fit(train_inputs[:,i][train_inputs[:,i]!=-999].reshape(-1,1))
            norm_train_inputs[:,i][train_inputs[:,i]!=-999]   = torch.Tensor(scaler.transform(train_inputs[:,i][train_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
            norm_val_inputs[:,i][val_inputs[:,i]!=-999]	  = torch.Tensor(scaler.transform(val_inputs[:,i][val_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
            norm_test_inputs[:,i][test_inputs[:,i]!=-999]     = torch.Tensor(scaler.transform(test_inputs[:,i][test_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
            scalers.append(scaler)
    else:
        for i in range(0,67): # do not apply scaling to default values, which were set to -999
            scaler = StandardScaler().fit(train_inputs[:,i][train_inputs[:,i]!=defaults[i]].reshape(-1,1))
            norm_train_inputs[:,i]   = torch.Tensor(scaler.transform(train_inputs[:,i].reshape(-1,1)).reshape(1,-1))
            norm_val_inputs[:,i]       = torch.Tensor(scaler.transform(val_inputs[:,i].reshape(-1,1)).reshape(1,-1))
            norm_test_inputs[:,i]     = torch.Tensor(scaler.transform(test_inputs[:,i].reshape(-1,1)).reshape(1,-1))
            scalers.append(scaler)
    
    
    train_inputs = norm_train_inputs.clone().detach().to(torch.float16)
    val_inputs = norm_val_inputs.clone().detach().to(torch.float16)
    test_inputs = norm_test_inputs.clone().detach().to(torch.float16)
    
        
    # TT to Semileptonic
    torch.save(train_inputs, f'/hpcwork/um106329/april_21/scaled_TT/train_inputs_%d_with_default_{default}.pt' % s)
    torch.save(val_inputs, f'/hpcwork/um106329/april_21/scaled_TT/val_inputs_%d_with_default_{default}.pt' % s)
    torch.save(test_inputs, f'/hpcwork/um106329/april_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % s)
    torch.save(DeepCSV_testset, f'/hpcwork/um106329/april_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % s)
    torch.save(train_targets, f'/hpcwork/um106329/april_21/scaled_TT/train_targets_%d_with_default_{default}.pt' % s)
    torch.save(val_targets, f'/hpcwork/um106329/april_21/scaled_TT/val_targets_%d_with_default_{default}.pt' % s)
    torch.save(test_targets, f'/hpcwork/um106329/april_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % s)
    torch.save(scalers, f'/hpcwork/um106329/april_21/scaled_TT/scalers_%d_with_default_{default}.pt' % s)
    
    
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
    
    
for s in range(NUM_DATASETS): #range(1,49):
    preprocess(np.load(dataset_paths[s]), np.load(DeepCSV_paths[s]), s)

