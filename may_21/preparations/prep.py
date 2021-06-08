import numpy as np

import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import binned_statistic_2d

import gc



import argparse
#import ast

parser = argparse.ArgumentParser(description="Perform data preprocessing")
parser.add_argument("default", type=float, help="Default value relative to the minimum of the distribution, with positive sign")
parser.add_argument("sample", help="TT or QCD samples")
args = parser.parse_args()

sample = args.sample
default = args.default
if int(default) == default:
    default = int(default)
minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')
defaults = minima - default

# if one wants to change pt-eta such that in each bin, there is an equal amount of entries from every flavour (target: average per bin over the four flavours), but keeping
# the average distribution (more low-pt than high-pt, more small eta than large eta and such)

b_weights = np.load('/home/um106329/aisafety/may_21/absweights_b.npy')
bb_weights = np.load('/home/um106329/aisafety/may_21/absweights_bb.npy')
c_weights = np.load('/home/um106329/aisafety/may_21/absweights_c.npy')
l_weights = np.load('/home/um106329/aisafety/may_21/absweights_l.npy')

flavour_lookuptables = np.array([b_weights,bb_weights,c_weights,l_weights])


# if one wants to use flat distributions (target for weighting is average over the whole pt-eta-histogram per flavour), multiplied by class imbalance, leads to rectangular
# shapes, naturally (almost) identical between the different flavours ('almost' because not every bin is filled for large eta / pt for every flavour)

b_weights_flat = np.load('/home/um106329/aisafety/may_21/weights_flat_b.npy')
bb_weights_flat = np.load('/home/um106329/aisafety/may_21/weights_flat_bb.npy')
c_weights_flat = np.load('/home/um106329/aisafety/may_21/weights_flat_c.npy')
l_weights_flat = np.load('/home/um106329/aisafety/may_21/weights_flat_l.npy')

flavour_lookuptables_flat = np.array([b_weights_flat,bb_weights_flat,c_weights_flat,l_weights_flat])

if sample=='TT':

    # this will have all starts from 0 to (including) 2400
    starts = np.arange(0,2450,50)
    # this will have all ends from 49 to 2399 as well as 2446 (this was the number of original .root-files)
    ends = np.concatenate((np.arange(49,2449,50), np.arange(2446,2447)))             



    #print(starts)
    #print(ends)
    NUM_DATASETS = len(starts)
    print(NUM_DATASETS)


    # TT to Semileptonic
    dataset_paths = [f'/hpcwork/um106329/may_21/cleaned_TT/inputs_{starts[k]}_to_{ends[k]}_with_default_{default}.npy' for k in range(0, NUM_DATASETS)]
    DeepCSV_paths = [f'/hpcwork/um106329/may_21/cleaned_TT/deepcsv_{starts[k]}_to_{ends[k]}_with_default_{default}.npy' for k in range(0, NUM_DATASETS)]

else:

    qcdstarts = np.arange(0,11450,50)
    # this will have all ends from 49 to 2399 as well as 2446 (this was the number of original .root-files)
    qcdends = np.concatenate((np.arange(49,11449,50), np.arange(11407,11408)))             



    #print(starts)
    #print(ends)
    NUM_DATASETS = len(qcdstarts)
    print(NUM_DATASETS)


    # QCD
    dataset_paths = [f'/hpcwork/um106329/may_21/cleaned_QCD/inputs_{qcdstarts[k]}_to_{qcdends[k]}_with_default_{default}.npy' for k in range(0, NUM_DATASETS)]
    DeepCSV_paths = [f'/hpcwork/um106329/may_21/cleaned_QCD/deepcsv_{qcdstarts[k]}_to_{qcdends[k]}_with_default_{default}.npy' for k in range(0, NUM_DATASETS)]





# preprocess the datasets, create train, val, test + DeepCSV
def preprocess(dataset, DeepCSV_dataset, s):    
    
    trainingset,testset,_,DeepCSV_testset = train_test_split(dataset, DeepCSV_dataset, test_size=0.2, random_state=1)
    torch.save(DeepCSV_testset, f'/hpcwork/um106329/may_21/scaled_{sample}/DeepCSV_testset_%d_with_default_{default}.pt' % s)
    del DeepCSV_testset
    trainset, valset = train_test_split(trainingset,test_size=0.1, random_state=1)
    del trainingset
    gc.collect()
    # get the indices of the binned 2d histogram (eta, pt) for each jet
    # these arrays will have the shape (2,len(data)) where len(data) is the length of the testset, valset and trainset
    # to not waste too much memory & diskspace later, one only needs 8-bit unsigned integer (each going from 0 to 255, which is enough for 50 bins in each direction,
    # so only 50 possible values --> use np.ubyte directly, also one only needs to unpack the fourth return value from binned_statistic_2d, we don't need the histogram
    # or the bin edges, just the indices that will serve as a kind of look-up-table during the sampling for the training)
    # first sub-array are the indices for eta, second one for pt (notice: this is really a nested array because expand_binnumbers was set to true, otherwise it would have been flat)                                                
    test_targets = (torch.Tensor(testset[:,-1])).long()      
    _,_,_,test_pt_eta_bins = binned_statistic_2d(testset[:,0],testset[:,1],None,'count',bins=(50,50),range=((-2.5,2.5),(20,1000)),expand_binnumbers=True)
    test_eta_bins = test_pt_eta_bins[0]-1
    test_pt_bins = test_pt_eta_bins[1]-1
    test_all_weights = flavour_lookuptables[test_targets,test_eta_bins,test_pt_bins]
    test_weights = test_all_weights/sum(test_all_weights)
    test_all_weights_flat = flavour_lookuptables_flat[test_targets,test_eta_bins,test_pt_bins]
    test_weights_flat = test_all_weights_flat/sum(test_all_weights_flat)
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/test_pt_eta_bins_%d_with_default_{default}.npy' % s,test_pt_eta_bins.astype(np.ubyte))
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/test_sample_weights_%d_with_default_{default}.npy' % s,test_weights)
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/test_sample_weights_flat_%d_with_default_{default}.npy' % s,test_weights_flat)
    del test_pt_eta_bins
    del test_eta_bins
    del test_pt_bins
    del test_all_weights
    del test_weights
    gc.collect()
    
    val_targets = (torch.Tensor(valset[:,-1])).long()
    _,_,_,val_pt_eta_bins = binned_statistic_2d(valset[:,0],valset[:,1],None,'count',bins=(50,50),range=((-2.5,2.5),(20,1000)),expand_binnumbers=True)
    val_eta_bins = val_pt_eta_bins[0]-1
    val_pt_bins = val_pt_eta_bins[1]-1
    val_all_weights = flavour_lookuptables[val_targets,val_eta_bins,val_pt_bins]
    val_weights = val_all_weights/sum(val_all_weights)
    val_all_weights_flat = flavour_lookuptables_flat[val_targets,val_eta_bins,val_pt_bins]
    val_weights_flat = val_all_weights_flat/sum(val_all_weights_flat)
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/val_pt_eta_bins_%d_with_default_{default}.npy' % s,val_pt_eta_bins.astype(np.ubyte))
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/val_sample_weights_%d_with_default_{default}.npy' % s,val_weights)
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/val_sample_weights_flat_%d_with_default_{default}.npy' % s,val_weights_flat)
    del val_pt_eta_bins
    del val_eta_bins
    del val_pt_bins
    del val_all_weights
    del val_weights
    gc.collect()
    
    train_targets = (torch.Tensor(trainset[:,-1])).long()
    _,_,_,train_pt_eta_bins = binned_statistic_2d(trainset[:,0],trainset[:,1],None,'count',bins=(50,50),range=((-2.5,2.5),(20,1000)),expand_binnumbers=True)
    train_eta_bins = train_pt_eta_bins[0]-1
    train_pt_bins = train_pt_eta_bins[1]-1
    train_all_weights = flavour_lookuptables[train_targets,train_eta_bins,train_pt_bins]
    train_weights = train_all_weights/sum(train_all_weights)
    train_all_weights_flat = flavour_lookuptables_flat[train_targets,train_eta_bins,train_pt_bins]
    train_weights_flat = train_all_weights_flat/sum(train_all_weights_flat)
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/train_pt_eta_bins_%d_with_default_{default}.npy' % s,train_pt_eta_bins.astype(np.ubyte))
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/train_sample_weights_%d_with_default_{default}.npy' % s,train_weights)
    np.save(f'/hpcwork/um106329/may_21/scaled_{sample}/train_sample_weights_flat_%d_with_default_{default}.npy' % s,train_weights_flat)
    del train_pt_eta_bins
    del train_eta_bins
    del train_pt_bins
    del train_all_weights
    del train_weights
    gc.collect()
    # the indices have been retrieved before the scaling happened (because afterwards, the values will be different and not be placed in the bins defined during the calculation
    # of the weights)
    
    test_inputs = torch.Tensor(testset[:,0:67])                                                
    #test_targets = (torch.Tensor(testset[:,-1])).long()        
    val_inputs = torch.Tensor(valset[:,0:67])
    #val_targets = (torch.Tensor(valset[:,-1])).long()
    train_inputs = torch.Tensor(trainset[:,0:67])
    #train_targets = (torch.Tensor(trainset[:,-1])).long()
    
    norm_train_inputs,norm_val_inputs,norm_test_inputs = train_inputs.clone().detach(),val_inputs.clone().detach(),test_inputs.clone().detach()
    scalers = []
    
    # scalers are computed without defaulted values, but applied to all values
    if default == 999:
        for i in range(0,67): # do not compute scalers with default values, which were set to -999
            scaler = StandardScaler().fit(train_inputs[:,i][train_inputs[:,i]!=-999].reshape(-1,1))
            norm_train_inputs[:,i][train_inputs[:,i]!=-999]   = torch.Tensor(scaler.transform(train_inputs[:,i][train_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
            norm_val_inputs[:,i][val_inputs[:,i]!=-999]	  = torch.Tensor(scaler.transform(val_inputs[:,i][val_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
            norm_test_inputs[:,i][test_inputs[:,i]!=-999]     = torch.Tensor(scaler.transform(test_inputs[:,i][test_inputs[:,i]!=-999].reshape(-1,1)).reshape(1,-1))
            scalers.append(scaler)
    else:
        for i in range(0,67): # do not compute scalers with default values, which were set to minima-default
            scaler = StandardScaler().fit(train_inputs[:,i][train_inputs[:,i]!=defaults[i]].reshape(-1,1))
            norm_train_inputs[:,i]   = torch.Tensor(scaler.transform(train_inputs[:,i].reshape(-1,1)).reshape(1,-1))
            norm_val_inputs[:,i]       = torch.Tensor(scaler.transform(val_inputs[:,i].reshape(-1,1)).reshape(1,-1))
            norm_test_inputs[:,i]     = torch.Tensor(scaler.transform(test_inputs[:,i].reshape(-1,1)).reshape(1,-1))
            scalers.append(scaler)
    
    
    train_inputs = norm_train_inputs.clone().detach().to(torch.float16)
    val_inputs = norm_val_inputs.clone().detach().to(torch.float16)
    test_inputs = norm_test_inputs.clone().detach().to(torch.float16)
    
    
    torch.save(train_inputs, f'/hpcwork/um106329/may_21/scaled_{sample}/train_inputs_%d_with_default_{default}.pt' % s)
    torch.save(val_inputs, f'/hpcwork/um106329/may_21/scaled_{sample}/val_inputs_%d_with_default_{default}.pt' % s)
    torch.save(test_inputs, f'/hpcwork/um106329/may_21/scaled_{sample}/test_inputs_%d_with_default_{default}.pt' % s)
    torch.save(train_targets, f'/hpcwork/um106329/may_21/scaled_{sample}/train_targets_%d_with_default_{default}.pt' % s)
    torch.save(val_targets, f'/hpcwork/um106329/may_21/scaled_{sample}/val_targets_%d_with_default_{default}.pt' % s)
    torch.save(test_targets, f'/hpcwork/um106329/may_21/scaled_{sample}/test_targets_%d_with_default_{default}.pt' % s)
    torch.save(scalers, f'/hpcwork/um106329/may_21/scaled_{sample}/scalers_%d_with_default_{default}.pt' % s)
    
    del train_inputs
    del val_inputs
    del test_inputs
    del train_targets
    del val_targets
    del test_targets
    del scaler
    del scalers
    del trainset
    del testset
    del valset
    gc.collect()    
    
    
for s in range(NUM_DATASETS): #range(1,49):
    preprocess(np.load(dataset_paths[s]), np.load(DeepCSV_paths[s]), s)

