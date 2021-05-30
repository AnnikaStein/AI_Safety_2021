import numpy as np

import gc
#import argparse
#import ast

#parser = argparse.ArgumentParser(description="Perform additional data cleaning")
#parser.add_argument("default", type=float, help="Default value relative to the minimum of the distribution, with positive sign")
#args = parser.parse_args()

#default = args.default
default = 0.001
if int(default) == default:
    default = int(default)
minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')
defaults = minima - default


'''
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
'''




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




# File-loading and Cleaning
def cleandataset(data, DeepCSV):
    TrackSumJetDeltaR_cut = data[:,57] <= 0.3
    data = data[TrackSumJetDeltaR_cut]
    DeepCSV = DeepCSV[TrackSumJetDeltaR_cut]
    
    return data, DeepCSV


for n in range(NUM_DATASETS):
    biggerdataset, biggerDeepCSV_dataset = np.load(dataset_paths[n]), np.load(DeepCSV_paths[n])
    lenbiggerdata = len(biggerdataset)
    print(f'Length dataset {n} before deleting large / default values: {lenbiggerdata}')
    
    dataset, DeepCSV_dataset = cleandataset(biggerdataset, biggerDeepCSV_dataset)
    lendata = len(dataset)
    print(f'Length  {n}th dataset after deleting large / default values: {lendata}')
    
    del biggerdataset
    del biggerDeepCSV_dataset
    
    np.save(dataset_paths[n], dataset)  
    np.save(DeepCSV_paths[n], DeepCSV_dataset)  
    
    gc.collect()
