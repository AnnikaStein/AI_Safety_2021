import numpy as np

import gc



NUM_DATASETS = 200

dataset_paths = ['/hpcwork/um106329/partially_cleaned/cleaned_data_n_x_all/biggerdataset_%d.npy' % k for k in range(0, NUM_DATASETS)]
DeepCSV_paths = ['/hpcwork/um106329/partially_cleaned/cleaned_data_n_x_all/biggerDeepCSV_dataset_%d.npy' % k for k in range(0, NUM_DATASETS)]


# File-loading and Cleaning
def cleandataset(data, DeepCSV):
    
    for j in range(len(data[0])):
        DeepCSV = DeepCSV[data[:, j] > -999]
        data = data[data[:, j] > -999]
        DeepCSV = DeepCSV[data[:, j] <= 1000]
        data = data[data[:, j] <= 1000]
    
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
    
    np.save('/hpcwork/um106329/cleaned/biggerdataset_%d.npy' % n, dataset)  
    np.save('/hpcwork/um106329/cleaned/biggerDeepCSV_dataset_%d.npy' % n, DeepCSV_dataset)  
    
    gc.collect()
