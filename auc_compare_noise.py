import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist

import time

import pickle

import argparse
import ast


parser = argparse.ArgumentParser(description="Compare AUC of different epochs")
parser.add_argument("listepochs", help="The epochs to be evaluated, specified as \"[x,y,z,...]\" ")
parser.add_argument("weighting", type=int, help="The weighting method of the training, either 0 or 2")
parser.add_argument("noisesigma", type=float, help="The magnitude of the attack (sigma for the noise in quotation marks)")
args = parser.parse_args()

weighting_method = args.weighting
noise_sigma = args.noisesigma

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



#at_epoch = [100]
if args.listepochs == "all":
    at_epoch = [i for i in range(1,121)]
else:
    at_epoch = ast.literal_eval(args.listepochs)
#at_epoch = [20,70,120]


print(f'Evaluate with noise at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')
print(f'And sigma={noise_sigma}')

'''

    Load inputs and targets
    
'''
NUM_DATASETS = 200
#NUM_DATASETS = 1   # defines the number of datasets that shall be used in the evaluation (test), if it is different from the number of files used for training

scalers_file_paths = ['/work/um106329/MA/cleaned/preprocessed/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]


allscalers = [torch.load(scalers_file_paths[s]) for s in range(NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len(test_inputs))


test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')

jetFlavour = test_targets+1

NUM_DATASETS = 200

BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))


gc.collect()


if weighting_method == 0:
    '''

        Setup models: Without weighting

    '''
    criterion0 = nn.CrossEntropyLoss()



    model0 = [nn.Sequential(nn.Linear(67, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, 4),
                          nn.Softmax(dim=1)) for _ in range(len(at_epoch))]



    checkpoint0 = [torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{ep}_epochs_v13_GPU_weighted_as_is.pt' % NUM_DATASETS, map_location=torch.device(device)) for ep in at_epoch]

    for e in range(len(at_epoch)):
        model0[e].load_state_dict(checkpoint0[e]["model_state_dict"])
        model0[e].to(device)
        #evaluate network on inputs
        model0[e].eval()
        gc.collect()
else:
    '''

        Setup models: With new weighting method

    '''

    # as calculated in dataset_info.ipynb
    allweights2 = [0.27580367992004956, 0.5756907770526237, 0.1270419391956182, 0.021463603831708488]
    class_weights2 = torch.FloatTensor(allweights2).to(device)

    criterion2 = nn.CrossEntropyLoss(weight=class_weights2)



    model2 = [nn.Sequential(nn.Linear(67, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Dropout(0.1),
                          nn.Linear(100, 100),
                          nn.ReLU(),
                          nn.Linear(100, 4),
                          nn.Softmax(dim=1)) for _ in range(len(at_epoch))]



    checkpoint2 = [torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{ep}_epochs_v13_GPU_weighted_new.pt' % NUM_DATASETS, map_location=torch.device(device)) for ep in at_epoch]

    for e in range(len(at_epoch)):
        model2[e].load_state_dict(checkpoint2[e]["model_state_dict"])
        model2[e].to(device)
        #evaluate network on inputs
        model2[e].eval()
        gc.collect()

del allscalers
del test_inputs
del test_targets
del jetFlavour

gc.collect()

def compare_auc(sigma=0.1,offset=0,method=0):
    start = time.time()
    ##### CREATING THE AUCs #####
    ### NOISE ###   
    auc_noise = []
    noise = torch.Tensor(np.random.normal(offset,sigma,(len(BvsUDSG_inputs),67)))
    for ep in range(len(at_epoch)):        
        if method == 0:
            noise_predictions = model0[ep](BvsUDSG_inputs + noise).detach().numpy()
        else:
            noise_predictions = model2[ep](BvsUDSG_inputs + noise).detach().numpy()

        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions[:,0])
        auc_noise.append(metrics.auc(fpr,tpr))
        del noise_predictions
        del fpr
        del tpr
        del thresholds
        
    sigmatext = str(sigma).replace('.','')
    
    with open(f'/home/um106329/aisafety/models/weighted/compare/auc/auc{args.weighting}_noise_{sigmatext}_{args.listepochs}.data', 'wb') as file:
        pickle.dump(auc_noise, file)
       
    end = time.time()
    print(f"Time to create Noise AUCs: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
    start = end
    
        

compare_auc(sigma=noise_sigma,method=weighting_method)
