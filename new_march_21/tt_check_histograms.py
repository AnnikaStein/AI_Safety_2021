#import uproot4 as uproot
import numpy as np
#import awkward1 as ak

#from sklearn import metrics
#from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import mplhep as hep

#import seaborn as sns

#import coffea.hist as hist

#import torch
#import torch.nn as nn
#from torch.utils.data import TensorDataset, ConcatDataset, WeightedRandomSampler


import time
#import random
import gc

import argparse
import ast

plt.style.use(hep.cms.style.ROOT)

# depending on what's available, or force cpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

#C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
#colorcode = ['firebrick','magenta','cyan','darkgreen']

display_names = ['Jet $\eta$',
                'Jet $p_T$',
                'Flight Distance 2D Sig','Flight Distance 2D Val','Flight Distance 3D Sig', 'Flight Distance 3D Val',
                'Track Decay Len Val [0]','Track Decay Len Val [1]','Track Decay Len Val [2]','Track Decay Len Val [3]','Track Decay Len Val [4]','Track Decay Len Val [5]',
                'Track $\Delta R$ [0]','Track $\Delta R$ [1]','Track $\Delta R$ [2]','Track $\Delta R$ [3]','Track $\Delta R$ [4]','Track $\Delta R$ [5]',
                'Track $\eta_{rel}$ [0]','Track $\eta_{rel}$ [1]','Track $\eta_{rel}$ [2]','Track $\eta_{rel}$ [3]',
                'Track Jet Dist Val [0]','Track Jet Dist Val [1]','Track Jet Dist Val [2]','Track Jet Dist Val [3]','Track Jet Dist Val [4]','Track Jet Dist Val [5]',
                'Track Jet $p_T$',
                'Track $p_T$ Ratio [0]','Track $p_T$ Ratio [1]','Track $p_T$ Ratio [2]','Track $p_T$ Ratio [3]','Track $p_T$ Ratio [4]','Track $p_T$ Ratio [5]',
                'Track $p_{T,rel}$ [0]','Track $p_{T,rel}$ [1]','Track $p_{T,rel}$ [2]','Track $p_{T,rel}$ [3]','Track $p_{T,rel}$ [4]','Track $p_{T,rel}$ [5]',
                'Track SIP 2D Sig Above Charm',
                'Track SIP 2D Sig [0]','Track SIP 2D Sig [1]','Track SIP 2D Sig [2]','Track SIP 2D Sig [3]','Track SIP 2D Sig [4]','Track SIP 2D Sig [5]',
                'Track SIP 2D Val Above Charm',
                'Track SIP 3D Sig Above Charm',
                'Track SIP 3D Sig [0]','Track SIP 3D Sig [1]','Track SIP 3D Sig [2]','Track SIP 3D Sig [3]','Track SIP 3D Sig [4]','Track SIP 3D Sig [5]',
                'Track SIP 3D Val Above Charm',
                'Track Sum Jet $\Delta R$','Track Sum Jet $E_T$ Ratio',
                'Vertex Category','Vertex Energy Ratio','Vertex Jet $\Delta R$','Vertex Mass',
                'Jet N Secondary Vertices','Jet N Selected Tracks','Jet N Tracks $\eta_{rel}$','Vertex N Tracks']



parser = argparse.ArgumentParser(description="Setup for input checking")
parser.add_argument("files", type=int, help="Number of files")
parser.add_argument("variable", type=int, help="Index of variable")
args = parser.parse_args()

NUM_DATASETS = args.files
variable = args.variable


# this will have all starts from 0 to (including) 11400
starts = np.arange(0,2450,50)
# this will have all ends from 49 to 11399 as well as 11407 (this was the number of original .root-files)
ends = np.concatenate((np.arange(49,2449,50), np.arange(2446,2447)))             



# TT to Semileptonic
dataset_paths = [f'/hpcwork/um106329/new_march_21/cleanedTTtoSemilep/inputs_{starts[k]}_to_{ends[k]}.npy' for k in range(0, NUM_DATASETS)]
#DeepCSV_paths = [f'/hpcwork/um106329/new_march_21/cleanedTTtoSemilep/deepcsv_{starts[k]}_to_{ends[k]}.npy' for k in range(0, NUM_DATASETS)]

ins = np.concatenate([np.load(dataset_paths[i]) for i in range(NUM_DATASETS)])
lenins = len(ins)

variable0 = ins[:,variable][ins[:,-1] == 0]
variable1 = ins[:,variable][ins[:,-1] == 1]
variable2 = ins[:,variable][ins[:,-1] == 2]
variable3 = ins[:,variable][ins[:,-1] == 3]
del ins
gc.collect()


plt.ioff()

plt.hist(variable0,histtype='step',bins=np.arange(max(variable0)+2)-0.5,log=False)
plt.hist(variable1,histtype='step',bins=np.arange(max(variable1)+2)-0.5,log=False)
plt.hist(variable2,histtype='step',bins=np.arange(max(variable2)+2)-0.5,log=False)
plt.hist(variable3,histtype='step',bins=np.arange(max(variable3)+2)-0.5,log=False)
plt.legend(['b','bb','c','udsg'])
plt.title(f'TT to Semileptonic, {lenins} jets\n{display_names[variable]}, absolute numbers', size=16, y=1.02)
plt.savefig(f'/home/um106329/aisafety/new_march_21/tt_{variable}_{NUM_DATASETS}_datasets_{lenins}_jets_absolute.svg', bbox_inches='tight', facecolor='w', transparent=False)
plt.show(block=False)
time.sleep(5)
plt.close('all')
gc.collect(2)

plt.hist(variable0,histtype='step',bins=np.arange(max(variable0)+2)-0.5,log=False, density=True)
plt.hist(variable1,histtype='step',bins=np.arange(max(variable1)+2)-0.5,log=False, density=True)
plt.hist(variable2,histtype='step',bins=np.arange(max(variable2)+2)-0.5,log=False, density=True)
plt.hist(variable3,histtype='step',bins=np.arange(max(variable3)+2)-0.5,log=False, density=True)
plt.legend(['b','bb','c','udsg'])
plt.title(f'TT to Semileptonic, {lenins} jets\n{display_names[variable]}, relative numbers', size=16, y=1.02)
plt.savefig(f'/home/um106329/aisafety/new_march_21/tt_{variable}_{NUM_DATASETS}_datasets_{lenins}_jets_relative.svg', bbox_inches='tight', facecolor='w', transparent=False)
plt.show(block=False)
time.sleep(5)
plt.close('all')
gc.collect(2)


