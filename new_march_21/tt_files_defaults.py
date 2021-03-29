import uproot4 as uproot
import numpy as np
import awkward1 as ak

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import mplhep as hep

import seaborn as sns

import coffea.hist as hist

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset, WeightedRandomSampler


import time
import random
import gc

import argparse
import ast

plt.style.use(hep.cms.style.ROOT)

# depending on what's available, or force cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
colorcode = ['firebrick','magenta','cyan','darkgreen']

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
args = parser.parse_args()

NUM_DATASETS = args.files




train_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)] 
val_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
val_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)] 
test_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]     


prepro_train_inputs = torch.cat(tuple(torch.load(vi).to(device) for vi in train_input_file_paths)).float().numpy()
prepro_train_targets = torch.cat(tuple(torch.load(vi).to(device) for vi in train_target_file_paths)).numpy()

tr_0 = prepro_train_inputs[prepro_train_targets == 0]
tr_1 = prepro_train_inputs[prepro_train_targets == 1]
tr_2 = prepro_train_inputs[prepro_train_targets == 2]
tr_3 = prepro_train_inputs[prepro_train_targets == 3]
ltr0 = len(tr_0)
ltr1 = len(tr_1)
ltr2 = len(tr_2)
ltr3 = len(tr_3)
ltr = ltr0 + ltr1 + ltr2 + ltr3
del prepro_train_inputs
del prepro_train_targets
gc.collect()

prepro_val_inputs = torch.cat(tuple(torch.load(vi).to(device) for vi in val_input_file_paths)).float().numpy()
prepro_val_targets = torch.cat(tuple(torch.load(vi).to(device) for vi in val_target_file_paths)).numpy()

va_0 = prepro_val_inputs[prepro_val_targets == 0]
va_1 = prepro_val_inputs[prepro_val_targets == 1]
va_2 = prepro_val_inputs[prepro_val_targets == 2]
va_3 = prepro_val_inputs[prepro_val_targets == 3]
lva0 = len(va_0)
lva1 = len(va_1)
lva2 = len(va_2)
lva3 = len(va_3)
lva = lva0 + lva1 + lva2 + lva3
del prepro_val_inputs
del prepro_val_targets
gc.collect()

prepro_test_inputs = torch.cat(tuple(torch.load(vi).to(device) for vi in test_input_file_paths)).float().numpy()
prepro_test_targets = torch.cat(tuple(torch.load(vi).to(device) for vi in test_target_file_paths)).numpy()

te_0 = prepro_test_inputs[prepro_test_targets == 0]
te_1 = prepro_test_inputs[prepro_test_targets == 1]
te_2 = prepro_test_inputs[prepro_test_targets == 2]
te_3 = prepro_test_inputs[prepro_test_targets == 3]
lte0 = len(te_0)
lte1 = len(te_1)
lte2 = len(te_2)
lte3 = len(te_3)
lte = lte0 + lte1 + lte2 + lte3
del prepro_test_inputs
del prepro_test_targets
gc.collect()

total = ltr + lva + lte

tr_defaults = []
va_defaults = []
te_defaults = []
total_defaults = []


f0 = ltr0 + lva0 + lte0
f1 = ltr1 + lva1 + lte1
f2 = ltr2 + lva2 + lte2
f3 = ltr3 + lva3 + lte3


for i in range(67):
        
    tr_defaults.append([len(tr_0[:,i][tr_0[:,i] < -998]) / f0, len(tr_1[:,i][tr_1[:,i] < -998]) / f1, len(tr_2[:,i][tr_2[:,i] < -998]) / f2, len(tr_3[:,i][tr_3[:,i] < -998]) / f3])
    va_defaults.append([len(va_0[:,i][va_0[:,i] < -998]) / f0, len(va_1[:,i][va_1[:,i] < -998]) / f1, len(va_2[:,i][va_2[:,i] < -998]) / f2, len(va_3[:,i][va_3[:,i] < -998]) / f3])
    te_defaults.append([len(te_0[:,i][te_0[:,i] < -998]) / f0, len(te_1[:,i][te_1[:,i] < -998]) / f1, len(te_2[:,i][te_2[:,i] < -998]) / f2, len(te_3[:,i][te_3[:,i] < -998]) / f3])
    
    total_defaults.append((len(tr_0[:,i][tr_0[:,i] < -998]) + len(tr_1[:,i][tr_1[:,i] < -998]) + len(tr_2[:,i][tr_2[:,i] < -998]) + len(tr_3[:,i][tr_3[:,i] < -998]) + 
                           len(va_0[:,i][va_0[:,i] < -998]) + len(va_1[:,i][va_1[:,i] < -998]) + len(va_2[:,i][va_2[:,i] < -998]) + len(va_3[:,i][va_3[:,i] < -998]) +
                           len(te_0[:,i][te_0[:,i] < -998]) + len(te_1[:,i][te_1[:,i] < -998]) + len(te_2[:,i][te_2[:,i] < -998]) + len(te_3[:,i][te_3[:,i] < -998])) / total)
    
    
print(tr_defaults)
print(va_defaults)
print(te_defaults)

combined_def = np.array(tr_defaults) + np.array(va_defaults) + np.array(te_defaults)


fig, ax = plt.subplots(figsize=(24, 2))
#hep.cms.label(loc=0)

hm = sns.heatmap(np.vstack([np.transpose(combined_def), np.array(total_defaults)]), cbar=True, vmin=0, vmax=1,
                 fmt='.2f', annot_kws={'size': 3}, annot=True, 
                 square=True, cmap=plt.cm.Reds)

ticksX = np.arange(67) + 0.5
ticksY = np.arange(5) + 0.5
ax.set_xticks(ticksX)
ax.set_xticklabels(display_names, rotation=90, fontsize=8)
ax.xaxis.set_ticks_position('none') 
ax.set_yticks(ticksY)
ax.set_yticklabels(['b', 'bb', 'c', 'udsg', 'all'], rotation=360, fontsize=8)
ax.yaxis.set_ticks_position('none') 

if NUM_DATASETS == 1:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\n{NUM_DATASETS} file, {total} jets', size=16, y=1.12)
    plt.savefig(f'/home/um106329/aisafety/new_march_21/tt_defaults_percentage_{NUM_DATASETS}_dataset_{total}_jets_v3.svg', bbox_inches='tight', facecolor='w', transparent=False)
else:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\n{NUM_DATASETS} files, {total} jets', size=16, y=1.12)
    plt.savefig(f'/home/um106329/aisafety/new_march_21/tt_defaults_percentage_{NUM_DATASETS}_datasets_{total}_jets_v3.svg', bbox_inches='tight', facecolor='w', transparent=False)

