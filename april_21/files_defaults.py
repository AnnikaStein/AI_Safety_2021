import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import mplhep as hep

import seaborn as sns


import torch

import time
import gc

import argparse
#import ast

plt.style.use(hep.cms.style.ROOT)

# depending on what's available, or force cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
parser.add_argument("default", type=float, help="Default value")
args = parser.parse_args()

NUM_DATASETS = args.files
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')  # this should only be needed when applying the attack
defaults = minima - default
#defaults = -999*np.ones(100)
 
    
    
# this will have all starts from 0 to (including) 2400
starts = np.arange(0,2450,50)
# this will have all ends from 49 to 2399 as well as 2446 (this was the number of original .root-files)
ends = np.concatenate((np.arange(49,2449,50), np.arange(2446,2447)))



n_defaults = np.zeros((67,5))
n_total = np.zeros(5)
    
for k in range(NUM_DATASETS):
    ins = np.load(f'/hpcwork/um106329/april_21/cleaned_TT/inputs_{starts[k]}_to_{ends[k]}_with_default_{default}.npy')
    targets = ins[:,-1]
    inputs = ins[:,0:67]
    in0 = inputs[targets == 0]
    in1 = inputs[targets == 1]
    in2 = inputs[targets == 2]
    in3 = inputs[targets == 3]
    n_total[0] += len(in0)
    n_total[1] += len(in1)
    n_total[2] += len(in2)
    n_total[3] += len(in3)
    for i in range(67):
        n_defaults[i][0] += len(in0[:,i][in0[:,i] == defaults[i]])
        n_defaults[i][1] += len(in1[:,i][in1[:,i] == defaults[i]])
        n_defaults[i][2] += len(in2[:,i][in2[:,i] == defaults[i]])
        n_defaults[i][3] += len(in3[:,i][in3[:,i] == defaults[i]])
        n_defaults[i][4] += n_defaults[i][0] + n_defaults[i][1] + n_defaults[i][2] + n_defaults[i][3]

n_total[4] = np.sum(n_total[0:4])    


percentages = np.transpose(n_defaults / n_total)
#print(percentages)
#print(n_total)
#print(n_total[-1])
total = int(n_total[-1])


plt.ioff()
'''
fig, ax = plt.subplots(figsize=(12, 1))
#hep.cms.label(loc=0)

hm = sns.heatmap(percentages, cbar=True, vmin=0, vmax=1,
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
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\n{NUM_DATASETS} file, {total} jets, default {default}', size=16, y=1.12)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_dataset_{total}_jets_default_{default}.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
else:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\n{NUM_DATASETS} files, {total} jets, default {default}', size=16, y=1.12)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_datasets_{total}_jets_default_{default}.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
plt.show(block=False)
time.sleep(5)
plt.close('all')
gc.collect(2)
'''


jetINDEX = [0,1,28,41,48,49,56,57,58,59,63,64,65] 
trackINDEX = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,50,51,52,53,54,55,]
svINDEX = [2,3,4,5,60,61,62,66]


fig, ax = plt.subplots(figsize=(24, 12))
#hep.cms.label(loc=0)

hm = sns.heatmap(percentages[:,jetINDEX], cbar=True, vmin=0, vmax=1,
                 fmt='.2f', annot_kws={'size': 14, 'rotation': 90}, annot=True, 
                 square=False, cmap=plt.cm.Reds)

ticksX = np.arange(len(jetINDEX)) + 0.5
ticksY = np.arange(5) + 0.5
ax.set_xticks(ticksX)
ax.set_xticklabels(np.array(display_names)[jetINDEX], rotation=90, fontsize=12)
ax.xaxis.set_ticks_position('none') 
ax.set_yticks(ticksY)
ax.set_yticklabels(['b', 'bb', 'c', 'udsg', 'all'], rotation=360, fontsize=12)
ax.yaxis.set_ticks_position('none') 

if NUM_DATASETS == 1:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\nJet variables: {NUM_DATASETS} file, {total} jets, default {default}', size=16, y=1.02)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_dataset_{total}_jets_default_{default}_Jet.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
else:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\nJet variables: {NUM_DATASETS} files, {total} jets, default {default}', size=16, y=1.02)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_datasets_{total}_jets_default_{default}_Jet.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
plt.show(block=False)
time.sleep(5)
plt.close('all')
gc.collect(2)


fig, ax = plt.subplots(figsize=(24, 12))
#hep.cms.label(loc=0)

hm = sns.heatmap(percentages[:,trackINDEX], cbar=True, vmin=0, vmax=1,
                 fmt='.2f', annot_kws={'size': 14, 'rotation': 90}, annot=True, 
                 square=False, cmap=plt.cm.Reds)

ticksX = np.arange(len(trackINDEX)) + 0.5
ticksY = np.arange(5) + 0.5
ax.set_xticks(ticksX)
ax.set_xticklabels(np.array(display_names)[trackINDEX], rotation=90, fontsize=12)
ax.xaxis.set_ticks_position('none') 
ax.set_yticks(ticksY)
ax.set_yticklabels(['b', 'bb', 'c', 'udsg', 'all'], rotation=360, fontsize=12)
ax.yaxis.set_ticks_position('none') 

if NUM_DATASETS == 1:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\nTrack variables: {NUM_DATASETS} file, {total} jets, default {default}', size=16, y=1.02)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_dataset_{total}_jets_default_{default}_Track.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
else:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\nTrack variables: {NUM_DATASETS} files, {total} jets, default {default}', size=16, y=1.02)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_datasets_{total}_jets_default_{default}_Track.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
plt.show(block=False)
time.sleep(5)
plt.close('all')
gc.collect(2)


fig, ax = plt.subplots(figsize=(24, 12))
#hep.cms.label(loc=0)

hm = sns.heatmap(percentages[:,svINDEX], cbar=True, vmin=0, vmax=1,
                 fmt='.2f', annot_kws={'size': 14, 'rotation': 90}, annot=True, 
                 square=False, cmap=plt.cm.Reds)

ticksX = np.arange(len(svINDEX)) + 0.5
ticksY = np.arange(5) + 0.5
ax.set_xticks(ticksX)
ax.set_xticklabels(np.array(display_names)[svINDEX], rotation=90, fontsize=12)
ax.xaxis.set_ticks_position('none') 
ax.set_yticks(ticksY)
ax.set_yticklabels(['b', 'bb', 'c', 'udsg', 'all'], rotation=360, fontsize=12)
ax.yaxis.set_ticks_position('none') 

if NUM_DATASETS == 1:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\nSecondary Vertex variables: {NUM_DATASETS} file, {total} jets, default {default}', size=16, y=1.02)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_dataset_{total}_jets_default_{default}_SV.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
else:
    ax.set_title(f'Percentage of default values, TT to Semileptonic samples\nSecondary Vertex variables: {NUM_DATASETS} files, {total} jets, default {default}', size=16, y=1.02)
    plt.savefig(f'/home/um106329/aisafety/april_21/tt_defaults_percentage_{NUM_DATASETS}_datasets_{total}_jets_default_{default}_SV.png', dpi=400, bbox_inches='tight', facecolor='w', transparent=False)
plt.show(block=False)
time.sleep(5)
plt.close('all')
gc.collect(2)
