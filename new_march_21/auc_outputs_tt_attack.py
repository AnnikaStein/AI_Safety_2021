#import multiprocessing as mp

#import uproot4 as uproot
import numpy as np
#import awkward1 as ak

from sklearn import metrics
#from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt
#import mplhep as hep

#import coffea.hist as hist

import torch
import torch.nn as nn
#from torch.utils.data import TensorDataset, ConcatDataset, WeightedRandomSampler


import time
import random
import gc

import argparse
import ast

import pandas as pd
#import seaborn as sns

#plt.style.use([hep.style.ROOT, hep.style.firamath])

# depending on what's available, or force cpu
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')



parser = argparse.ArgumentParser(description="Setup for AUC for model output")
parser.add_argument("files", type=int, help="Number of files for AUC")
parser.add_argument("mode", type=str, help="Mode: raw, deepcsv, noise, FGSM")
parser.add_argument("param", type=float, help="Parameter used for attack, or 0 for raw / deepcsv")
parser.add_argument("traindataset", type=str, help="Dataset used during training, qcd or tt")
args = parser.parse_args()

NUM_DATASETS = args.files
mode = args.mode
param = args.param
traindataset = args.traindataset

#print('Parameters for this AUC ranking:')
#print(f'NUM_DATASETS:\t{NUM_DATASETS}\nstart:\t{start}\nend:\t{end}\nmode:\t{mode}\nparam:\t{param}\ntraindataset:\t{traindataset}\n')



#C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
#colorcode = ['firebrick','magenta','cyan','darkgreen']

#categories = ['b', 'bb', 'c', 'udsg']
#labels = [0, 1, 2, 3]

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

input_names = ['Jet_eta',
 'Jet_pt',
 'Jet_DeepCSV_flightDistance2dSig',
 'Jet_DeepCSV_flightDistance2dVal',
 'Jet_DeepCSV_flightDistance3dSig',
 'Jet_DeepCSV_flightDistance3dVal',
 'Jet_DeepCSV_trackDecayLenVal_0',
 'Jet_DeepCSV_trackDecayLenVal_1',
 'Jet_DeepCSV_trackDecayLenVal_2',
 'Jet_DeepCSV_trackDecayLenVal_3',
 'Jet_DeepCSV_trackDecayLenVal_4',
 'Jet_DeepCSV_trackDecayLenVal_5',
 'Jet_DeepCSV_trackDeltaR_0',
 'Jet_DeepCSV_trackDeltaR_1',
 'Jet_DeepCSV_trackDeltaR_2',
 'Jet_DeepCSV_trackDeltaR_3',
 'Jet_DeepCSV_trackDeltaR_4',
 'Jet_DeepCSV_trackDeltaR_5',
 'Jet_DeepCSV_trackEtaRel_0',
 'Jet_DeepCSV_trackEtaRel_1',
 'Jet_DeepCSV_trackEtaRel_2',
 'Jet_DeepCSV_trackEtaRel_3',
 'Jet_DeepCSV_trackJetDistVal_0',
 'Jet_DeepCSV_trackJetDistVal_1',
 'Jet_DeepCSV_trackJetDistVal_2',
 'Jet_DeepCSV_trackJetDistVal_3',
 'Jet_DeepCSV_trackJetDistVal_4',
 'Jet_DeepCSV_trackJetDistVal_5',
 'Jet_DeepCSV_trackJetPt',
 'Jet_DeepCSV_trackPtRatio_0',
 'Jet_DeepCSV_trackPtRatio_1',
 'Jet_DeepCSV_trackPtRatio_2',
 'Jet_DeepCSV_trackPtRatio_3',
 'Jet_DeepCSV_trackPtRatio_4',
 'Jet_DeepCSV_trackPtRatio_5',
 'Jet_DeepCSV_trackPtRel_0',
 'Jet_DeepCSV_trackPtRel_1',
 'Jet_DeepCSV_trackPtRel_2',
 'Jet_DeepCSV_trackPtRel_3',
 'Jet_DeepCSV_trackPtRel_4',
 'Jet_DeepCSV_trackPtRel_5',
 'Jet_DeepCSV_trackSip2dSigAboveCharm',
 'Jet_DeepCSV_trackSip2dSig_0',
 'Jet_DeepCSV_trackSip2dSig_1',
 'Jet_DeepCSV_trackSip2dSig_2',
 'Jet_DeepCSV_trackSip2dSig_3',
 'Jet_DeepCSV_trackSip2dSig_4',
 'Jet_DeepCSV_trackSip2dSig_5',
 'Jet_DeepCSV_trackSip2dValAboveCharm',
 'Jet_DeepCSV_trackSip3dSigAboveCharm',
 'Jet_DeepCSV_trackSip3dSig_0',
 'Jet_DeepCSV_trackSip3dSig_1',
 'Jet_DeepCSV_trackSip3dSig_2',
 'Jet_DeepCSV_trackSip3dSig_3',
 'Jet_DeepCSV_trackSip3dSig_4',
 'Jet_DeepCSV_trackSip3dSig_5',
 'Jet_DeepCSV_trackSip3dValAboveCharm',
 'Jet_DeepCSV_trackSumJetDeltaR',
 'Jet_DeepCSV_trackSumJetEtRatio',
 'Jet_DeepCSV_vertexCategory',
 'Jet_DeepCSV_vertexEnergyRatio',
 'Jet_DeepCSV_vertexJetDeltaR',
 'Jet_DeepCSV_vertexMass',
 'Jet_DeepCSV_jetNSecondaryVertices',
 'Jet_DeepCSV_jetNSelectedTracks',
 'Jet_DeepCSV_jetNTracksEtaRel',
 'Jet_DeepCSV_vertexNTracks',]



if traindataset == 'tt':
    scalers_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

    DeepCSV_testset_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]

    test_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    test_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

    #val_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    #val_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

    #train_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    #train_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    
    suffix = 'TT'

else:
    scalers_file_paths = ['/work/um106329/MA/cleaned/preprocessed/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

    DeepCSV_testset_file_paths = ['/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]

    test_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    test_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

    #val_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    #val_target_file_paths = [/work/um106329/MA/cleaned/preprocessed/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

    #train_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
    #train_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
    
    suffix = 'QCD'

#all_target_file_paths_2D = [[test_target_file_paths[i],val_target_file_paths[i],train_target_file_paths[i]] for i in range(0,NUM_DATASETS)]
#all_target_file_paths = [item for sublist in all_target_file_paths_2D for item in sublist]
#all_target_file_paths = test_target_file_paths

#flav = torch.cat(tuple(torch.load(ti) for ti in all_target_file_paths)).numpy().astype(int) + 1


allscalers = [torch.load(scalers_file_paths[s]) for s in range(NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len(test_inputs))


test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')


jetFlavour = test_targets+1


# ================= Load model and corresponding loss function (weights) =================

model = nn.Sequential(nn.Linear(67, 100),
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
                      nn.Softmax(dim=1))


if traindataset == 'tt':
    allweights = compute_class_weight(
           'balanced',
            classes=np.array([0,1,2,3]), 
            y=test_targets.numpy())
    class_weights = torch.FloatTensor(allweights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    checkpoint = torch.load(f'/home/um106329/aisafety/new_march_21/models/model_all_TT_180_epochs_v10_GPU_weighted_new_49_datasets.pt', map_location=torch.device(device))
    
    with open('/home/um106329/aisafety/new_march_21/length_test_datasets_TT.npy', 'rb') as f:
        length_data_test = np.load(f)
    
else:
    criterion = nn.CrossEntropyLoss()
        
    checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/200_full_files_120_epochs_v13_GPU_weighted_as_is.pt', map_location=torch.device(device))
    
    with open('/home/um106329/aisafety/length_data_test.npy', 'rb') as f:
        length_data_test = np.load(f)
    
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

model.eval()


# ========================================================================================

    
    
    



# ======================================== Raw ===========================================    

if mode == 'raw':
    BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==2],test_inputs[jetFlavour==4]))
    y_true_bvl = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
    del jetFlavour
    del test_inputs
    del test_targets
    gc.collect()
    with torch.no_grad():
        y_pred_bvl = model(BvsUDSG_inputs).detach().numpy()
    del BvsUDSG_inputs
    gc.collect()
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvl[i]==0 or y_true_bvl[i]==1 else 0) for i in range(len(y_true_bvl))], (y_pred_bvl[:,0]+y_pred_bvl[:,1]))
    del y_pred_bvl
    del y_true_bvl
    gc.collect()
    
     
    auc_bvl = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
    test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()

    jetFlavour = test_targets+1
    
    BvsC_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==2],test_inputs[jetFlavour==3]))
    y_true_bvc = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==3]))
    del jetFlavour
    del test_inputs
    del test_targets
    gc.collect()
    
    with torch.no_grad():
        y_pred_bvc = model(BvsC_inputs).detach().numpy()
    del BvsC_inputs
    gc.collect()
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvc[i]==0 or y_true_bvc[i]==1 else 0) for i in range(len(y_true_bvc))], (y_pred_bvc[:,0]+y_pred_bvc[:,1]))
    auc_bvc = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    print(f'Raw AUC bvl: {auc_bvl}, AUC bvc: {auc_bvc}')
    
    df = pd.DataFrame(data={'input_name' : ['Raw'], 'auc_bvl' : [auc_bvl], 'auc_bvc' : [auc_bvc]}, columns =['input_name', 'auc_bvl', 'auc_bvc'])
    df.to_pickle(f'/home/um106329/aisafety/new_march_21/df_auc_ranking_NFiles_{NUM_DATASETS}_MODE_{mode}_CUSTOM_TAGGER_OUT_{suffix}.pkl')
    
     
    
# ========================================================================================



# ======================================= DeepCSV ========================================    

elif mode == 'deepcsv':
    DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
    print('DeepCSV test done')
    
    y_pred_bvl = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==2],DeepCSV_testset[jetFlavour==4]))
    y_true_bvl = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvl[i]==0 or y_true_bvl[i]==1 else 0) for i in range(len(y_true_bvl))], (y_pred_bvl[:,0]+y_pred_bvl[:,1]))
    del y_pred_bvl
    del y_true_bvl
    gc.collect()
    
     
    auc_bvl = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    
    y_pred_bvc = np.concatenate((test_inputs[jetFlavour==1],test_inputs[jetFlavour==2],test_inputs[jetFlavour==3]))
    y_true_bvc = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==3]))
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvc[i]==0 or y_true_bvc[i]==1 else 0) for i in range(len(y_true_bvc))], (y_pred_bvc[:,0]+y_pred_bvc[:,1]))
    auc_bvc = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    print(f'Raw AUC bvl: {auc_bvl}, AUC bvc: {auc_bvc}')
    
    df = pd.DataFrame(data={'input_name' : ['DeepCSV'], 'auc_bvl' : [auc_bvl], 'auc_bvc' : [auc_bvc]}, columns =['input_name', 'auc_bvl', 'auc_bvc'])
    df.to_pickle(f'/home/um106329/aisafety/new_march_21/df_auc_ranking_NFiles_{NUM_DATASETS}_MODE_{mode}_{suffix}.pkl')
    
     
    
# ========================================================================================



# ====================================== Noise ===========================================

def apply_noise(magn=param,sample=None, offset=[0]):
    
    noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),67)))
    
    xadv = (sample + noise).detach()
            
    xadv[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')] = sample[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')]
    xadv[:,input_names.index('Jet_DeepCSV_vertexCategory')] = sample[:,input_names.index('Jet_DeepCSV_vertexCategory')]
    xadv[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')] = sample[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')]
    xadv[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')] = sample[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')]
    xadv[:,input_names.index('Jet_DeepCSV_vertexNTracks')] = sample[:,input_names.index('Jet_DeepCSV_vertexNTracks')]
    
    for i in range(67):
        defaults = np.zeros(len(sample))
        for l, s in enumerate(length_data_test[:NUM_DATASETS]):
            scalers = allscalers[l]
            if l == 0:
                defaults[:int(s)] = abs(scalers[i].inverse_transform(sample[:int(s),i].cpu()) + 999) < 300
            else:
                defaults[int(length_data_test[l-1]) : int(s)] = abs(scalers[i].inverse_transform(sample[int(length_data_test[l-1]) : int(s),i].cpu()) + 999) < 300

        if np.sum(defaults) != 0:
            for i in range(67):
                xadv[:,i][defaults] = sample[:,i][defaults]
            break
    
    # ====== For QCD with -1 bin =======
    if traindataset == 'qcd':
        for i in [41, 48, 49, 56]:
            defaults = np.zeros(len(sample))
            for l, s in enumerate(length_data_test[:NUM_DATASETS]):
                scalers = allscalers[l]
                if l == 0:
                    defaults[:int(s)] = abs(scalers[i].inverse_transform(sample[:int(s),i].cpu()) + 1.0) < 0.001
                else:
                    defaults[int(length_data_test[l-1]) : int(s)] = abs(scalers[i].inverse_transform(sample[int(length_data_test[l-1]) : int(s),i].cpu()) + 1.0) < 0.001

            if np.sum(defaults) != 0:
                for i in [41, 48, 49, 56]:
                    xadv[:,i][defaults] = sample[:,i][defaults]
                break
    
    
    
    return xadv.detach()

if mode == 'noise':
    
    adv_inputs = apply_noise(param,test_inputs).cpu()
    
    adv_bvl = torch.cat((adv_inputs[jetFlavour==1],adv_inputs[jetFlavour==2],adv_inputs[jetFlavour==4]))
    y_true_bvl = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
    del jetFlavour
    del test_inputs
    del test_targets
    gc.collect()
    with torch.no_grad():
        y_pred_bvl = model(adv_bvl).detach().numpy()
    del adv_bvl
    gc.collect()
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvl[i]==0 or y_true_bvl[i]==1 else 0) for i in range(len(y_true_bvl))], (y_pred_bvl[:,0]+y_pred_bvl[:,1]))
    auc_bvl = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
    test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()

    jetFlavour = test_targets+1
    
    adv_bvc = torch.cat((adv_inputs[jetFlavour==1],adv_inputs[jetFlavour==2],test_inputs[jetFlavour==3]))
    y_true_bvc = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==3]))
    del jetFlavour
    del test_inputs
    del test_targets
    gc.collect()
    with torch.no_grad():
        y_pred_bvc = model(adv_bvc).detach().numpy()
    del adv_bvc
    gc.collect()
    
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvc[i]==0 or y_true_bvc[i]==1 else 0) for i in range(len(y_true_bvc))], (y_pred_bvc[:,0]+y_pred_bvc[:,1]))
    auc_bvc = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    print(f'Noise {param} AUC bvl: {auc_bvl}, AUC bvc: {auc_bvc}')
    
    df = pd.DataFrame(data={'input_name' : [f'Noise $\sigma={param}$'], 'auc_bvl' : [auc_bvl], 'auc_bvc' : [auc_bvc]}, columns =['input_name', 'auc_bvl', 'auc_bvc'])
    df.to_pickle(f'/home/um106329/aisafety/new_march_21/df_auc_ranking_NFiles_{NUM_DATASETS}_MODE_{mode}_CUSTOM_TAGGER_OUT_PARAM_{param}_{suffix}.pkl')
    
# =======================================================================================



# ===================================== FGSM ============================================

def fgsm_attack(epsilon=1e-1,sample=None,targets=None,reduced=True):
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    preds = model(xadv)
    loss = criterion(preds, targets.long()).mean()
    model.zero_grad()
    
    
    loss.backward()
    
    with torch.no_grad():
        #now we obtain the gradient of the input. It has the same dimensions as the tensor xadv, and it "points" in the direction of increasing loss values.
        dx = torch.sign(xadv.grad.detach())
        
        #so, we take a step in that direction!
        xadv += epsilon*torch.sign(dx)
        
        #remove the impact on selected variables. This is nessecary to avoid problems that occur otherwise in the input shapes.
        if reduced:
            
            xadv[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')] = sample[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')]
            xadv[:,input_names.index('Jet_DeepCSV_vertexCategory')] = sample[:,input_names.index('Jet_DeepCSV_vertexCategory')]
            xadv[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')] = sample[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')]
            xadv[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')] = sample[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')]
            xadv[:,input_names.index('Jet_DeepCSV_vertexNTracks')] = sample[:,input_names.index('Jet_DeepCSV_vertexNTracks')]
            #xadv[:,12:][sample[:,12:]==0] = 0   # TagVarCSVTrk_trackJetDistVal, but I have not set any variable to 0 manually during cleaning
            #xadv[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):][sample[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):] == 0] = 0
            
            for i in range(67):
                defaults = np.zeros(len(sample))
                for l, s in enumerate(length_data_test[:NUM_DATASETS]):
                    scalers = allscalers[l]
                    if l == 0:
                        defaults[:int(s)] = abs(scalers[i].inverse_transform(sample[:int(s),i].cpu()) + 999) < 300
                    else:
                        defaults[int(length_data_test[l-1]) : int(s)] = abs(scalers[i].inverse_transform(sample[int(length_data_test[l-1]) : int(s),i].cpu()) + 999) < 300
                        
                if np.sum(defaults) != 0:
                    for i in range(67):
                        xadv[:,i][defaults] = sample[:,i][defaults]
                    break
            
            # ====== For QCD with -1 bin =======
            if traindataset == 'qcd':
                for i in [41, 48, 49, 56]:
                    defaults = np.zeros(len(sample))
                    for l, s in enumerate(length_data_test[:NUM_DATASETS]):
                        scalers = allscalers[l]
                        if l == 0:
                            defaults[:int(s)] = abs(scalers[i].inverse_transform(sample[:int(s),i].cpu()) + 1.0) < 0.001
                        else:
                            defaults[int(length_data_test[l-1]) : int(s)] = abs(scalers[i].inverse_transform(sample[int(length_data_test[l-1]) : int(s),i].cpu()) + 1.0) < 0.001

                    if np.sum(defaults) != 0:
                        for i in [41, 48, 49, 56]:
                            xadv[:,i][defaults] = sample[:,i][defaults]
                        break

            
        return xadv.detach()



if mode == 'FGSM':
    adv_inputs = fgsm_attack(param,test_inputs,test_targets,reduced=True).cpu()
    
    adv_bvl = torch.cat((adv_inputs[jetFlavour==1],adv_inputs[jetFlavour==2],adv_inputs[jetFlavour==4]))
    y_true_bvl = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
    del jetFlavour
    del test_inputs
    del test_targets
    gc.collect()
    with torch.no_grad():
        y_pred_bvl = model(adv_bvl).detach().numpy()
    del adv_bvl
    gc.collect()
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvl[i]==0 or y_true_bvl[i]==1 else 0) for i in range(len(y_true_bvl))], (y_pred_bvl[:,0]+y_pred_bvl[:,1]))
    auc_bvl = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    
    test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
    test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()

    jetFlavour = test_targets+1
    
    adv_bvc = torch.cat((adv_inputs[jetFlavour==1],adv_inputs[jetFlavour==2],adv_inputs[jetFlavour==3]))
    y_true_bvc = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==3]))
    del jetFlavour
    del test_inputs
    del test_targets
    gc.collect()
    with torch.no_grad():
        y_pred_bvc = model(adv_bvc).detach().numpy()
    del adv_bvc
    gc.collect()
    
    fpr,tpr,_ = metrics.roc_curve([(1 if y_true_bvc[i]==0 or y_true_bvc[i]==1 else 0) for i in range(len(y_true_bvc))], (y_pred_bvc[:,0]+y_pred_bvc[:,1]))
    auc_bvc = metrics.auc(fpr,tpr)
    del fpr
    del tpr
    gc.collect()
    
    print(f'FGSM {param} AUC bvl: {auc_bvl}, AUC bvc: {auc_bvc}')
    
    df = pd.DataFrame(data={'input_name' : [f'FGSM $\epsilon={param}$'], 'auc_bvl' : [auc_bvl], 'auc_bvc' : [auc_bvc]}, columns =['input_name', 'auc_bvl', 'auc_bvc'])
    df.to_pickle(f'/home/um106329/aisafety/new_march_21/df_auc_ranking_NFiles_{NUM_DATASETS}_MODE_{mode}_CUSTOM_TAGGER_OUT_PARAM_{param}_{suffix}.pkl')