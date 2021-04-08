#import uproot4 as uproot
import numpy as np
#import awkward1 as ak

#from sklearn import metrics
from sklearn.metrics import roc_auc_score
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



parser = argparse.ArgumentParser(description="Setup for AUC ranking")
parser.add_argument("files", type=int, help="Number of files for AUC ranking")
parser.add_argument("mode", type=str, help="Mode: raw, noise, FGSM")
parser.add_argument("param", type=float, help="Parameter used for attack, or 0 for raw")
args = parser.parse_args()

NUM_DATASETS = args.files
mode = args.mode
param = args.param





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




scalers_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

val_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
val_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

train_input_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_target_file_paths = ['/hpcwork/um106329/new_march_21/scaledTTtoSemilep/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

all_target_file_paths = test_target_file_paths + val_target_file_paths + train_target_file_paths

flav = torch.cat(tuple(torch.load(ti) for ti in all_target_file_paths)).numpy().astype(int) + 1


def auc_ranking(inputs):
    list_variables = []
    list_auc_bvl = []
    list_auc_bvc = []

    for deepcsv_input in range(67):

        bvl = inputs[inputs[:,-1] != 3]
        y_true_bvl = bvl[:,-1] != 4
        y_pred_bvl = bvl[:,deepcsv_input]

        bvc = inputs[inputs[:,-1] != 4]
        y_true_bvc = bvc[:,-1] != 3
        y_pred_bvc = bvc[:,deepcsv_input]


        auc_bvl = roc_auc_score(y_true_bvl, y_pred_bvl)
        auc_bvc = roc_auc_score(y_true_bvc, y_pred_bvc)

        print(f'Variable: {display_names[deepcsv_input]}, AUC bvl: {auc_bvl}, AUC bvc: {auc_bvc}')

        list_variables.append(display_names[deepcsv_input])
        list_auc_bvl.append(auc_bvl)
        list_auc_bvc.append(auc_bvc)

    df = pd.DataFrame(list(zip(list_variables, list_auc_bvl, list_auc_bvc)), columns =['input_name', 'auc_bvl', 'auc_bvc'])
    df.to_pickle(f'/home/um106329/aisafety/new_march_21/df_auc_ranking_NFiles_{NUM_DATASETS}_MODE_{mode}_PARAM_{param}.pkl')

    #sorted_df = df.sort_values('auc_bvl')


    #sorted_df = df.sort_values('auc_bvc')


# ======================================== Raw ===========================================    

if mode == 'raw':
    #test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
    #val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
    #train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
    
    all_input_file_paths = test_input_file_paths + val_input_file_paths + train_input_file_paths
    
    raw_inputs = torch.cat(tuple(torch.load(ti).to(device) for ti in all_input_file_paths)).numpy()
    
    raw_inputs = np.c_[raw_inputs, flav]
    
    auc_ranking(raw_inputs)
    
# ========================================================================================



# ====================================== Noise ===========================================

if mode == 'noise':

    def apply_noise(magn=[1],offset=[0],variable=0):
        xmagn = []
        for s in range(0, NUM_DATASETS):
            scalers = torch.load(scalers_file_paths[s])
            test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
            val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
            train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
            #test_targets =  torch.load(test_target_file_paths[s]).to(device)
            #val_targets =  torch.load(val_target_file_paths[s]).to(device)
            #train_targets =  torch.load(train_target_file_paths[s]).to(device)            
            all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))
            del test_inputs
            del val_inputs
            del train_inputs
            gc.collect()
            
            
            for i, m in enumerate(magn):
                noise = torch.Tensor(np.random.normal(offset,m,(len(all_inputs),67))).to(device)
                all_inputs_noise = all_inputs + noise
                if s > 0:
                    xadv = scalers[variable].inverse_transform(all_inputs_noise[:][:,variable].cpu())
                    integervars = [59,63,64,65,66]
                    if variable in integervars:
                        xadv = np.rint(scalers[variable].inverse_transform(all_inputs[:][:,variable].cpu()))

                    '''
                    if variable in [41, 48, 49, 56]:
                        defaults = abs(scalers[variable].inverse_transform(all_inputs[:,variable].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                    '''

                    '''
                    # as long as nothing was set to 0 manually, not really necessary
                    vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
                    vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
                    vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
                    vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
                    vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
                    vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
                    if variable in vars_with_0_defaults:
                        defaults = abs(scalers[i].inverse_transform(all_inputs[:,variable].cpu())) < 0.001   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = all_inputs[:,variable][defaults]
                    '''        

                    '''
                        # For cleaned files (QCD or TT to Semileptonic)
                    '''

                    if variable in range(67):
                        defaults = abs(scalers[i].inverse_transform(all_inputs[:,i].cpu()) + 999) < 300   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                    '''        

                    if variable in range(67):
                        defaults = abs(all_inputs[:,variable].cpu() + 900) < 0
                        if np.sum(defaults) != 0:
                            xadv[defaults] = all_inputs[:,variable].cpu()[defaults]
                    '''        
                    xadv_new = np.concatenate((xmagn[i], xadv))
                    xmagn[i] = xadv_new
                else:
                    xadv = scalers[variable].inverse_transform(all_inputs_noise[:][:,variable].cpu())
                    integervars = [59,63,64,65,66]
                    if variable in integervars:
                        xadv = np.rint(scalers[variable].inverse_transform(all_inputs[:][:,variable].cpu()))
                    '''
                    if variable in [41, 48, 49, 56]:
                        defaults = abs(scalers[variable].inverse_transform(all_inputs[:,variable].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                    '''        

                    '''
                    # as long as nothing was set to 0 manually, not really necessary
                    vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
                    vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
                    vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
                    vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
                    vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
                    vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
                    if variable in vars_with_0_defaults:
                        defaults = abs(scalers[i].inverse_transform(all_inputs[:,variable].cpu())) < 0.001   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = all_inputs[:,variable][defaults]
                    '''        

                    '''
                        # For cleaned files (QCD or TT to Semileptonic)
                    '''

                    if variable in range(67):
                        defaults = abs(scalers[i].inverse_transform(all_inputs[:,i].cpu()) + 999) < 300   # "floating point error" --> allow some error margin
                        if np.sum(defaults) != 0:
                            xadv[defaults] = scalers[variable].inverse_transform(all_inputs[:,variable].cpu())[defaults]
                    '''        
                    if variable in range(67):
                        defaults = abs(all_inputs[:,variable].cpu() + 900) < 0
                        if np.sum(defaults) != 0:
                            xadv[defaults] = all_inputs[:,variable].cpu()[defaults]
                    '''         

                    xmagn.append(xadv)

            del all_inputs
            del all_inputs_noise
            del noise
            del xadv
            gc.collect()

        return np.array(xmagn)
    
    noise_inputs = np.vstack([apply_noise(magn=[param], offset=[0], variable=p) for p in range(67)])
    noise_inputs = noise_inputs.transpose()
    #noise_inputsNP = noise_inputs
    
    noise_inputs = np.c_[noise_inputs, flav]
    
    auc_ranking(noise_inputs)

# =======================================================================================



# ===================================== FGSM ============================================

if mode == 'FGSM':
    allweights = compute_class_weight(
               'balanced',
                classes=np.array([0,1,2,3]), 
                y=flav-1)
    class_weights = torch.FloatTensor(allweights).to(device)
    del allweights
    gc.collect()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    del class_weights
    gc.collect()

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



    checkpoint = torch.load(f'/home/um106329/aisafety/new_march_21/models/model_all_TT_149_epochs_v10_GPU_weighted_new_49_datasets.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])


    #evaluate network on inputs
    model.to(device)
    #test_inputs.cuda()
    #test_inputs = test_inputs.type(torch.cuda.FloatTensor)
    #print(type(test_inputs))
    #model.eval()
    #predsTensor = model(test_inputs).detach()
    #predictions = predsTensor.cpu().numpy()
    #print('predictions done')

    def fgsm_attack(epsilon=1e-1,sample=None,targets=None,reduced=True, scalers=None):
        xadv = sample.clone().detach()

        # calculate the gradient of the model w.r.t. the *input* tensor:
        # first we tell torch that x should be included in grad computations
        xadv.requires_grad = True

        # then we just do the forward and backwards pass as usual:
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
                #xadv[:,2] = sample[:,2]     # TagVarCSV_jetNSecondaryVertices
                xadv[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')] = sample[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')]
                #xadv[:,5] = sample[:,5]     # TagVarCSV_vertexCategory
                xadv[:,input_names.index('Jet_DeepCSV_vertexCategory')] = sample[:,input_names.index('Jet_DeepCSV_vertexCategory')]
                #xadv[:,10] = sample[:,10]   # TagVarCSV_jetNSelectedTracks
                xadv[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')] = sample[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')]
                #xadv[:,11] = sample[:,11]   # TagVarCSV_jetNTracksEtaRel
                xadv[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')] = sample[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')]
                #xadv[:,59] = sample[:,59]   # TagVarCSV_vertexNTracks
                xadv[:,input_names.index('Jet_DeepCSV_vertexNTracks')] = sample[:,input_names.index('Jet_DeepCSV_vertexNTracks')]
                #xadv[:,12:][sample[:,12:]==0] = 0   # TagVarCSVTrk_trackJetDistVal and so forth, but I have not set any variable to 0 manually during cleaning
                #xadv[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):][sample[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):] == 0] = 0

                '''
                for i in [41, 48, 49, 56]:
                    defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        for i in [41, 48, 49, 56]:
                            xadv[:,i][defaults] = sample[:,i][defaults]
                        break
                vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
                vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
                vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
                vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
                vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
                vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
                for i in vars_with_0_defaults:
                    defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu())) < 0.001   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        for i in vars_with_0_defaults:
                            xadv[:,i][defaults] = sample[:,i][defaults]
                        break
                '''
                for i in range(67):
                    defaults = scalers[i].inverse_transform(sample[:,i].cpu()) + 900 < 0   # "floating point error" --> allow some error margin
                    if np.sum(defaults) != 0:
                        for i in range(67):
                            xadv[:,i][defaults] = sample[:,i][defaults]
                        break
            return xadv.detach()



    def compare_inputs(prop=0,epsilon=0.1,minimum=None,maximum=None,reduced=True):
        xmagn = []
        for s in range(0, NUM_DATASETS):
            scalers = torch.load(scalers_file_paths[s])
            #scalers = all_scalers[s]
            test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
            val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
            train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
            test_targets =  torch.load(test_target_file_paths[s]).to(device)
            val_targets =  torch.load(val_target_file_paths[s]).to(device)
            train_targets =  torch.load(train_target_file_paths[s]).to(device)
            all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))
            all_targets = torch.cat((test_targets,val_targets,train_targets))
            del test_inputs
            del val_inputs
            del train_inputs
            del test_targets
            del val_targets
            del train_targets
            gc.collect()
            #print(f'number of default -1 values for jet variables:\t{np.sum(abs(scalers[41].inverse_transform(all_inputs[:,41].cpu()) + 1.0) < 0.001)}')
            #print(f'percentage of default -1 values for jet variables:\t{np.sum(abs(scalers[41].inverse_transform(all_inputs[:,41].cpu() + 1.0)) < 0.01)/len(all_inputs[:,41].cpu())*100}%')

            for i, m in enumerate(epsilon):
                if s > 0:
                    xadv = np.concatenate((xmagn[i], scalers[prop].inverse_transform(fgsm_attack(epsilon[i],all_inputs,all_targets,reduced=reduced, scalers=scalers, model=method)[:,prop].cpu())))
                    integervars = [59,63,64,65,66]
                    if prop in integervars:
                        xadv = np.rint(xadv)
                    xmagn[i] = xadv
                else:
                    xadv = scalers[prop].inverse_transform(fgsm_attack(epsilon[i],all_inputs,all_targets,reduced=reduced, scalers=scalers)[:,prop].cpu())
                    integervars = [59,63,64,65,66]
                    if prop in integervars:
                        xadv = np.rint(xadv)
                    xmagn.append(xadv)

            del scalers
           
            del all_inputs
            del all_targets
            gc.collect()
        return np.array(xmagn)

    
    fgsm_inputs = np.vstack([compare_inputs(p,[param],minimum=None,maximum=None,reduced=True) for p in range(67)])
    fgsm_inputs = fgsm_inputs.transpose()
    #fgsm_inputsNP = fgsm_inputs
    
    fgsm_inputs = np.c_[fgsm_inputs, flav]

    auc_ranking(fgsm_inputs)








