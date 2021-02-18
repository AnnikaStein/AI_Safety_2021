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
parser.add_argument("fgsmepsilon", type=float, help="The magnitude of the attack (epsilon for the fgsm in quotation marks)")
args = parser.parse_args()

weighting_method = args.weighting
fgsm_epsilon = args.fgsmepsilon

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



#at_epoch = [100]
if args.listepochs == "all":
    at_epoch = [i for i in range(1,121)]
else:
    at_epoch = ast.literal_eval(args.listepochs)
#at_epoch = [20,70,120]


print(f'Evaluate with fgsm at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')
print(f'And epsilon={fgsm_epsilon}')

'''

    Load inputs and targets
    
'''
NUM_DATASETS = 200
#NUM_DATASETS = 1   # defines the number of datasets that shall be used in the evaluation (test), if it is different from the number of files used for training


NUM_DATASETS_TEST = NUM_DATASETS

scalers_file_paths = ['/work/um106329/MA/cleaned/preprocessed/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]


allscalers = [torch.load(scalers_file_paths[s]) for s in range(NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len_test)


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


del test_inputs
del test_targets
del jetFlavour

gc.collect()

with open('/home/um106329/aisafety/length_cleaned_data_test.npy', 'rb') as f:
    length_data_test = np.load(f)

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


def fgsm_attack(epsilon=1e-1,sample=BvsUDSG_inputs,targets=BvsUDSG_targets,reduced=True, model=0, epoch=120):
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # then we just do the forward and backwards pass as usual:
    if model == 0:
        preds = model0[epoch](xadv)
        loss = criterion0(preds, targets.long()).mean()
        model0[epoch].zero_grad()
    #elif model == 1:
    #    preds = model1(xadv)
    #    loss = criterion1(preds, targets.long()).mean()
    #    model1.zero_grad()
    else:
        preds = model2[epoch](xadv)
        loss = criterion2(preds, targets.long()).mean()
        model2[epoch].zero_grad()
    
    
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
            
            for i in [41, 48, 49, 56]:
                defaults = np.zeros(len(sample))
                for l, s in enumerate(length_data_test[:NUM_DATASETS_TEST]):
                    scalers = allscalers[l]
                    if l == 0:
                        defaults[:int(s)] = abs(scalers[i].inverse_transform(sample[:int(s),i].cpu()) + 1.0) < 0.001
                    else:
                        defaults[int(length_data_test[l-1]) : int(s)] = abs(scalers[i].inverse_transform(sample[int(length_data_test[l-1]) : int(s),i].cpu()) + 1.0) < 0.001
                        
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
                defaults = np.zeros(len(sample))
                for l, s in enumerate(length_data_test[:NUM_DATASETS_TEST]):
                    scalers = allscalers[l]
                    if l == 0:
                        defaults[:int(s)] = abs(scalers[i].inverse_transform(sample[:int(s),i].cpu())) < 0.001
                    else:
                        defaults[int(length_data_test[l-1]) : int(s)] = abs(scalers[i].inverse_transform(sample[int(length_data_test[l-1]) : int(s),i].cpu())) < 0.001
                        
                if np.sum(defaults) != 0:
                    for i in vars_with_0_defaults:
                        xadv[:,i][defaults] = sample[:,i][defaults]
                    break
        return xadv.detach()


   
    
def compare_auc(epsilon=0.01,method=0,reduced=True):
    start = time.time()
    ##### CREATING THE AUCs #####
    ### FGSM ###   
    auc_fgsm = []
        
    for ep in range(len(at_epoch)):        
        if method == 0:
            adv_inputs0 = fgsm_attack(epsilon,reduced=reduced,model=0,epoch=ep)
            fgsm_predictions = model0[ep](adv_inputs0).detach().numpy()
        else:
            adv_inputs2 = fgsm_attack(epsilon,reduced=reduced,model=2,epoch=ep)
            fgsm_predictions = model2[ep](adv_inputs2).detach().numpy()

        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions[:,0])
        auc_fgsm.append(metrics.auc(fpr,tpr))
        del fgsm_predictions
        del fpr
        del tpr
        del thresholds
        
    epsilontext = str(epsilon).replace('.','')
    
    with open(f'/home/um106329/aisafety/models/weighted/compare/auc/auc{args.weighting}_fgsm_{epsilontext}_reduced_{reduced}_{args.listepochs}.data', 'wb') as file:
        pickle.dump(auc_fgsm, file)
       
    end = time.time()
    print(f"Time to create FGSM AUCs: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
    start = end
    
        

compare_auc(epsilon=fgsm_epsilon,method=weighting_method)
