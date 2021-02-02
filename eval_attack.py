import uproot4 as uproot
import numpy as np
import awkward1 as ak

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist

import time


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use(hep.cms.style.ROOT)
C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
colormapping = ['blue','','','','purple','red','chocolate','grey']
colorcode = ['firebrick','magenta','cyan','darkgreen']
path = 'Figures/'



at_epoch = 25

NUM_DATASETS = 200




criterion = nn.CrossEntropyLoss()



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



checkpoint = torch.load(f'/home/um106329/aisafety/models/%d_full_files_{at_epoch}_epochs_v12_GPU_new.pt' % NUM_DATASETS, map_location=device)
#checkpoint = torch.load(f'/home/um106329/aisafety/models/200_full_files_1_epochs_v12_GPU.pt', map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

#NUM_DATASETS = 2   # defines the number of datasets that shall be used in the evaluation (test), if it is different from the number of files used for training

scalers_file_paths = ['/work/um106329/MA/preprocessed_files/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_input_file_paths = ['/work/um106329/MA/preprocessed_files/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/work/um106329/MA/preprocessed_files/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/preprocessed_files/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]


allscalers = [torch.load(scalers_file_paths[s]) for s in range(NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len(test_inputs))


test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
print('DeepCSV test done')

#evaluate network on inputs
model.eval()
predictions = model(test_inputs).detach().numpy()
print('predictions done')


with open('/home/um106329/aisafety/length_data_test.npy', 'rb') as f:
    length_data_test = np.load(f)


jetFlavour = test_targets+1


BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))
BvsUDSG_predictions = np.concatenate((predictions[jetFlavour==1],predictions[jetFlavour==4]))
BvsUDSG_DeepCSV = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==4]))
gc.collect()


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


def apply_noise(magn=[1],offset=[0]):
    
    fprl,tprl = [],[]
    for m in magn:
        noise = torch.Tensor(np.random.normal(offset,m,(len(BvsUDSG_inputs),67)))
        noise_predictions = model(BvsUDSG_inputs + noise).detach().numpy()
        
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions[:,0])
        fprl.append(fpr)
        tprl.append(tpr)
    
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    plt.title(f'ROCs with noise\n After {at_epoch} epochs, evaluated on {len_test} jets')
    
    #plt.plot([0,0.1,0.1,0,0],[0.4,0.4,0.9,0.9,0.4],'--',color='black')
    #plt.plot([0.1,1.01],[0.9,0.735],'--',color='grey')
    #plt.plot([0,0.3],[0.4,0.0],'--',color='grey')
    
    for i in range(len(magn)):
        plt.plot(fprl[i],tprl[i],colormapping[i])
    
    #ax = plt.axes([.37, .16, .5, .5])
    #for i in range(len(magn)):
    #    ax.plot(fprl[i],tprl[i],colormapping[i])
        
    #plt.xlim(0,0.1)
    #plt.ylim(0.4,0.9)
    
    legend = [f'$\sigma={m}$' for m in magn]
    legend[0] = 'undisturbed'
    plt.legend(legend)
    plt.savefig(f'/home/um106329/aisafety/models/compare/after_{at_epoch}/{NUM_DATASETS}_files/noise.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)


def fgsm_attack(epsilon=1e-1,sample=BvsUDSG_inputs,targets=BvsUDSG_targets,reduced=True):
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
            
            xadv[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')] = sample[:,input_names.index('Jet_DeepCSV_jetNSecondaryVertices')]
            xadv[:,input_names.index('Jet_DeepCSV_vertexCategory')] = sample[:,input_names.index('Jet_DeepCSV_vertexCategory')]
            xadv[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')] = sample[:,input_names.index('Jet_DeepCSV_jetNSelectedTracks')]
            xadv[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')] = sample[:,input_names.index('Jet_DeepCSV_jetNTracksEtaRel')]
            xadv[:,input_names.index('Jet_DeepCSV_vertexNTracks')] = sample[:,input_names.index('Jet_DeepCSV_vertexNTracks')]
            #xadv[:,12:][sample[:,12:]==0] = 0   # TagVarCSVTrk_trackJetDistVal, but I have not set any variable to 0 manually during cleaning
            #xadv[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):][sample[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):] == 0] = 0
            
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
    
    
def execute_fgsm(epsilon=[1e-1],reduced=True):
    
    fprl,tprl = [],[]
    for e in epsilon:
        adv_inputs = fgsm_attack(e,reduced=reduced)
        fgsm_predictions = model(adv_inputs).detach().numpy()
        
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions[:,0])
        fprl.append(fpr)
        tprl.append(tpr)
        del adv_inputs
        del fgsm_predictions
        gc.collect()
    
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    if reduced:
        plt.title(f'ROCs with reduced FGSM\n After {at_epoch} epochs, evaluated on {len_test} jets')
    else:
        plt.title(f'ROCs with full FGSM\n After {at_epoch} epochs, evaluated on {len_test} jets')
    
    #plt.plot([0,0.1,0.1,0,0],[0.4,0.4,0.9,0.9,0.4],'--',color='black')
    #plt.plot([0.1,1.01],[0.9,0.735],'--',color='grey')
    #plt.plot([0,0.3],[0.4,0.0],'--',color='grey')

    for i in range(len(epsilon)):
        plt.plot(fprl[i],tprl[i],colormapping[i])
        
    #ax = plt.axes([.37, .16, .5, .5])
    #for i in range(len(epsilon)):
    #    ax.plot(fprl[i],tprl[i],colormapping[i])
        
    #plt.xlim(0,0.1)
    #plt.ylim(0.4,0.9)
    
    legend = [f'$\epsilon={e}$' for e in epsilon]
    legend[0] = 'undisturbed'
    plt.legend(legend)
    plt.savefig(f'/home/um106329/aisafety/models/compare/after_{at_epoch}/{NUM_DATASETS}_files/fgsm_reduced_{reduced}.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)
    
    
    

    
#apply_noise([0,0.1,0.2,0.3,0.4,0.5,1,2])    
execute_fgsm([0,0.01,0.02,0.03,0.04,0.05,0.1,0.2],False)
#execute_fgsm([0,0.01,0.02,0.03,0.04,0.05,0.1,0.2],True)
