import numpy as np

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
colormapping = ['blue','','','','purple','red','chocolate','grey']



at_epoch = 40

NUM_DATASETS = 200

print(f'Evaluate attack at epoch {at_epoch}')


'''

    Load inputs and targets
    
'''
#NUM_DATASETS = 2   # defines the number of datasets that shall be used in the evaluation (test), if it is different from the number of files used for training

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
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
print('DeepCSV test done')

jetFlavour = test_targets+1



'''

    Predictions: Without weighting
    
'''
criterion0 = nn.CrossEntropyLoss()



model0 = nn.Sequential(nn.Linear(67, 100),
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



checkpoint0 = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted_as_is.pt' % NUM_DATASETS, map_location=torch.device(device))
model0.load_state_dict(checkpoint0["model_state_dict"])

model0.to(device)




#evaluate network on inputs
model0.eval()
predictions_as_is = model0(test_inputs).detach().numpy()
print('predictions without weighting done')



'''

    Predictions: With first weighting method
    
'''

'''
# as calculated in dataset_info.ipynb
allweights1 = [0.9393934969162745, 0.9709644530642717, 0.8684253665882813, 0.2212166834311725]
class_weights1 = torch.FloatTensor(allweights1).to(device)

criterion1 = nn.CrossEntropyLoss(weight=class_weights1)



model1 = nn.Sequential(nn.Linear(67, 100),
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



checkpoint1 = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted.pt' % NUM_DATASETS, map_location=torch.device(device))
model1.load_state_dict(checkpoint1["model_state_dict"])

model1.to(device)



#evaluate network on inputs
model1.eval()
predictions = model1(test_inputs).detach().numpy()
print('predictions with first weighting method done')
'''


'''

    Predictions: With new weighting method
    
'''

# as calculated in dataset_info.ipynb
allweights2 = [0.27580367992004956, 0.5756907770526237, 0.1270419391956182, 0.021463603831708488]
class_weights2 = torch.FloatTensor(allweights2).to(device)

criterion2 = nn.CrossEntropyLoss(weight=class_weights2)



model2 = nn.Sequential(nn.Linear(67, 100),
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



checkpoint2 = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted_new.pt' % NUM_DATASETS, map_location=torch.device(device))
model2.load_state_dict(checkpoint2["model_state_dict"])

model2.to(device)




#evaluate network on inputs
model2.eval()
predictions_new = model2(test_inputs).detach().numpy()
print('predictions with new weighting method done')






with open('/home/um106329/aisafety/length_cleaned_data_test.npy', 'rb') as f:
    length_data_test = np.load(f)





BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))
BvsUDSG_predictions_as_is = np.concatenate((predictions_as_is[jetFlavour==1],predictions_as_is[jetFlavour==4]))
#BvsUDSG_predictions = np.concatenate((predictions[jetFlavour==1],predictions[jetFlavour==4]))
BvsUDSG_predictions_new = np.concatenate((predictions_new[jetFlavour==1],predictions_new[jetFlavour==4]))
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
        noise_predictions0 = model0(BvsUDSG_inputs + noise).detach().numpy()
        #noise_predictions1 = model1(BvsUDSG_inputs + noise).detach().numpy()
        noise_predictions2 = model2(BvsUDSG_inputs + noise).detach().numpy()
        
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions0[:,0])
        fprl.append(fpr)
        tprl.append(tpr)
        #fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions1[:,0])
        #fprl.append(fpr)
        #tprl.append(tpr)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions2[:,0])
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
    
    for i in range(len(magn)*3):
        plt.plot(fprl[i],tprl[i],colormapping[i])
    
    #ax = plt.axes([.37, .16, .5, .5])
    #for i in range(len(magn)):
    #    ax.plot(fprl[i],tprl[i],colormapping[i])
        
    #plt.xlim(0,0.1)
    #plt.ylim(0.4,0.9)
    
    sig = magn[-1]
    sigm = str(sig).replace('.','')
    
    #legend = ['undisturbed, no weighting', 'undisturbed, 1 - rel. freq. weighting', 'undisturbed, 1/rel. freq. weighting', 
    #          f'$\sigma={sig}$, no weighting', f'$\sigma={sig}$, 1 - rel. freq. weighting', f'$\sigma={sig}$, 1/rel. freq. weighting'] 
    legend = ['undisturbed, no weighting', 'undisturbed, 1/rel. freq. weighting', 
              f'$\sigma={sig}$, no weighting', f'$\sigma={sig}$, 1/rel. freq. weighting'] 
    plt.legend(legend)
    
    plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/compare_noise_{sigm}.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)


def fgsm_attack(epsilon=1e-1,sample=BvsUDSG_inputs,targets=BvsUDSG_targets,reduced=True, model=0):
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # then we just do the forward and backwards pass as usual:
    if model == 0:
        preds = model0(xadv)
        loss = criterion0(preds, targets.long()).mean()
        model0.zero_grad()
    elif model == 1:
        preds = model1(xadv)
        loss = criterion1(preds, targets.long()).mean()
        model1.zero_grad()
    else:
        preds = model2(xadv)
        loss = criterion2(preds, targets.long()).mean()
        model2.zero_grad()
    
    
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
        adv_inputs0 = fgsm_attack(e,reduced=reduced,model=0)
        #adv_inputs1 = fgsm_attack(e,reduced=reduced,model=1)
        adv_inputs2 = fgsm_attack(e,reduced=reduced,model=2)
        fgsm_predictions0 = model0(adv_inputs0).detach().numpy()
        #fgsm_predictions1 = model1(adv_inputs1).detach().numpy()
        fgsm_predictions2 = model2(adv_inputs2).detach().numpy()
        
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions0[:,0])
        fprl.append(fpr)
        tprl.append(tpr)
        #fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions1[:,0])
        #fprl.append(fpr)
        #tprl.append(tpr)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions2[:,0])
        fprl.append(fpr)
        tprl.append(tpr)
        del adv_inputs0
        del fgsm_predictions0
        #del adv_inputs1
        #del fgsm_predictions1
        del adv_inputs2
        del fgsm_predictions2
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

    for i in range(len(epsilon)//2):
        plt.plot(fprl[i],tprl[i],colormapping[i])
    for j in range(len(epsilon)//2, len(epsilon)):
        plt.plot(fprl[j],tprl[j],colormapping[j],'--')
        
    #ax = plt.axes([.37, .16, .5, .5])
    #for i in range(len(epsilon)):
    #    ax.plot(fprl[i],tprl[i],colormapping[i])
        
    #plt.xlim(0,0.1)
    #plt.ylim(0.4,0.9)
    
    eps = epsilon[-1]
    epsi = str(eps).replace('.','')
    
    #legend = ['undisturbed, no weighting', 'undisturbed, 1 - rel. freq. weighting', 'undisturbed, 1/rel. freq. weighting', 
    #          f'$\epsilon={eps}$, no weighting', f'$\epsilon={eps}$, 1 - rel. freq. weighting', f'$\epsilon={eps}$, 1/rel. freq. weighting'] 
    
    legend = ['undisturbed, no weighting', 'undisturbed, 1/rel. freq. weighting', 
              f'$\epsilon={eps}$, no weighting', f'$\epsilon={eps}$, 1/rel. freq. weighting'] 
    
    
    plt.legend(legend)
    plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/compare_{epsi}_fgsm_reduced_{reduced}_v3.png', bbox_inches='tight', dpi=300)
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)
    
    
    

    
#apply_noise([0,0.1])    

#execute_fgsm([0,0.01,0.02,0.03,0.04,0.05,0.1,0.2],False)

#execute_fgsm([0,0.1],True)
execute_fgsm([0,0.01],True)
