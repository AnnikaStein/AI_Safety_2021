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


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use(hep.cms.style.ROOT)
colormapping = ['blue','','','','purple','red','chocolate','grey']
new_colors_all = ['darkblue', 'orange','forestgreen', 'red', 'dodgerblue', 'gold', 'magenta']


#at_epoch = [20,40,60,80,100,120]
#at_epoch = [i for i in range(1,121)]
at_epoch = [20,70,120]

NUM_DATASETS = 200

print(f'Evaluate attack at epoch {at_epoch}')

max_epoch = max(at_epoch)
n_diff_epochs = len(at_epoch)

new_colors = new_colors_all[0:n_diff_epochs] + new_colors_all[0:n_diff_epochs]
print(new_colors)

legDCSV = at_epoch + ['DeepCSV']
print(legDCSV)

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
predictions_as_is = []
for e in range(len(at_epoch)):
    model0[e].load_state_dict(checkpoint0[e]["model_state_dict"])

    model0[e].to(device)




    #evaluate network on inputs
    model0[e].eval()
    predictions_as_is.append(model0[e](test_inputs).detach().numpy())

print('predictions without weighting done')




'''

    Predictions: With new weighting method
    
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
predictions_new = []
for e in range(len(at_epoch)):
    model2[e].load_state_dict(checkpoint2[e]["model_state_dict"])

    model2[e].to(device)




    #evaluate network on inputs
    model2[e].eval()
    predictions_new.append(model2[e](test_inputs).detach().numpy())
    
print('predictions with new weighting method done')






with open('/home/um106329/aisafety/length_cleaned_data_test.npy', 'rb') as f:
    length_data_test = np.load(f)





BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))
BvsUDSG_predictions_as_is = [np.concatenate((predictions_as_is[e][jetFlavour==1],predictions_as_is[e][jetFlavour==4])) for e in range(len(at_epoch))]
BvsUDSG_predictions_new = [np.concatenate((predictions_new[e][jetFlavour==1],predictions_new[e][jetFlavour==4])) for e in range(len(at_epoch))]
BvsUDSG_DeepCSV = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==4]))
fprlDeepCSV,tprlDeepCSV,thresholdsDeepCSV = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0])
aucDeepCSV = metrics.auc(fprlDeepCSV,tprlDeepCSV)
print(f'AUC B vs UDSG, DeepCSV: {aucDeepCSV}')

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

def compare():
    fprl0,tprl0,auc0 = [],[],[]
    fprl2,tprl2,auc2 = [],[],[]
    for e in range(len(at_epoch)):
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[e][:,0])
        fprl0.append(fpr)
        tprl0.append(tpr)
        auc0.append(metrics.auc(fpr,tpr))
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[e][:,0])
        fprl2.append(fpr)
        tprl2.append(tpr)
        auc2.append(metrics.auc(fpr,tpr))
    
    print('AUC B vs UDSG, no weighting')
    for i, ep in enumerate(at_epoch):
        print(f'{ep}:\t {auc0[i]}')
    print('AUC B vs UDSG, 1 / rel. freq. weighting')
    for i, ep in enumerate(at_epoch):
        print(f'{ep}:\t {auc2[i]}')
    
    
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    plt.title(f'ROCs B vs UDSG, evaluated on {len_test} test jets')
    
        
    for i in range(len(at_epoch)):
        plt.plot(fprl0[i],tprl0[i],color=new_colors[i])
         
    
    plt.plot(fprlDeepCSV,tprlDeepCSV)
    
    plt.legend(legDCSV, title = 'Epoch')
    
    #plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{max_epoch}/roc_compare_epoch_as_is_{n_diff_epochs}_different_epochs_v2.svg', bbox_inches='tight')
    plt.savefig(f'/home/um106329/aisafety/dpg21/after_{max_epoch}_roc_compare_epoch_as_is_{n_diff_epochs}_different_epochs_v2.svg', bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)
                   
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    plt.title(f'ROCsB vs UDSG, evaluated on {len_test} test jets\n 1 / rel. freq. weighting')
    
        
    for i in range(len(at_epoch)):
        plt.plot(fprl2[i],tprl2[i],color=new_colors[i])
         
    plt.plot(fprlDeepCSV,tprlDeepCSV)
    
    plt.legend(legDCSV, title = 'Epoch')
    
    #plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{max_epoch}/roc_compare_epoch_new_{n_diff_epochs}_different_epochs_v2.svg', bbox_inches='tight')
    plt.savefig(f'/home/um106329/aisafety/dpg21/after_{max_epoch}_roc_compare_epoch_new_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)

def apply_noise(magn=[1],offset=[0]):
    fprl0,tprl0,auc0 = [],[],[]
    fprl2,tprl2,auc2 = [],[],[]
    for e in range(len(at_epoch)):
        for m in magn:
            noise = torch.Tensor(np.random.normal(offset,m,(len(BvsUDSG_inputs),67)))
            noise_predictions0 = model0[e](BvsUDSG_inputs + noise).detach().numpy()
            noise_predictions2 = model2[e](BvsUDSG_inputs + noise).detach().numpy()

            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions0[:,0])
            fprl0.append(fpr)
            tprl0.append(tpr)
            auc0.append(metrics.auc(fpr,tpr))
            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions2[:,0])
            fprl2.append(fpr)
            tprl2.append(tpr)
            auc2.append(metrics.auc(fpr,tpr))
    
    sig = magn[-1]
    sigm = str(sig).replace('.','')
    
    print('AUC B vs UDSG, no weighting')
    for i, ep in enumerate(at_epoch):
        print(f'{ep}, \t Undisturbed:\t {auc0[2*i]}')
        print(f'{ep}, \t With noise (sigma={sig}):\t {auc0[2*i+1]}')
    print('AUC B vs UDSG, 1 / rel. freq. weighting')
    for i, ep in enumerate(at_epoch):
        print(f'{ep}, \t Undisturbed:\t {auc2[2*i]}')
        print(f'{ep}, \t With noise (sigma={sig}):\t {auc2[2*i+1]}')
    
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    plt.title(f'ROCs B vs UDSG with noise, evaluated on {len_test} jets')
    
    
    for i in range(len(at_epoch)):
        plt.plot(fprl0[2*i],tprl0[2*i],color=new_colors[i],label=f'{at_epoch[i]}, Undisturbed')
        plt.plot(fprl0[2*i+1],tprl0[2*i+1],linestyle='dashed',color=new_colors[i],label=f'{at_epoch[i]}, $\sigma={sig}$')
    
    
    plt.plot(fprlDeepCSV,tprlDeepCSV,label='DeepCSV')
    plt.legend(title='Epoch')
    
    #plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{max_epoch}/compare_noise_{sigm}_as_is_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.savefig(f'/home/um106329/aisafety/dpg21/after_{max_epoch}_compare_noise_{sigm}_as_is_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)


    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    plt.title(f'ROCs B vs UDSG with noise, evaluated on {len_test} jets\n 1 / rel. freq. weighting')
    
    
    for i in range(len(at_epoch)):
        plt.plot(fprl2[2*i],tprl2[2*i],color=new_colors[i],label=f'{at_epoch[i]}, Undisturbed')
        plt.plot(fprl2[2*i+1],tprl2[2*i+1],linestyle='dashed',color=new_colors[i],label=f'{at_epoch[i]}, $\sigma={sig}$')
    
    plt.plot(fprlDeepCSV,tprlDeepCSV,label='DeepCSV')
    plt.legend(title='Epoch')
    
    
    #plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{max_epoch}/compare_noise_{sigm}_new_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.savefig(f'/home/um106329/aisafety/dpg21/after_{max_epoch}_compare_noise_{sigm}_new_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)
    
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
            vars_with_0_defaults = [6, 7, 8, 9, 10, 11]                 # trackDecayLenVal_0 to _5
            vars_with_0_defaults.extend([12, 13, 14, 15, 16, 17])       # trackDeltaR_0 to _5
            vars_with_0_defaults.extend([18, 19, 20, 21])               # trackEtaRel_0 to _3
            vars_with_0_defaults.extend([22, 23, 24, 25, 26, 27])       # trackJetDistVal_0 to _5
            vars_with_0_defaults.extend([29, 30, 31, 32, 33, 34])       # trackPtRatio_0 to _5
            vars_with_0_defaults.extend([35, 36, 37, 38, 39, 40])       # trackPtRel_0 to _5
            for i in vars_with_0_defaults:
                defaults = np.zeros(len(sample))
                for l, s in enumerate(length_data_test[:NUM_DATASETS]):
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
    
    
def execute_fgsm(epsilon=[1e-1],reduced=True):
    fprl0,tprl0,auc0 = [],[],[]
    fprl2,tprl2,auc2 = [],[],[]
    for ep in range(len(at_epoch)):
        for e in epsilon:
            adv_inputs0 = fgsm_attack(e,reduced=reduced,model=0,epoch=ep)
            adv_inputs2 = fgsm_attack(e,reduced=reduced,model=2,epoch=ep)
            fgsm_predictions0 = model0[ep](adv_inputs0).detach().numpy()
            fgsm_predictions2 = model2[ep](adv_inputs2).detach().numpy()

            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions0[:,0])
            fprl0.append(fpr)
            tprl0.append(tpr)
            auc0.append(metrics.auc(fpr,tpr))
            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions2[:,0])
            fprl2.append(fpr)
            tprl2.append(tpr)
            auc2.append(metrics.auc(fpr,tpr))
            del adv_inputs0
            del fgsm_predictions0
            del adv_inputs2
            del fgsm_predictions2
            gc.collect()
    
    
    eps = epsilon[-1]
    epsi = str(eps).replace('.','')
    
    
    print('AUC B vs UDSG, no weighting')
    for i, ep in enumerate(at_epoch):
        print(f'{ep}, \t Undisturbed:\t {auc0[2*i]}')
        print(f'{ep}, \t FGSM (epsilon={eps}):\t {auc0[2*i+1]}')
    print('AUC B vs UDSG, 1 / rel. freq. weighting')
    for i, ep in enumerate(at_epoch):
        print(f'{ep}, \t Undisturbed:\t {auc2[2*i]}')
        print(f'{ep}, \t FGSM (epsilon={eps}):\t {auc2[2*i+1]}')
    
    
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    if reduced:
        plt.title(f'ROCs B vs UDSG with reduced FGSM\n Evaluated on {len_test} jets')
    else:
        plt.title(f'ROCs B vs UDSG with full FGSM\n Evaluated on {len_test} jets')
  
        
    for i in range(len(at_epoch)):
        plt.plot(fprl0[2*i],tprl0[2*i],color=new_colors[i],label=f'{at_epoch[i]}, Undisturbed')
        plt.plot(fprl0[2*i+1],tprl0[2*i+1],linestyle='dashed',color=new_colors[i],label=f'{at_epoch[i]}, $\epsilon={eps}$')
    
    plt.plot(fprlDeepCSV,tprlDeepCSV,label='DeepCSV')    
    plt.legend(title='Epoch')
    plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{max_epoch}/compare_{epsi}_fgsm_reduced_{reduced}_as_is_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)
    
    
    plt.figure(5,[15,15])
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.xlabel('mistag rate')
    plt.ylabel('efficiency')
    if reduced:
        plt.title(f'ROCs B vs UDSG with reduced FGSM\n Evaluated on {len_test} jets, 1 / rel. freq. weighting')
    else:
        plt.title(f'ROCs B vs UDSG with full FGSM\n Evaluated on {len_test} jets, 1 / rel. freq. weighting')
  
    for i in range(len(at_epoch)):
        plt.plot(fprl2[2*i],tprl2[2*i],color=new_colors[i],label=f'{at_epoch[i]}, Undisturbed')
        plt.plot(fprl2[2*i+1],tprl2[2*i+1],linestyle='dashed',color=new_colors[i],label=f'{at_epoch[i]}, $\epsilon={eps}$')
    
    plt.plot(fprlDeepCSV,tprlDeepCSV,label='DeepCSV')    
    plt.legend(title='Epoch')
    #plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{max_epoch}/compare_{epsi}_fgsm_reduced_{reduced}_new_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.savefig(f'/home/um106329/aisafety/dpg21/after_{max_epoch}_compare_{epsi}_fgsm_reduced_{reduced}_new_{n_diff_epochs}_different_epochs_v5.svg', bbox_inches='tight')
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    gc.collect(2)
    

#compare()    
apply_noise([0,0.3])    

#execute_fgsm([0,0.01,0.02,0.03,0.04,0.05,0.1,0.2],False)


#execute_fgsm([0,0.01],True)
#execute_fgsm([0,0.1],True)



def compare_auc(sigmas=[0.1],epsilons=[0.01],reduced=True,offset=[0]):
    start = time.time()
    ##### CREATING THE AUCs #####
    ### RAW ###
    auc0_raw = []
    auc2_raw = []
    for ep in range(len(at_epoch)):
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[ep][:,0])
        auc0_raw.append(metrics.auc(fpr,tpr))
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[ep][:,0])
        auc2_raw.append(metrics.auc(fpr,tpr))
    
    with open('/home/um106329/aisafety/models/weighted/compare/auc/auc0_raw.data', 'wb') as file:
        pickle.dump(auc0_raw, file)
    
    with open('/home/um106329/aisafety/models/weighted/compare/auc/auc2_raw.data', 'wb') as file:
        pickle.dump(auc2_raw, file)
    
    end = time.time()
    print(f"Time to create raw AUCs: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
    start = end
    
        
    ### NOISE ###    
    auc0_noise = []
    auc2_noise = []
    for s in sigmas:
        auc0_noise_s = []
        auc2_noise_s = []
        for ep in range(len(at_epoch)):
            noise = torch.Tensor(np.random.normal(offset,s,(len(BvsUDSG_inputs),67)))
            noise_predictions0 = model0[ep](BvsUDSG_inputs + noise).detach().numpy()
            noise_predictions2 = model2[ep](BvsUDSG_inputs + noise).detach().numpy()

            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions0[:,0])
            auc0_noise_s.append(metrics.auc(fpr,tpr))
            
            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],noise_predictions2[:,0])
            auc2_noise_s.append(metrics.auc(fpr,tpr))
        auc0_noise.append(auc0_noise_e)
        auc2_noise.append(auc2_noise_e)
    
    with open('/home/um106329/aisafety/models/weighted/compare/auc/auc0_noise.data', 'wb') as file:
        pickle.dump(auc0_noise, file)
    
    with open('/home/um106329/aisafety/models/weighted/compare/auc/auc2_noise.data', 'wb') as file:
        pickle.dump(auc2_noise, file)
    
    
    auc0_noise_diff = [[auc0_noise[s][ep] - auc0_raw[ep] for ep in range(len(at_epoch))] for s in range(len(sigmas))]
    auc2_noise_diff = [[auc2_noise[s][ep] - auc2_raw[ep] for ep in range(len(at_epoch))] for s in range(len(sigmas))]
    auc0_noise_diff_rel = [[(auc0_noise[s][ep] - auc0_raw[ep]) / auc0_raw[ep] for ep in range(len(at_epoch))] for s in range(len(sigmas))]
    auc2_noise_diff_rel = [[(auc2_noise[s][ep] - auc2_raw[ep]) / auc2_raw[ep] for ep in range(len(at_epoch))] for s in range(len(sigmas))]
    auc0_noise_rel = [[auc0_noise[s][ep] / auc0_raw[ep] for ep in range(len(at_epoch))] for s in range(len(sigmas))]
    auc2_noise_rel = [[auc2_noise[s][ep] / auc2_raw[ep] for ep in range(len(at_epoch))] for s in range(len(sigmas))]
    
    
    end = time.time()
    print(f"Time to create Noise AUCs: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
    start = end
    
    
    ### FGSM ###
    '''
    auc0_fgsm = []
    auc2_fgsm = []
    for e in epsilons:
        auc0_fgsm_e = []
        auc2_fgsm_e = []
        for ep in range(len(at_epoch)):
            adv_inputs0 = fgsm_attack(e,reduced=reduced,model=0,epoch=ep)
            adv_inputs2 = fgsm_attack(e,reduced=reduced,model=2,epoch=ep)
            fgsm_predictions0 = model0[ep](adv_inputs0).detach().numpy()
            fgsm_predictions2 = model2[ep](adv_inputs2).detach().numpy()

            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions0[:,0])
            auc0_fgsm_e.append(metrics.auc(fpr,tpr))
            auc0_fgsm_e.append(ep)
            fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],fgsm_predictions2[:,0])
            auc2_fgsm_e.append(metrics.auc(fpr,tpr))
            auc2_fgsm_e.append(ep)
            del adv_inputs0
            del fgsm_predictions0
            del adv_inputs2
            del fgsm_predictions2
            gc.collect()
        auc0_fgsm.append(auc0_fgsm_e)
        auc2_fgsm.append(auc2_fgsm_e)
    #print(len(at_epoch), len(auc0_raw))
    #print(len(at_epoch), len(auc0_fgsm))
    auc0_fgsm_diff = [[auc0_fgsm[s][ep] - auc0_raw[ep] for ep in range(len(at_epoch))] for s in range(len(epsilons))]
    auc2_fgsm_diff = [[auc2_fgsm[s][ep] - auc2_raw[ep] for ep in range(len(at_epoch))] for s in range(len(epsilons))]
    auc0_fgsm_diff_rel = [[(auc0_fgsm[s][ep] - auc0_raw[ep]) / auc0_raw[ep] for ep in range(len(at_epoch))] for s in range(len(epsilons))]
    auc2_fgsm_diff_rel = [[(auc2_fgsm[s][ep] - auc2_raw[ep]) / auc2_raw[ep] for ep in range(len(at_epoch))] for s in range(len(epsilons))]
    auc0_fgsm_rel = [[auc0_fgsm[s][ep] / auc0_raw[ep] for ep in range(len(at_epoch))] for s in range(len(epsilons))]
    auc2_fgsm_rel = [[auc2_fgsm[s][ep] / auc2_raw[ep] for ep in range(len(at_epoch))] for s in range(len(epsilons))]
    
    end = time.time()
    print(f"Time to create FGSM AUCs: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
    start = end
    '''
    
    
    ######### PLOTTING #########
    ## Just AUC over Epoch #####
    
    ## "Delta AUC" over Epoch ##

    
    '''
    One plot per fixed weighting method.
        For the different sigmas (epsilons), plot the difference of the disturbed AUC and the raw AUC per epoch.
        That would be one line per value for sigma or epsilon on each plot.
    '''
    
    
    # Noise
    for method in [0,2]:
        plt.figure(5,[15,15])
        plt.xlabel('Epoch')
        plt.ylabel('Difference disturbed AUC to raw AUC')
        if method == 0:
            plt.title(f'Difference disturbed to raw AUC B vs UDSG with noise\n Evaluated on {len_test} jets')
        else:
            plt.title(f'Difference disturbed to raw AUC B vs UDSG with noise\n Evaluated on {len_test} jets, 1 / rel. freq. weighting')
        for i, s in enumerate(sigmas):
            if method == 0:
                plt.plot(at_epoch,auc0_noise_diff[i],label=f'$\sigma={s}$')
            else:
                plt.plot(at_epoch,auc2_noise_diff[i],label=f'$\sigma={s}$')

        if method == 0:
            plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/compare_auc_as_is_diff_noise_v1.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/compare_auc_new_diff_noise_v1.png', bbox_inches='tight', dpi=300)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
    '''        
    # FGSM
    for method in [0,2]:
        plt.figure(5,[15,15])
        plt.xlabel('Epoch')
        plt.ylabel('Difference disturbed AUC to raw AUC')
        if reduced:
            if method == 0:
                plt.title(f'Difference disturbed to raw AUC B vs UDSG with reduced FGSM\n Evaluated on {len_test} jets')
            else:
                plt.title(f'Difference disturbed to raw AUC B vs UDSG with reduced FGSM\n Evaluated on {len_test} jets, 1 / rel. freq. weighting')
        else:
            if method == 0:
                plt.title(f'Difference disturbed to raw AUC B vs UDSG with full FGSM\n Evaluated on {len_test} jets')
            else:
                plt.title(f'Difference disturbed to raw AUC B vs UDSG with full FGSM\n Evaluated on {len_test} jets, 1 / rel. freq. weighting')
        for i, e in enumerate(epsilons):
            if method == 0:
                plt.plot(at_epoch,auc0_fgsm_diff[i],label=f'$\epsilon={e}$')
            else:
                plt.plot(at_epoch,auc2_fgsm_diff[i],label=f'$\epsilon={e}$')

        if method == 0:
            plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/compare_auc_as_is_diff_fgsm_reduced_{reduced}_v1.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'/home/um106329/aisafety/models/weighted/compare/compare_auc_new_diff_fgsm_reduced_{reduced}_v1.png', bbox_inches='tight', dpi=300)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
     '''
        
#compare_auc([0.03,0.05,0.1,0.3],[0.01,0.03,0.05,0.1])
#compare_auc([0.03,0.05,0.1,0.3],[0.01,0.1])
