import uproot4 as uproot
import numpy as np
#import awkward1 as ak

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist


plt.style.use(hep.cms.style.ROOT)
C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
path = 'Figures/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

at_epoch = 100

#model = torch.load(f'/home/um106329/aisafety/models/200_full_files_1_epochs_v12_GPU_justModel.pt', map_location=torch.device(device))
#model = torch.load(f'~/aisafety/models/200_full_files_{at_epoch}_epochs_v12_GPU_justModel.pt', map_location=torch.device(device))
criterion = nn.CrossEntropyLoss()


NUM_DATASETS = 200


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

checkpoint = torch.load(f'/home/um106329/aisafety/models/%d_full_files_{at_epoch}_epochs_v12_GPU_new.pt' % NUM_DATASETS)
#checkpoint = torch.load(f'/home/um106329/aisafety/models/200_full_files_2_epochs_v12_GPU_old.pt')
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

# scalers
scalers_file_paths = ['/work/um106329/MA/preprocessed_files/scalers_%d.pt' % k for k in range(0,NUM_DATASETS)]
#scalers = np.concatenate([torch.load(ti) for ti in scalers_file_paths])
#all_scalers = [torch.load(scalers_file_paths[s]) for s in range(NUM_DATASETS)]

test_input_file_paths = ['/work/um106329/MA/preprocessed_files/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
test_target_file_paths = ['/work/um106329/MA/preprocessed_files/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/preprocessed_files/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]
val_input_file_paths = ['/work/um106329/MA/preprocessed_files/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
val_target_file_paths = ['/work/um106329/MA/preprocessed_files/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_input_file_paths = ['/work/um106329/MA/preprocessed_files/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_target_file_paths = ['/work/um106329/MA/preprocessed_files/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]

'''
# test input, truth
test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
print('DeepCSV test done')


# val input, truth
val_inputs = torch.cat(tuple(torch.load(ti) for ti in val_input_file_paths)).float()
print('val inputs done')
val_targets = torch.cat(tuple(torch.load(ti) for ti in val_target_file_paths)).float()
print('val targets done')


# train input, truth
train_inputs = torch.cat(tuple(torch.load(ti) for ti in train_input_file_paths)).float()
print('train inputs done')
train_targets = torch.cat(tuple(torch.load(ti) for ti in train_target_file_paths)).float()
print('train targets done')

with open('/home/um106329/aisafety/length_data_in.npy', 'rb') as f:
    length_data_train = np.load(f)
    
with open('/home/um106329/aisafety/length_data_val.npy', 'rb') as f:
    length_data_val = np.load(f)    

with open('/home/um106329/aisafety/length_data_test.npy', 'rb') as f:
    length_data_test = np.load(f)
'''



#test_inputs =  torch.load(test_input_file_paths[0]).to(device).float()
#DeepCSV_testset = torch.load(DeepCSV_testset_file_paths[0])
#val_inputs =  torch.load(val_input_file_paths[0]).to(device).float()
#train_inputs =  torch.load(train_input_file_paths[0]).to(device).float()
#test_targets =  torch.load(test_target_file_paths[0]).to(device).float()
#val_targets =  torch.load(val_target_file_paths[0]).to(device).float()
#train_targets =  torch.load(train_target_file_paths[0]).to(device).float()


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
                'Jet N Secondary Vertices','Jet N Selected Tracks','Jet N Tracks $\eta_{rel}$','Vertex N Tracks',]


colormapping = ['blue','','','','purple','red','chocolate','grey']
colorcode = ['firebrick','magenta','cyan','darkgreen']


#evaluate network on inputs
model.eval()
#predictions = model(test_inputs).detach()..cpu()numpy()

'''
jetFlavour = test_targets+1

BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
del gc.garbage[:]
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))
BvsUDSG_predictions = np.concatenate((predictions[jetFlavour==1],predictions[jetFlavour==4]))
del predictions
BvsUDSG_DeepCSV = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==4]))
del DeepCSV_testset
gc.collect()
'''


def fgsm_attack(epsilon=1e-1,sample=None,targets=None,reduced=True, scalers=None):
    xadv = sample.clone().detach()
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # then we just do the forward and backwards pass as usual:
    preds = model(xadv)
    loss = criterion(preds, targets).mean()
    
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
            #xadv[:,12:][sample[:,12:]==0] = 0   # TagVarCSVTrk_trackJetDistVal, but I have not set any variable to 0 manually during cleaning
            #xadv[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):][sample[:,input_names.index('Jet_DeepCSV_trackJetDistVal_0'):] == 0] = 0
            #for i in range(6,10):
            #    defaults = scalers[i].inverse_transform(sample[:,i]) == -1 #this number might change due to some weird rounding error
            #    if np.sum(defaults) != 0:
            #        for i in range(6,10):
            #            xadv[:,i][defaults] = sample[:,i][defaults]
            #        break
            for i in [41, 48, 49, 56]:
                defaults = abs(scalers[i].inverse_transform(sample[:,i].cpu()) + 1.0) < 0.001   # "floating point error" --> allow some error margin
                if np.sum(defaults) != 0:
                    for i in [41, 48, 49, 56]:
                        xadv[:,i][defaults] = sample[:,i][defaults]
                    break
        return xadv.detach()
    
    
'''
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
        plt.title('ROCs with reduced FGSM')
    else:
        plt.title('ROCs with full FGSM')
    
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
'''        
    

def compare_inputs(prop=0,epsilon=0.1,minimum=None,maximum=None,reduced=True):
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
        #print(f'number of default -1 values for jet variables:\t{np.sum(abs(scalers[41].inverse_transform(all_inputs[:,41].cpu()) + 1.0) < 0.001)}')
        #print(f'percentage of default -1 values for jet variables:\t{np.sum(abs(scalers[41].inverse_transform(all_inputs[:,41].cpu() + 1.0)) < 0.01)/len(all_inputs[:,41].cpu())*100}%')
        if s > 0:
            x = np.concatenate((x, scalers[prop].inverse_transform(all_inputs[:][:,prop].cpu())))
            xadv = np.concatenate((xadv, scalers[prop].inverse_transform(fgsm_attack(epsilon,all_inputs,all_targets,reduced=reduced, scalers=scalers)[:,prop].cpu())))
        else:
            x = scalers[prop].inverse_transform(all_inputs[:][:,prop].cpu())
            xadv = scalers[prop].inverse_transform(fgsm_attack(epsilon,all_inputs,all_targets,reduced=reduced, scalers=scalers)[:,prop].cpu())

        del scalers
        del test_inputs
        del val_inputs
        del train_inputs
        del test_targets
        del val_targets
        del train_targets
        del all_inputs
        del all_targets
        gc.collect()
    
    if minimum is None:
        minimum = min(min(x),min(xadv))
    if maximum is None:
        maximum = max(max(x),max(xadv))
        
    compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",display_names[prop],100,minimum,maximum))
    compHist.fill(sample="raw",prop=x)
    compHist.fill(sample=f"fgsm $\epsilon$={epsilon}",prop=xadv)
    
    bins = np.linspace(minimum+(maximum-minimum)/100/2,maximum-(maximum-minimum)/100/2,100)
    num = compHist[f"fgsm $\epsilon$={epsilon}"].sum('sample').values()[()]
    denom = compHist['raw'].sum('sample').values()[()]
    ratio = num / denom
    num_err = np.sqrt(num)
    denom_err = np.sqrt(denom)
    ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,5],gridspec_kw={'height_ratios': [3, 1],'hspace': .3})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['blue','red']})
    ax1.get_legend().remove()
    ax1.legend(['FGSM','raw'])
    ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='black')
    ax2.plot([minimum,maximum],[1,1],color='black')
    ax2.set_ylim(0,2)
    ax2.set_xlim(minimum,maximum)
    ax2.set_ylabel('FGSM/raw')
    fig.savefig(f'/home/um106329/aisafety/models/compare/after_{at_epoch}/{NUM_DATASETS}_files/compare_0005/{prop}_{input_names[prop]}_reduced_{reduced}_new.png', bbox_inches='tight', dpi=300)
    del fig, ax1, ax2
    gc.collect(2)
    

epsilon = 0.005
'''
#for i in range(0,67):
#    compare_inputs(i,epsilon,minimum=None,maximum=None,reduced=False)
#    compare_inputs(i,epsilon,minimum=None,maximum=None,reduced=True)

# Jet eta
#compare_inputs(0,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(0,epsilon,minimum=None,maximum=None,reduced=True)

# Jet pt
#compare_inputs(1,epsilon,minimum=None,maximum=1000,reduced=False)
compare_inputs(1,epsilon,minimum=None,maximum=1000,reduced=True)

# flightDist2DSig
#compare_inputs(2,epsilon,minimum=None,maximum=80,reduced=False)
compare_inputs(2,epsilon,minimum=None,maximum=80,reduced=True)

# flightDist2DVal
#compare_inputs(3,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(3,epsilon,minimum=None,maximum=None,reduced=True)

# flightDist3DSig
#compare_inputs(4,epsilon,minimum=None,maximum=80,reduced=False)
compare_inputs(4,epsilon,minimum=None,maximum=80,reduced=True)

# flightDist3DVal
#compare_inputs(5,epsilon,minimum=None,maximum=3.5,reduced=False)
compare_inputs(5,epsilon,minimum=None,maximum=3.5,reduced=True)

# trackDecayLenVal
#compare_inputs(6,epsilon,minimum=-0.1,maximum=5,reduced=False)
compare_inputs(6,epsilon,minimum=-0.1,maximum=5,reduced=True)

#compare_inputs(7,epsilon,minimum=-0.1,maximum=5,reduced=False)
compare_inputs(7,epsilon,minimum=-0.1,maximum=5,reduced=True)

#compare_inputs(8,epsilon,minimum=-0.1,maximum=5,reduced=False)
compare_inputs(8,epsilon,minimum=-0.1,maximum=5,reduced=True)

#compare_inputs(9,epsilon,minimum=-0.1,maximum=5,reduced=False)
compare_inputs(9,epsilon,minimum=-0.1,maximum=5,reduced=True)

#compare_inputs(10,epsilon,minimum=-0.1,maximum=5,reduced=False)
compare_inputs(10,epsilon,minimum=-0.1,maximum=5,reduced=True)

#compare_inputs(11,epsilon,minimum=-0.1,maximum=5,reduced=False)
compare_inputs(11,epsilon,minimum=-0.1,maximum=5,reduced=True)

#
# trackDeltaR
#compare_inputs(12,epsilon,minimum=0,maximum=0.301,reduced=False)
compare_inputs(12,epsilon,minimum=0,maximum=0.301,reduced=True)

#compare_inputs(13,epsilon,minimum=0,maximum=0.301,reduced=False)
compare_inputs(13,epsilon,minimum=0,maximum=0.301,reduced=True)

#compare_inputs(14,epsilon,minimum=0,maximum=0.5,reduced=False)
compare_inputs(14,epsilon,minimum=0,maximum=0.5,reduced=True)

#compare_inputs(15,epsilon,minimum=0,maximum=0.5,reduced=False)
compare_inputs(15,epsilon,minimum=0,maximum=0.5,reduced=True)

#compare_inputs(16,epsilon,minimum=0,maximum=0.5,reduced=False)
compare_inputs(16,epsilon,minimum=0,maximum=0.5,reduced=True)

#compare_inputs(17,epsilon,minimum=0,maximum=0.5,reduced=False)
compare_inputs(17,epsilon,minimum=0,maximum=0.5,reduced=True)


# trackEtaRel
#compare_inputs(18,epsilon,minimum=0,maximum=9,reduced=False)
compare_inputs(18,epsilon,minimum=0,maximum=9,reduced=True)

#compare_inputs(19,epsilon,minimum=0,maximum=9,reduced=False)
compare_inputs(19,epsilon,minimum=0,maximum=9,reduced=True)

#compare_inputs(20,epsilon,minimum=0,maximum=9,reduced=False)
compare_inputs(20,epsilon,minimum=0,maximum=9,reduced=True)

#compare_inputs(21,epsilon,minimum=0,maximum=9,reduced=False)
compare_inputs(21,epsilon,minimum=0,maximum=9,reduced=True)


# trackJetDistVal
#compare_inputs(22,epsilon,minimum=-0.08,maximum=0.0025,reduced=False)
compare_inputs(22,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

#compare_inputs(23,epsilon,minimum=-0.08,maximum=0.0025,reduced=False)
compare_inputs(23,epsilon,minimum=-0.08,maximum=0.0025,reduced=True)

#compare_inputs(24,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
compare_inputs(24,epsilon,minimum=-0.1,maximum=0.01,reduced=True)

#compare_inputs(25,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
compare_inputs(25,epsilon,minimum=-0.1,maximum=0.01,reduced=True)

#compare_inputs(26,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
compare_inputs(26,epsilon,minimum=-0.1,maximum=0.01,reduced=True)

#compare_inputs(27,epsilon,minimum=-0.1,maximum=0.01,reduced=False)
compare_inputs(27,epsilon,minimum=-0.1,maximum=0.01,reduced=True)

#
# trackJetPt
#compare_inputs(28,epsilon,minimum=None,maximum=575,reduced=False)
compare_inputs(28,epsilon,minimum=None,maximum=575,reduced=True)


# trackPtRatio
#compare_inputs(29,epsilon,minimum=0,maximum=0.301,reduced=False)
compare_inputs(29,epsilon,minimum=0,maximum=0.301,reduced=True)

#compare_inputs(30,epsilon,minimum=0,maximum=0.301,reduced=False)
compare_inputs(30,epsilon,minimum=0,maximum=0.301,reduced=True)

#compare_inputs(31,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
compare_inputs(31,epsilon,minimum=-0.05,maximum=0.4,reduced=True)

#compare_inputs(30,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
compare_inputs(32,epsilon,minimum=-0.05,maximum=0.4,reduced=True)

#compare_inputs(33,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
compare_inputs(33,epsilon,minimum=-0.05,maximum=0.4,reduced=True)

#compare_inputs(34,epsilon,minimum=-0.05,maximum=0.4,reduced=False)
compare_inputs(34,epsilon,minimum=-0.05,maximum=0.4,reduced=True)

'''
# trackPtRel
#compare_inputs(35,epsilon,minimum=-0.1,maximum=6,reduced=False)
compare_inputs(35,epsilon,minimum=-0.1,maximum=6,reduced=True)
'''
#compare_inputs(36,epsilon,minimum=-0.1,maximum=6,reduced=False)
compare_inputs(36,epsilon,minimum=-0.1,maximum=6,reduced=True)

#compare_inputs(37,epsilon,minimum=-0.1,maximum=6,reduced=False)
compare_inputs(37,epsilon,minimum=-0.1,maximum=6,reduced=True)

#compare_inputs(38,epsilon,minimum=-0.1,maximum=6,reduced=False)
compare_inputs(38,epsilon,minimum=-0.1,maximum=6,reduced=True)

#compare_inputs(39,epsilon,minimum=-0.1,maximum=6,reduced=False)
compare_inputs(39,epsilon,minimum=-0.1,maximum=6,reduced=True)

#compare_inputs(40,epsilon,minimum=-0.1,maximum=6,reduced=False)
compare_inputs(40,epsilon,minimum=-0.1,maximum=6,reduced=True)


# trackSip2d (SigAboveCharm, Sig, ValAbove Charm)
#compare_inputs(41,epsilon,minimum=-5,maximum=20,reduced=False)
compare_inputs(41,epsilon,minimum=-5,maximum=20,reduced=True)

#compare_inputs(42,epsilon,minimum=-5,maximum=20,reduced=False)
compare_inputs(42,epsilon,minimum=-5,maximum=20,reduced=True)

#compare_inputs(43,epsilon,minimum=-5,maximum=20,reduced=False)
compare_inputs(43,epsilon,minimum=-5,maximum=20,reduced=True)
#
#compare_inputs(44,epsilon,minimum=-20,maximum=20,reduced=False)
compare_inputs(44,epsilon,minimum=-20,maximum=20,reduced=True)

#compare_inputs(45,epsilon,minimum=-20,maximum=20,reduced=False)
compare_inputs(45,epsilon,minimum=-20,maximum=20,reduced=True)

#compare_inputs(46,epsilon,minimum=-20,maximum=20,reduced=False)
compare_inputs(46,epsilon,minimum=-20,maximum=20,reduced=True)

#compare_inputs(47,epsilon,minimum=-20,maximum=20,reduced=False)
compare_inputs(47,epsilon,minimum=-20,maximum=20,reduced=True)

#compare_inputs(48,epsilon,minimum=-2.1,maximum=0.1,reduced=False)
compare_inputs(48,epsilon,minimum=-1,maximum=0.1,reduced=True)


# trackSip3d (SigAboveCharm, Sig, ValAbove Charm)
#compare_inputs(49,epsilon,minimum=-5,maximum=20,reduced=False)
compare_inputs(49,epsilon,minimum=-5,maximum=20,reduced=True)

#compare_inputs(50,epsilon,minimum=-10,maximum=40,reduced=False)
compare_inputs(50,epsilon,minimum=-10,maximum=40,reduced=True)

#compare_inputs(51,epsilon,minimum=-10,maximum=40,reduced=False)
compare_inputs(51,epsilon,minimum=-10,maximum=40,reduced=True)

#compare_inputs(52,epsilon,minimum=-25,maximum=75,reduced=False)
compare_inputs(52,epsilon,minimum=-25,maximum=75,reduced=True)

#compare_inputs(53,epsilon,minimum=-25,maximum=75,reduced=False)
compare_inputs(53,epsilon,minimum=-25,maximum=75,reduced=True)

#compare_inputs(54,epsilon,minimum=-25,maximum=75,reduced=False)
compare_inputs(54,epsilon,minimum=-25,maximum=75,reduced=True)

#compare_inputs(55,epsilon,minimum=-25,maximum=75,reduced=False)
compare_inputs(55,epsilon,minimum=-25,maximum=75,reduced=True)

#compare_inputs(56,epsilon,minimum=-2.1,maximum=0.1,reduced=False)
compare_inputs(56,epsilon,minimum=-1.1,maximum=0.1,reduced=True)


# trackSumJetDeltaR
#compare_inputs(57,epsilon,minimum=None,maximum=0.3,reduced=False)
compare_inputs(57,epsilon,minimum=None,maximum=0.3,reduced=True)


# trackSumJetEtRatio
#compare_inputs(58,epsilon,minimum=None,maximum=2.1,reduced=False)
compare_inputs(58,epsilon,minimum=None,maximum=2.1,reduced=True)


# vertexCat
#compare_inputs(59,epsilon,minimum=-0.1,maximum=2.1,reduced=False)
compare_inputs(59,epsilon,minimum=-0.1,maximum=2.1,reduced=True)


# vertexEnergyRatio
#compare_inputs(60,epsilon,minimum=None,maximum=2.2,reduced=False)
compare_inputs(60,epsilon,minimum=None,maximum=2.2,reduced=True)


# vertexJetDeltaR
# ok
#compare_inputs(61,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(61,epsilon,minimum=None,maximum=None,reduced=True)


# vertexMass
#compare_inputs(62,epsilon,minimum=None,maximum=75,reduced=False)
compare_inputs(62,epsilon,minimum=None,maximum=75,reduced=True)


# jetNSecondaryVertices
# ok
#compare_inputs(63,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(63,epsilon,minimum=None,maximum=None,reduced=True)


# jetNSelectedTracks
#compare_inputs(64,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(64,epsilon,minimum=None,maximum=None,reduced=True)


# jetNTracksEtaRel
#compare_inputs(65,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(65,epsilon,minimum=None,maximum=None,reduced=True)


# vertexNTracks
#compare_inputs(66,epsilon,minimum=None,maximum=None,reduced=False)
compare_inputs(66,epsilon,minimum=None,maximum=None,reduced=True)
'''

