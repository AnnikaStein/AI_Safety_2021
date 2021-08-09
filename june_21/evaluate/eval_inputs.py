import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn
#from torch.utils.data import TensorDataset, ConcatDataset

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import entropy

import gc

import coffea.hist as hist

import time

import argparse
import ast

import sys

sys.path.append("/home/um106329/aisafety/june_21/attack/")
from disturb_inputs import fgsm_attack
from disturb_inputs import apply_noise
import definitions
sys.path.append("/home/um106329/aisafety/june_21/train_models/")
from focal_loss import FocalLoss, focal_loss
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])

# for reproducible results
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("variable", type=int, help="Index of input variable")
parser.add_argument("attack", help="The type of the attack, noise or fgsm")
parser.add_argument("fixRange", help="Use predefined range (yes) or just as is (no)")
parser.add_argument("para", help="Parameter for attack or noise (epsilon or sigma), can be comma-separated.")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("dominimal_eval", help="Only minimal number of files for evaluation")
args = parser.parse_args()

variable = args.variable
attack = args.attack
fixRange = args.fixRange
para = args.para
param = [float(p) for p in para.split(',')]
NUM_DATASETS = args.files
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',')]
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
    
n_samples = args.jets
do_minimal_eval = args.dominimal_eval
    
print(f'Evaluate training at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')

gamma = [((weighting_method.split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0] for weighting_method in wmets]
alphaparse = [((weighting_method.split('_gamma')[-1]).split('_alpha')[-1]).split('_adv_tr_eps')[0] for weighting_method in wmets]
epsilon = [(weighting_method.split('_adv_tr_eps')[-1]) for weighting_method in wmets]
print('gamma',gamma)
print('alpha',alphaparse)
print('epsilon',epsilon)

wm_def_text = {'_noweighting': 'No weighting', 
               '_ptetaflavloss' : r'$p_T, \eta$ rew.',
               '_flatptetaflavloss' : r'$p_T, \eta$ rew. (Flat)',
               '_ptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (F.L.)', 
               '_flatptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (Flat, F.L.)', 
              }

more_text = [(f'_ptetaflavloss_focalloss_gamma{g}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g}'+r', $\alpha=$'+f'{a})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_alpha{a}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'2.0'+r', $\alpha=$'+f'{a})') for a in alphaparse] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'{g})') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'2.0)') for g in gamma] + \
            [(f'_ptetaflavloss_focalloss' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'2.0)') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}_alpha{a}' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'{g}'+r', $\alpha=$'+f'{a})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g}, $\epsilon=$'+f'{e})') for g, a, e in zip(gamma,alphaparse,epsilon)] + \
            [(f'_ptetaflavloss_adv_tr_eps{e}' , r'$p_T, \eta$ rew. ($\epsilon=$'+f'{e})') for e in epsilon]

more_text_dict = {k:v for k, v in more_text}
wm_def_text =  {**wm_def_text, **more_text_dict}


scalers = [torch.load(f'/hpcwork/um106329/june_21/scaler_{i}_with_default_{default}.pt') for i in range(67)]

minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')
defaults_per_variable = minima - default

if do_minimal_eval == 'no':
    test_input_file_paths = [f'/hpcwork/um106329/june_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/june_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    test_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    #DeepCSV_testset_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    
        
if do_minimal_eval == 'yes':
    test_input_file_paths = [f'/hpcwork/um106329/june_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/june_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,5)]
    test_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,5)]
    #DeepCSV_testset_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,5)]
    
        
if do_minimal_eval == 'medium':
    rng = np.random.default_rng(12345)
    some_files = rng.integers(low=0, high=278, size=50)
    test_input_file_paths = np.array([f'/hpcwork/um106329/june_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/june_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
    test_target_file_paths = np.array([f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
    #DeepCSV_testset_file_paths = np.array([f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
    
    
    

relative_entropies = []


plt.ioff()
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
    
model.to(device)
model.eval()
    

n_compare = 1
checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{wmets[0]}_{NUM_DATASETS}_{default}_{n_samples}/model_{epochs[0]}_epochs_v10_GPU_weighted{wmets[0]}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])
if ('_ptetaflavloss' in wmets[0]) or ('_flatptetaflavloss' in wmets[0]):
    if 'focalloss' not in wmets[0]:
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif 'focalloss' in wmets[0]:
        if 'alpha' not in wmets[0]:
            alpha = None
        else:
            commasep_alpha = [a for a in ((wmets[0].split('_alpha')[-1]).split('_adv_tr_eps')[0]).split(',')]
            alpha = torch.Tensor([float(commasep_alpha[0]),float(commasep_alpha[1]),float(commasep_alpha[2]),float(commasep_alpha[3])]).to(device)
        if 'gamma' not in wmets[0]:
            gamma = 2.0
        else:
            gamma = float(((wmets[0].split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0])
        criterion = FocalLoss(alpha, gamma, reduction='none')

else:
    criterion = nn.CrossEntropyLoss()
        
print('Loaded model and corresponding criterion.')
        
def plot(variable=0,mode=attack,param=0.1,minim=None,maxim=None,reduced=True):
    xmagn = []
    #for s in range(0, len(test_target_file_paths)):
    for s in range(0, 1):
        #scalers = torch.load(scalers_file_paths[s])
        #scalers = all_scalers[s]
        #test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
        all_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
        #val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
        #train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
        #test_targets =  torch.load(test_target_file_paths[s]).to(device)
        all_targets =  torch.load(test_target_file_paths[s]).to(device)
        #val_targets =  torch.load(val_target_file_paths[s]).to(device)
        #train_targets =  torch.load(train_target_file_paths[s]).to(device)
        #all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))
        #all_targets = torch.cat((test_targets,val_targets,train_targets))
        
        for i, m in enumerate(param):
            if s > 0:
                if mode == 'fgsm':
                    xadv = np.concatenate((xmagn[i], scalers[variable].inverse_transform(fgsm_attack(epsilon=param[i],sample=all_inputs,targets=all_targets,thismodel=model,thiscriterion=criterion,reduced=reduced)[:,variable].cpu())))
                else:
                    xadv = np.concatenate((xmagn[i], scalers[variable].inverse_transform(apply_noise(all_inputs,magn=param[i])[:,variable].cpu())))
                integervars = [59,63,64,65,66]
                if variable in integervars:
                    xadv = np.rint(xadv)
                xmagn[i] = xadv
            else:
                if mode == 'fgsm':
                    xadv = scalers[variable].inverse_transform(fgsm_attack(epsilon=param[i],sample=all_inputs,targets=all_targets,thismodel=model,thiscriterion=criterion,reduced=reduced)[:,variable].cpu())
                else:
                    xadv = scalers[variable].inverse_transform(apply_noise(all_inputs,magn=param[i])[:,variable].cpu())
                integervars = [59,63,64,65,66]
                if variable in integervars:
                    xadv = np.rint(xadv)
                    
                xmagn.append(xadv)
        
        
        del all_inputs
        del all_targets
        gc.collect()
    
     
    minimum = min([min(xmagn[i]) for i in range(len(param))])
    maximum = max([max(xmagn[i]) for i in range(len(param))])
    
    bins = np.linspace(minimum+(maximum-minimum)/100/2,maximum-(maximum-minimum)/100/2,100)
    
    
    compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",definitions.display_names[variable],100,minimum,maximum))
    compHist.fill(sample="raw",prop=xmagn[0])
    
    for si in range(1,len(param)):
        if mode == 'fgsm':
            compHist.fill(sample=f"fgsm $\epsilon$={param[si]}",prop=xmagn[si])
        else:
            compHist.fill(sample=f"noise $\sigma$={param[si]}",prop=xmagn[si])
            
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2.5],'hspace': .25})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.get_legend().remove()
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'])
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'])
        
    running_relative_entropies = []
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        '''
            Kullback-Leibler divergence with raw (denom ratio plot) and disturbed (num ratio plot) data: relative entropy
            
            As explained above
        '''
        num[(num == 0) & (denom != 0)] = 1
        entr = entropy(denom, qk=num)
        #print(f'{variable} ({definitions.input_names[variable]}):\t FGSM $\sigma$={param[si]}\t {entr}')
        running_relative_entropies.append([variable, param[si], entr])
        
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.plot([minimum,maximum],[1,1],color='black')    
        ax2.set_ylim(0,2)
        ax2.set_xlim(minimum,maximum)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw')
        else:
            ax2.set_ylabel('Noise/raw')
            
            
    relative_entropies.append(running_relative_entropies)
    print(relative_entropies)
        
        
    name_var = definitions.input_names[variable]
    range_text = ''
    log_text = ''
    n_samples_text = int(sum(denom))
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.svg', bbox_inches='tight')
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.pdf', bbox_inches='tight')
    
    
    kl = np.array(relative_entropies)
    print(kl)
    np.save(f'inputs/kl_div/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.npy', kl)
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    del fig, ax1, ax2
    gc.collect(2)
    
    
    # !!! logarithmic axis --> separate plot
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2.5],'hspace': .25})
    #ax1 = hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1 = hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.set_yscale('log')
    ax1.set_ylim(None, None)
    ax1.get_legend().remove()
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'])
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'])
        
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.plot([minimum,maximum],[1,1],color='black')    
        ax2.set_ylim(0,2)
        ax2.set_xlim(minimum,maximum)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw')
        else:
            ax2.set_ylabel('Noise/raw')
            
            
    name_var = definitions.input_names[variable]
    range_text = ''
    log_text = '_logAxis'
    n_samples_text = int(sum(denom))
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.svg', bbox_inches='tight')
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.pdf', bbox_inches='tight')
    
    del fig, ax1, ax2
    gc.collect()

    
    # =================================================================================================================
    # 
    #
    #                                                 Fixed range!
    #
    #
    # -----------------------------------------------------------------------------------------------------------------
    
    
    if minim is None:
        minimum = min([min(xmagn[i]) for i in range(len(param))])
    if maxim is None:
        maximum = max([max(xmagn[i]) for i in range(len(param))])
    
    bins = np.linspace(minimum+(maximum-minimum)/100/2,maximum-(maximum-minimum)/100/2,100)
    
    
    compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",definitions.display_names[variable],100,minimum,maximum))
    compHist.fill(sample="raw",prop=xmagn[0])
    
    for si in range(1,len(param)):
        if mode == 'fgsm':
            compHist.fill(sample=f"fgsm $\epsilon$={param[si]}",prop=xmagn[si])
        else:
            compHist.fill(sample=f"noise $\sigma$={param[si]}",prop=xmagn[si])
            
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2.5],'hspace': .25})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.get_legend().remove()
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'])
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'])
        
    running_relative_entropies = []
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        '''
            Kullback-Leibler divergence with raw (denom ratio plot) and disturbed (num ratio plot) data: relative entropy
            
            As explained above
        '''
        num[(num == 0) & (denom != 0)] = 1
        entr = entropy(denom, qk=num)
        #print(f'{variable} ({definitions.input_names[variable]}):\t FGSM $\sigma$={param[si]}\t {entr}')
        running_relative_entropies.append([variable, param[si], entr])
        
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.plot([minimum,maximum],[1,1],color='black')    
        ax2.set_ylim(0,2)
        ax2.set_xlim(minimum,maximum)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw')
        else:
            ax2.set_ylabel('Noise/raw')
            
            
    relative_entropies.append(running_relative_entropies)
    print(relative_entropies)
        
        
    name_var = definitions.input_names[variable]
    range_text = '_specRange'
    log_text = ''
    n_samples_text = int(sum(denom))
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.svg', bbox_inches='tight')
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.pdf', bbox_inches='tight')
    
    
    kl = np.array(relative_entropies)
    print(kl)
    np.save(f'inputs/kl_div/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.npy', kl)
    
        #pass
    del fig, ax1, ax2
    gc.collect(2)
    
    
    # !!! logarithmic axis --> separate plot
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2.5],'hspace': .25})
    ax1 = hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.set_yscale('log')
    ax1.set_ylim(None, None)
    ax1.get_legend().remove()
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'])
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'])
        
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.plot([minimum,maximum],[1,1],color='black')    
        ax2.set_ylim(0,2)
        ax2.set_xlim(minimum,maximum)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw')
        else:
            ax2.set_ylabel('Noise/raw')
            
            
    name_var = definitions.input_names[variable]
    range_text = '_specRange'
    log_text = '_logAxis'
    n_samples_text = int(sum(denom))
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.svg', bbox_inches='tight')
    fig.savefig(f'inputs/input_{variable}_{name_var}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}.pdf', bbox_inches='tight')
    
    del fig, ax1, ax2
    gc.collect()
    
    
    
    
if fixRange == 'yes':
    min_max = definitions.manual_ranges[variable]
else:
    min_max = [None,None]
    
plot(variable,mode=attack,param=[0]+param,minim=min_max[0],maxim=min_max[1],reduced=True)

