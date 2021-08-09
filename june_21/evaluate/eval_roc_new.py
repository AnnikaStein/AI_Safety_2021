import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn
#from torch.utils.data import TensorDataset, ConcatDataset

from sklearn import metrics

import gc

import coffea.hist as hist

import time

import argparse

import sys

sys.path.append("/home/um106329/aisafety/june_21/attack/")
from disturb_inputs import fgsm_attack
from disturb_inputs import apply_noise
sys.path.append("/home/um106329/aisafety/june_21/train_models/")
from focal_loss import FocalLoss, focal_loss
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])


parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("comparesetup", help="Setup for comparison, examples: BvL_raw, CvB_sigma0.01, bb_eps0.01, can be a comma-separated list")
parser.add_argument("plotdeepcsv", help="Plot DeepCSV for comparison")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
#parser.add_argument("dominimal", help="Only do training with minimal setup, i.e. 15 QCD, 5 TT files")
parser.add_argument("dominimal_eval", help="Only minimal number of files for evaluation")
#parser.add_argument("compare", help="Compare with earlier epochs", default='no')  # one can infer if user wants to compare epochs, if user put in more than one epoch
#parser.add_argument("dofl", help="Use Focal Loss")
parser.add_argument("add_axis", help="Add a second axis as inset to the plot")
parser.add_argument("FGSM_setup", type=int, help="If FGSM, create adversarial inputs from individual models (-1) or only for a given setup (specify this setup, e.g. type 2 for the 0,1,2 --> second index of all requested methods (counted from 0).)")
parser.add_argument("log_axis", help="Flip axis and use log scale (yes/no)")
parser.add_argument("save_mode", help="Save AUC only, or save plots of ROC curves (ROC/AUC), only useful for multiple specified setups (compare).")
args = parser.parse_args()

NUM_DATASETS = args.files
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
compare_setup = args.comparesetup
setups = [s for s in compare_setup.split(',')]
plot_deepcsv = True if args.plotdeepcsv == 'yes' else False
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',')]
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
    
n_samples = args.jets
#do_minimal = args.dominimal
do_minimal_eval = args.dominimal_eval
compare = True if (len(epochs) > 1 or len(setups) > 1 or len(wmets) > 1) else False
#do_FL = args.dofl

#if do_FL == 'yes':
#    fl_text = '_focalloss'
#else:
#    fl_text = ''
add_axis = True if args.add_axis == 'yes' else False
FGSM_setup = args.FGSM_setup
log_axis = True if args.log_axis == 'yes' else False
save_mode = args.save_mode

if save_mode=='AUC':
    epochs = range(epochs[0],epochs[1]+1)
    
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

#gamma = (weighting_method.split('_gamma')[-1]).split('_alpha')[0]
#alphaparse = (weighting_method.split('_gamma')[-1]).split('_alpha')[-1]
#if gamma[0] != '_': print('gamma',gamma)
#if alphaparse[0] != '_': print('alpha',alphaparse)

#wm_def_text = {'_noweighting': 'No weighting', 
#               '_ptetaflavloss' : r'$p_T, \eta$ rew.',
#               '_flatptetaflavloss' : r'$p_T, \eta$ rew. (Flat)',
#               '_ptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (Focal Loss)', 
#               '_flatptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (Flat, Focal Loss)',
#               f'_ptetaflavloss_focalloss_gamma{gamma}' : r'$p_T, \eta$ rew. (Focal Loss $\gamma=$'+f'{gamma})', 
#               f'_ptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : r'$p_T, \eta$ rew. (Focal Loss $\gamma=$'+f'{gamma}'+r',$\alpha=$'+f'{alphaparse})', 
#               f'_ptetaflavloss_focalloss_alpha{alphaparse}' : r'$p_T, \eta$ rew. (Focal Loss $\gamma=$'+f'2.0'+r',$\alpha=$'+f'{alphaparse})', 
#               f'_flatptetaflavloss_focalloss_gamma{gamma}' : r'$p_T, \eta$ rew. (Flat, focal Loss $\gamma=$'+f'{gamma})', 
#               f'_flatptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : r'$p_T, \eta$ rew. (Flat, Focal Loss $\gamma=$'+f'{gamma}'+r',$\alpha=$'+f'{alphaparse})', 
#              }
#wm_def_color = {'_noweighting': 'yellow', 
#               '_ptetaflavloss' : 'orange',
#               '_flatptetaflavloss' : 'brown',
#               '_ptetaflavloss_focalloss' : 'cyan', 
#               '_flatptetaflavloss_focalloss' : 'blue',
#               f'_ptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : '#FEC55C', 
#              }

if do_minimal_eval == 'no':
    test_input_file_paths = [f'/hpcwork/um106329/june_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/june_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    test_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    DeepCSV_testset_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    
        
if do_minimal_eval == 'yes':
    test_input_file_paths = [f'/hpcwork/um106329/june_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/june_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,5)]
    test_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,5)]
    DeepCSV_testset_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,5)]
    
        
if do_minimal_eval == 'medium':
    rng = np.random.default_rng(12345)
    some_files = rng.integers(low=0, high=278, size=50)
    test_input_file_paths = np.array([f'/hpcwork/um106329/june_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/june_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
    test_target_file_paths = np.array([f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
    DeepCSV_testset_file_paths = np.array([f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
    
'''

    Load inputs and targets
    
'''
#test_input_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths))
print('test inputs done')

#test_target_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]
#DeepCSV_testset_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]

        
#test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths))
print('test targets done')

# currently, all test inputs (from 278 files!) at once is too much for memory in the later steps --> use every third jet
if do_minimal_eval == 'no':
    test_inputs = test_inputs[::3]
    test_targets = test_targets[::3]
    
len_test = len(test_targets)
print('number of test inputs', len_test)

jetFlavour = test_targets+1


#do_noweighting   = True if ( weighting_method == '_all' or weighting_method == '_noweighting'   ) else False
#do_ptetaflavloss = True if ( weighting_method == '_all' or weighting_method == '_ptetaflavloss' ) else False
#do_flatptetaflavloss = True if ( weighting_method == '_all' or weighting_method == '_flatptetaflavloss' ) else False


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
    

if compare == False:
    with torch.no_grad():
        DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
        print('DeepCSV test done')
        
        checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        predictions = model(test_inputs.float()).detach().numpy()
        # to not divide by zero when calculating the discriminators
        #predictions[:,0][predictions[:,0] > 0.999999] = 0.999999
        #predictions[:,1][predictions[:,1] > 0.999999] = 0.999999
        #predictions[:,2][predictions[:,2] > 0.999999] = 0.999999
        #predictions[:,3][predictions[:,3] > 0.999999] = 0.999999
        #predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
        #predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
        #predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
        #predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
        # but I'll rather select only those predictions manually where the division by zero does not occur, for each discriminator separately (all others would be -1)
        
        
        wm_text = wm_def_text[weighting_method]
        #'''
        fig = plt.figure(figsize=[12,12])       
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==0, torch.ones(len_test), torch.zeros(len_test)),predictions[:,0])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for b-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==0, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,0])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for b-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}', f'DeepCSV, AUC = {deepcsvauc:.4f}'],title='ROC b tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC b tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        fig = plt.figure(figsize=[12,12])
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==1, torch.ones(len_test), torch.zeros(len_test)),predictions[:,1])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for bb-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==1, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,1])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for bb-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}', f'DeepCSV, AUC = {deepcsvauc:.4f}'],title='ROC bb tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC bb tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_bb_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_bb_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fig = plt.figure(figsize=[12,12])
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==2, torch.ones(len_test), torch.zeros(len_test)),predictions[:,2])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for c-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==2, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,2])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for c-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}', f'DeepCSV, AUC = {deepcsvauc:.4f}'],title='ROC c tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC c tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        #fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len_test)],predictions_new_flat[:,3])
        # trying out new way to slice targets instead of looping over them
        #torch.where(m<0,m,n)
        
        fig = plt.figure(figsize=[12,12])
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==3, torch.ones(len_test), torch.zeros(len_test)),predictions[:,3])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for udsg-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==3, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,3])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for udsg-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}', f'DeepCSV, AUC = {deepcsvauc:.4f}'],title='ROC udsg tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC udsg tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #'''


        '''
            B vs Light jets
        '''
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del predictions_new_flat
        
        matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del DeepCSV_testset
        #gc.collect()
        
        
        # now select only those that won't lead to division by zero
        # just to be safe: select only those values where the range is 0-1 (here for Prob(b) and Prob(bb), so far it looks like the -1 default is always present for all outputs simultaneously, but you never know...)
        # because we slice based on the outputs, and have to apply the slicing in exactly the same way for targets and outputs, the targets need to go first (slicing with the 'old' outputs), then slice the outputs
        matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
        matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
        matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0]
        matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0]
        

        #len_BvsUDSG = len(matching_targets)
        #len_BvLDeepCSV = len(matching_DeepCSV_targets)

        # just some checks to see what the ranges are
        #print(min(matching_predictions[:,0]),min(matching_predictions[:,1]),min(matching_predictions[:,2]),min(matching_predictions[:,3]))
        #print(min(matching_DeepCSV[:,0]),min(matching_DeepCSV[:,1]),min(matching_DeepCSV[:,2]),min(matching_DeepCSV[:,3]))
        
        #print(max(matching_predictions[:,0]),max(matching_predictions[:,1]),max(matching_predictions[:,2]),max(matching_predictions[:,3]))
        #print(max(matching_DeepCSV[:,0]),max(matching_DeepCSV[:,1]),max(matching_DeepCSV[:,2]),max(matching_DeepCSV[:,3]))
        
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve((matching_targets==0) | (matching_targets==1),(matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for B vs UDSG {wm_text}: {customauc}")
        fpr_DeepCSV,tpr_DeepCSV,thresholds_DeepCSV = metrics.roc_curve((matching_DeepCSV_targets==0) | (matching_DeepCSV_targets==1),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]))
        plt.plot(tpr_DeepCSV,fpr_DeepCSV)
        deepcsvauc = metrics.auc(fpr_DeepCSV,tpr_DeepCSV)
        print(f"auc for B vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],title='ROC B vs L',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        # more checks
        #print(min(thresholds_custom))
        #print(min(thresholds_DeepCSV))
        
        #print(max(thresholds_custom))
        #print(max(thresholds_DeepCSV))
        
        #print(np.unique(thresholds_custom)[-2])
        #print(np.unique(thresholds_DeepCSV)[-2])
        
        #print((matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,2]))
        #print(tpr_custom)
        #print(fpr_custom)
        
        fig = plt.figure(figsize=[12,12],num=40)
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,2]))
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(1-matching_DeepCSV[:,2]))
        plt.plot(thresholds_DeepCSV,tpr_DeepCSV,c='orange')
        plt.plot(thresholds_DeepCSV,fpr_DeepCSV,linestyle='dashed',c='orange')
        #deepcsvauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('TPR/FPR B vs L')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}',f'TPR DeepCSV', f'FPR DeepCSV'],title='B vs L',loc='center',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #sys.exit()
        '''
            B vs C jets
        '''
        #matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del gc.garbage[:]
        #del test_inputs
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del predictions_new_flat
        
        matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del DeepCSV_testset
        #gc.collect()
        
        
        # now select only those that won't lead to division by zero
        #matching_inputs = matching_inputs[(1-matching_predictions[:,3]) != 0]
        matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
        matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
        matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0]
        matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0]

        #len_BvsC = len(matching_targets)
        #len_BvCDeepCSV = len(matching_DeepCSV_targets)
        
        

        
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve((matching_targets==0) | (matching_targets==1),(matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for B vs C {wm_text}: {customauc}")
        fpr_DeepCSV,tpr_DeepCSV,thresholds_DeepCSV = metrics.roc_curve((matching_DeepCSV_targets==0) | (matching_DeepCSV_targets==1),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]))
        plt.plot(tpr_DeepCSV,fpr_DeepCSV)
        deepcsvauc = metrics.auc(fpr_DeepCSV,tpr_DeepCSV)
        print(f"auc for B vs C DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for b vs. c\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],title='ROC B vs C',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)    
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')           
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
        fig = plt.figure(figsize=[12,12],num=40)
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,2]))
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(1-matching_DeepCSV[:,2]))
        plt.plot(thresholds_DeepCSV,tpr_DeepCSV,c='orange')
        plt.plot(thresholds_DeepCSV,fpr_DeepCSV,linestyle='dashed',c='orange')
        #deepcsvauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('TPR/FPR B vs C')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}',f'TPR DeepCSV', f'FPR DeepCSV'],title='B vs C',loc='lower left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvC_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvC_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #sys.exit()
        '''
            C vs B jets
        '''
        # inputs and targets are the same as for the previous classifier / discriminator
       
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve(matching_targets==2,(matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for C vs B {wm_text}: {customauc}")
        fpr_DeepCSV,tpr_DeepCSV,thresholds_DeepCSV = metrics.roc_curve(matching_DeepCSV_targets==2,(matching_DeepCSV[:,2])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]))
        plt.plot(tpr_DeepCSV,fpr_DeepCSV)
        deepcsvauc = metrics.auc(fpr_DeepCSV,tpr_DeepCSV)
        print(f"auc for C vs B DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for c vs. b\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],title='ROC C vs B',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400) 
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight') 
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)         
        
        fig = plt.figure(figsize=[12,12],num=40)
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,2]))
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(1-matching_DeepCSV[:,2]))
        plt.plot(thresholds_DeepCSV,tpr_DeepCSV,c='orange')
        plt.plot(thresholds_DeepCSV,fpr_DeepCSV,linestyle='dashed',c='orange')
        #deepcsvauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('TPR/FPR C vs B')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}',f'TPR DeepCSV', f'FPR DeepCSV'],title='C vs B',loc='lower left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvB_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvB_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #sys.exit() 
        '''
            C vs Light jets
        '''
        matching_targets = test_targets[(jetFlavour==3) | (jetFlavour==4)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==3) | (jetFlavour==4)]
        #del predictions_new_flat
        
        matching_DeepCSV = DeepCSV_testset[(jetFlavour==3) | (jetFlavour==4)]
        #del DeepCSV_testset
        #gc.collect()
        
        
        matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,2]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
        matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,2]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
        matching_targets = matching_targets[(matching_predictions[:,2]+matching_predictions[:,3]) != 0]
        matching_predictions = matching_predictions[(matching_predictions[:,2]+matching_predictions[:,3]) != 0]

        #len_CvsUDSG = len(matching_targets)


        
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve(matching_targets==2,(matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for C vs UDSG {wm_text}: {customauc}")
        fpr_DeepCSV,tpr_DeepCSV,thresholds_DeepCSV = metrics.roc_curve(matching_DeepCSV_targets==2,(matching_DeepCSV[:,2])/(matching_DeepCSV[:,2]+matching_DeepCSV[:,3]))
        plt.plot(tpr_DeepCSV,fpr_DeepCSV)
        deepcsvauc = metrics.auc(fpr_DeepCSV,tpr_DeepCSV)
        print(f"auc for C vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for c vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],title='ROC C vs L',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
        fig = plt.figure(figsize=[12,12],num=40)
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,2]))
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        #fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(1-matching_DeepCSV[:,2]))
        plt.plot(thresholds_DeepCSV,tpr_DeepCSV,c='orange')
        plt.plot(thresholds_DeepCSV,fpr_DeepCSV,linestyle='dashed',c='orange')
        #deepcsvauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('TPR/FPR C vs L')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}',f'TPR DeepCSV', f'FPR DeepCSV'],title='C vs L',loc='upper right',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        sys.exit()
            
else:
    # ================================================================================
    #
    #
    #              Compare different things like epochs, methods, parameters
    #
    # ................................................................................
    #
    #                              Check what is requested
    #
    # ................................................................................
    # construct a combination of properties for every ROC
    # if only one thing varies, adjust the length of the other properties to match the max.
    # epochs and wmets defines the file that has to be read
    # setups controls how the inputs are constructed
    n_compare = max(len(epochs),len(setups),len(wmets))
    if len(epochs) == 1: 
        same_epoch = True
        epochs = n_compare * epochs
    else:
        same_epoch = False

    if len(setups) == 1:
        same_setup = True
        setups = n_compare * setups
    else:
        same_setup = False

    if len(wmets) == 1: 
        same_wm = True
        wmets = n_compare * wmets
    else:
        same_wm = False

    # get the output variable or discriminator that will be compared
    outdisc = setups[0].split('_')[0]
    print(outdisc)

    # get linestyle / colour depending on what is requested
    # all raw, compare different epochs, all raw compare different weighting methods --> just different colours, linestyle identical
    # raw and distorted for different epochs, but same parameter --> raw - / distorted --
    # basic and adversarial training --> basic - / adversarial --
    # raw and distorted for same epoch, but different parameters --> raw - / distorted -, just different colours
    # deepcsv (if requested) --> black -

    # evaluate the distorted inputs at the end, ckeck if it's always the same parameter / method to decide how the plots will look like (see different styles above)
    non_raw_setups_sigma = []
    non_raw_setups_epsilon = []
    for s in setups:
        if 'sigma' in s:
            non_raw_setups_sigma.append(s.split('sigma')[-1])
        elif 'eps' in s:
            non_raw_setups_epsilon.append(s.split('eps')[-1])

    if len(non_raw_setups_sigma) == 0 and len(non_raw_setups_epsilon) == 0:
        # everything is raw only, so the order does not matter --> always use undisturbed inputs
        raw_only = True
    elif len(non_raw_setups_sigma) == 0 and len(non_raw_setups_epsilon) != 0:
        # FGSM, no Noise
        raw_only = False
        if non_raw_setups_epsilon.count(non_raw_setups_epsilon[0]) == len(non_raw_setups_epsilon):
            always_same_parameter = True
            that_one_epsilon = non_raw_setups_epsilon[0]
    elif len(non_raw_setups_sigma) != 0 and len(non_raw_setups_epsilon) == 0:
        # Noise, no FGSM
        raw_only = False
        if non_raw_setups_sigma.count(non_raw_setups_sigma[0]) == len(non_raw_setups_sigma):
            always_same_parameter = True
            that_one_sigma = non_raw_setups_sigma[0]
    else:
        # compare Noise, FGSM in same script
        raw_only = False

    has_basic = False
    has_adv = False
    for w in wmets:
        if 'adv' in w:
            has_adv = True
        else:
            has_basic = True
    has_nonflat = False
    has_flat = False
    for w in wmets:
        if 'flat' in w:
            has_flat = True
        if '_ptetaflavloss' in w:
            has_nonflat = True

    new_line_before_AUC = True

    possible_colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


    if raw_only:  # no attacks for comparison
        if same_wm:
            # just go through standard colours, just different epochs, no attacks, all solid lines
            used_colours = possible_colours[:n_compare]
            legend_setup = 'epochs'
            linestyles = ['-' for l in range(n_compare)]
            addition_leg_text = '\n'+wm_def_text[weighting_method]
            individual_legend = [f'Epoch {e}' for e in epochs]
            print('All raw, same weighting method, compare different epochs.')
            #print(linestyles)
            #print(addition_leg_text)
            #print(individual_legend)
            #print(used_colours)
            #sys.exit()
            new_line_before_AUC = False

        else:
            if has_adv != has_basic:  # XOR = means only one type of training shall be used (either basic or adversarial, not both)
                # everything solid lines only
                used_colours = possible_colours[:n_compare]
                linestyles = ['-' for l in range(n_compare)]
                print('All raw, but different weighting methods.')
                if has_flat and has_nonflat:
                    linestyles = []
                    used_colours = []
                    colourpointer = 0
                    for i,w in enumerate(wmets):
                        if (i>0) and (i%2 == 0):
                            colourpointer += 1
                        used_colours.append(possible_colours[colourpointer])
                        if 'flat' in w:
                            linestyles.append('--')
                        else:
                            linestyles.append('-')
                new_line_before_AUC = False
            else:
                # basic and adversarial training present in selection
                # adversarial shall get dashed lines, but colour shall correspond to basic training colour
                # assuming even number of weighting methods, basic and adversarial always consecutive (and corresponding)
                if n_compare%2 != 0:
                    print('Check number of weighting methods, need basic / adversarial consecutively')
                    sys.exit()
                print('All raw, but compare basic with adversarial training.')
                linestyles = []
                used_colours = []
                colourpointer = 0
                for i,w in enumerate(wmets):
                    if (i>0) and (i%2 == 0):
                        colourpointer += 1
                    used_colours.append(possible_colours[colourpointer])
                    if 'adv' in w:
                        linestyles.append('--')
                    else:
                        linestyles.append('-')
                #print(linestyles)
                #print(used_colours)

            if same_epoch:
                legend_setup = 'wmets'
                addition_leg_text = '\n'+f'Epoch {at_epoch}'
                individual_legend = [wm_def_text[w] for w in wmets]
            else:
                legend_setup = 'wmets_epochs'
                addition_leg_text = ''
                individual_legend = [wm_def_text[w]+f'\nEpoch {e}' for w,e in zip(wmets,epochs)]
            #print(addition_leg_text)
            #print(individual_legend)
            #sys.exit()
    else:
        #individual_legend = ['ToDo' for l in range(n_compare)]
        #used_colours = possible_colours[:n_compare]
        #addition_leg_text = 'ToDo'
        #linestyles = ['-' for l in range(n_compare)]
        # ToDo
        linestyles = []
        used_colours = []
        individual_legend = []
        colourpointer = 0
        n_raw = 0
        for i,s in enumerate(setups):
            if (i>0) and (i%2 == 0):
                colourpointer += 1
            used_colours.append(possible_colours[colourpointer])
            if ('sigma' in s) or ('eps' in s):
                linestyles.append('--')
                if 'sigma' in s:
                    sig = s.split('sigma')[-1]
                    individual_legend.append(f'Noise $\sigma=${sig}')

                else:
                    eps = s.split('eps')[-1]
                    individual_legend.append(f'FGSM $\epsilon=${eps}')

            else:
                linestyles.append('-')
                individual_legend.append('Raw')
                n_raw += 1
        addition_leg_text = ''
        if same_epoch:
            addition_leg_text = addition_leg_text + f', Epoch {at_epoch}'
        else:
            for e in range(n_compare):
                individual_legend[e] = f'Epoch {epochs[e]} ' + individual_legend[e]
        if same_wm:
            addition_leg_text = addition_leg_text + '\n'+wm_def_text[weighting_method]
            if n_raw == 1:
                used_colours = possible_colours
        else:
            for w in range(n_compare):
                individual_legend[w] = individual_legend[w] + f'\n{wm_def_text[wmets[w]]}' 


    if new_line_before_AUC:
        for n in range(n_compare):
            individual_legend[n] = individual_legend[n]+',\n'

    else:
        for n in range(n_compare):
            individual_legend[n] = individual_legend[n]+', '


    #print(setups)
    #print(used_colours)
    #print(individual_legend)
    #sys.exit()
    # ................................................................................
    #
    #                              Prepare inputs, targets
    #
    # ................................................................................

    if plot_deepcsv:
        DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
        print('DeepCSV test done')

    # prepare inputs to be able to calculate discriminators
    if outdisc in ['b','bb','c','udsg']:
        # simply test inputs / targets
        matching_inputs = test_inputs
        del test_inputs
        gc.collect()
        matching_targets = test_targets
        del test_targets
        gc.collect()

    elif outdisc == 'BvL':
        # create BvL inputs / targets (if requested, also for DeepCSV)
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        del test_inputs
        gc.collect()

        if plot_deepcsv:
            matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
            del DeepCSV_testset
            # it might be that DeepCSV has some default outputs (-1 bin), plot them at the lower end of the histograms
            matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            gc.collect()


    elif outdisc == 'BvC':
        # create BvC inputs / targets (if requested, also for DeepCSV)
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        del test_inputs
        gc.collect()

        if plot_deepcsv:
            matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
            del DeepCSV_testset
            matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            gc.collect()


    elif outdisc == 'CvB':
        # create CvB inputs / targets (if requested, also for DeepCSV)
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        del test_inputs
        gc.collect()

        if plot_deepcsv:
            matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
            del DeepCSV_testset
            matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            gc.collect()


    elif outdisc == 'CvL':
        # create CvL inputs / targets (if requested, also for DeepCSV)
        matching_targets = test_targets[(jetFlavour==3) | (jetFlavour==4)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==3) | (jetFlavour==4)]
        del test_inputs
        gc.collect()

        if plot_deepcsv:
            matching_DeepCSV = DeepCSV_testset[(jetFlavour==3) | (jetFlavour==4)]
            del DeepCSV_testset
            matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,2]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            matching_DeepCSV = matching_DeepCSV[((matching_DeepCSV[:,2]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
            gc.collect()

    del jetFlavour
    gc.collect()
    print('Prepared matching inputs and targets for this output/discriminator.')



    # .........................................................................................................
    #                                        Begin figure and style
    # .........................................................................................................
    #fig = plt.figure(figsize=[12,12])
    fig,ax = plt.subplots(figsize=[12,12])
    ax.set_xlim(left=-0.05,right=1.05)

    if log_axis:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3)
        ax.set_ylim(top=1)

    if outdisc=='b':
        tag_name = ' (b)'
        mistag_name = ' (bb,c,udsg)'
    elif outdisc=='bb':
        tag_name = ' (bb)'
        mistag_name = ' (b,c,udsg)'
    elif outdisc=='c':
        tag_name = ' (c)'
        mistag_name = ' (b,bb,udsg)'
    elif outdisc=='udsg':
        tag_name = ' (udsg)'
        mistag_name = ' (b,bb,c)'
    else:
        if outdisc[0]=='B':
            tag_name = ' (b+bb)'
        elif outdisc[0]=='C':
            tag_name = ' (c)'
        if outdisc[-1]=='B':
            mistag_name = ' (b+bb)'
        elif outdisc[-1]=='C':
            mistag_name = ' (c)'
        elif outdisc[-1]=='L':
            mistag_name = ' (udsg)'
    if log_axis:
        ax.set_ylabel('Mistagging rate'+mistag_name)
        ax.set_xlabel('Tagging efficiency'+tag_name)
    else:
        ax.set_xlabel('Mistagging rate'+mistag_name)
        ax.set_ylabel('Tagging efficiency'+tag_name)
    if add_axis:
        if log_axis:
            if outdisc=='b':
                ax.plot([0.47,0.57,0.57,0.47,0.47],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([1,0.57],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.565,0.47],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.565, .15, .3, .15])
                ax2.set_xlim(0.47,0.57)
            elif outdisc=='bb':
                ax.plot([0.45,0.55,0.55,0.45,0.45],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([1,0.55],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.565,0.45],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.565, .15, .3, .15])
                ax2.set_xlim(0.45,0.55)
            elif outdisc=='c':
                ax.plot([0.06,0.11,0.11,0.06,0.06],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.55,0.06],[3e-1,2e-2],'--',color='grey')
                ax.plot([0.975,0.11],[8e-2,8e-3],'--',color='grey')
                ax2 = plt.axes([.55, .6, .3, .15])
                ax2.set_xlim(0.06,0.11)
            elif outdisc=='udsg':
                ax.plot([0.04,0.09,0.09,0.04,0.04],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.04],[8.4e-1,2e-2],'--',color='grey')
                ax.plot([0.512,0.09],[2.2e-1,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .71, .3, .15])
                ax2.set_xlim(0.04,0.09)
            elif outdisc=='BvL':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='BvC':
                ax.plot([0.35,0.45,0.45,0.35,0.35],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([1,0.45],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.565,0.35],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.565, .15, .3, .15])
                ax2.set_xlim(0.35,0.45)
            elif outdisc=='CvB':
                ax.plot([0.05,0.15,0.15,0.05,0.05],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.05],[8.4e-1,2e-2],'--',color='grey')
                ax.plot([0.512,0.15],[2.2e-1,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .71, .3, .15])
                ax2.set_xlim(0.05,0.15)
            elif outdisc=='CvL':
                ax.plot([0.2,0.3,0.3,0.2,0.2],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.2],[2.2e-1,8e-3],'--',color='grey')
                ax.plot([0.48,0.3],[2.2e-1,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .71, .28, .15])
                ax2.set_xlim(0.2,0.3)   
            ax2.set_yscale('log')
            ax2.set_ylim(8e-3,2e-2)
        else:  # ToDo! But: not urgent
            if outdisc=='b':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='bb':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='c':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='udsg':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='BvL':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='BvC':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='CvB':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='CvL':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            ax2.set_ylim(-0.05,1.05)


    # --------------------------------------------------------------------------------
    #
    #                        Run over all requested combinations
    #
    # ................................................................................
    if save_mode == 'AUC':
        auc_list =  []
    for i in range(n_compare):
        # https://pytorch.org/docs/stable/generated/torch.autograd.set_grad_enabled.html#torch.autograd.set_grad_enabled --> activate gradients based on a condition (avoiding exec statements --> make code cleaner without duplication)
        activate_grad = True if 'eps' in setups[i] else False
        with torch.set_grad_enabled(activate_grad):  # if FGSM attack is used --> initially, require grad, only disable for final step in external function
            # load correct checkpoint
            if (i == 0) or (same_epoch == False) or (same_wm == False):
                checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{wmets[i]}_{NUM_DATASETS}_{default}_{n_samples}/model_{epochs[i]}_epochs_v10_GPU_weighted{wmets[i]}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
                model.load_state_dict(checkpoint["model_state_dict"])
                if ('_ptetaflavloss' in wmets[i]) or ('_flatptetaflavloss' in wmets[i]):
                    if 'focalloss' not in wmets[i]:
                        criterion = nn.CrossEntropyLoss(reduction='none')
                    elif 'focalloss' in wmets[i]:
                        if 'alpha' not in wmets[i]:
                            alpha = None
                        else:
                            commasep_alpha = [a for a in ((wmets[i].split('_alpha')[-1]).split('_adv_tr_eps')[0]).split(',')]
                            alpha = torch.Tensor([float(commasep_alpha[0]),float(commasep_alpha[1]),float(commasep_alpha[2]),float(commasep_alpha[3])]).to(device)
                        if 'gamma' not in wmets[i]:
                            gamma = 2.0
                        else:
                            gamma = float(((wmets[i].split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0])
                        criterion = FocalLoss(alpha, gamma, reduction='none')

                else:
                    criterion = nn.CrossEntropyLoss()

            # .........................................................................................................
            #                                             predict
            # .........................................................................................................
            
            fgsm_setup_text = ''
            # use raw or distorted
            if 'raw' in setups[i]:
                matching_predictions = model(matching_inputs.float()).detach().numpy()
                #if ('adv' in wmets[i]) and has_basic:
                #    this_line = '--'
                #else:
                #    this_line = '-'
                #setup_text = ''

            elif 'sigma' in setups[i]:
                sig = float(setups[i].split('sigma')[-1])
                matching_predictions = model(apply_noise(matching_inputs.float(),sig)).detach().numpy()
                #this_line = '--'
                #setup_text = f'Noise $\sig={sig}$'   

            elif 'eps' in setups[i]:
                eps = float(setups[i].split('eps')[-1])
                #matching_inputs.requires_grad = True
                if FGSM_setup==-1:
                    matching_predictions = model(fgsm_attack(eps,matching_inputs.float(),matching_targets,model,criterion,dev=device)).detach().numpy()
                else:
                    fgsm_model = nn.Sequential(nn.Linear(67, 100),
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
    
                    fgsm_model.to(device)
                    fgsm_model.eval()
                    fgsm_setup_text = f'* FGSM attack based on epoch {epochs[FGSM_setup]} with\n{wm_def_text[wmets[FGSM_setup]]}'
                    fgsm_checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{wmets[FGSM_setup]}_{NUM_DATASETS}_{default}_{n_samples}/model_{epochs[FGSM_setup]}_epochs_v10_GPU_weighted{wmets[FGSM_setup]}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
                    fgsm_model.load_state_dict(fgsm_checkpoint["model_state_dict"])
                    matching_predictions = model(fgsm_attack(eps,matching_inputs.float(),matching_targets,fgsm_model,criterion,dev=device)).detach().numpy()
                #this_line = '--'      
                #setup_text = f'FGSM $\epsilon={eps}$'   
            #matching_predictions = np.float32(matching_predictions)                                 
            print('Predictions done.')

        wm_text = wm_def_text[wmets[i]]

        #this_label = wm_text + '\n' + setup_text

        this_line = linestyles[i]
        this_colour = used_colours[i]
        this_legtext = individual_legend[i]

        # .........................................................................................................
        #                                             ROC & AUC
        # .........................................................................................................
        if outdisc == 'b':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==0, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,0])
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            print(f"auc for b-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            legtitle = 'ROC b tagging'+addition_leg_text
        elif outdisc == 'bb':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==1, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,1])
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            print(f"auc for bb-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            legtitle = 'ROC bb tagging'+addition_leg_text
        elif outdisc == 'c':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==2, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,2])
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            print(f"auc for c-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            legtitle = 'ROC c tagging'+addition_leg_text
        elif outdisc == 'udsg':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==3, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,3])
            customauc = metrics.auc(fpr,tpr)
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            print(f"auc for udsg-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC udsg tagging'+addition_leg_text
        # every discriminator has different properties / different conditions for the computation to work
        elif outdisc == 'BvL':
            # checking the predictions works only for every iteration specifically (because this depends on the model with which predictions are done)
            # now select only those that won't lead to division by zero
            # just to be safe: select only those values where the range is 0-1 (here for Prob(b) and Prob(bb), so far it looks like the -1 default is always present for all outputs simultaneously, but you never know...)
            # because we slice based on the outputs, and have to apply the slicing in exactly the same way for targets and outputs, the targets need to go first (slicing with the 'old' outputs), then slice the outputs
            actually_matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0]

            fpr,tpr,_ = metrics.roc_curve((actually_matching_targets==0) | (actually_matching_targets==1),(matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]))
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            print(f"auc for bvl-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC B vs L'+addition_leg_text
            del actually_matching_targets
            gc.collect()

        elif outdisc == 'BvC':
            actually_matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0]

            fpr,tpr,_ = metrics.roc_curve((actually_matching_targets==0) | (actually_matching_targets==1),(matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]))
            del matching_predictions
            gc.collect()
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            print(f"auc for bvc-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC B vs C'+addition_leg_text
            del actually_matching_targets
            gc.collect()

        elif outdisc == 'CvB':
            actually_matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0]

            fpr,tpr,_ = metrics.roc_curve(actually_matching_targets==2,(matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]))
            del matching_predictions
            gc.collect()
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            print(f"auc for cvb-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC C vs B'+addition_leg_text
            del actually_matching_targets
            gc.collect()

        elif outdisc == 'CvL':
            actually_matching_targets = matching_targets[(matching_predictions[:,2]+matching_predictions[:,3]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,2]+matching_predictions[:,3]) != 0]

            fpr,tpr,_ = metrics.roc_curve(actually_matching_targets==2,(matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]))
            del matching_predictions
            gc.collect()
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.4f}', linestyle=this_line, color=this_colour)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour)
            print(f"auc for cvl-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC C vs L'+addition_leg_text
            del actually_matching_targets
            gc.collect()
            
    if plot_deepcsv:
        # copy code from above but switch everything to DeepCSV, use black colour
        if outdisc == 'b':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==0, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,0])
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for b-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'bb':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==1, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,1])
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for bb-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'c':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==2, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,2])
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for c-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'udsg':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==3, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,3])
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for udsg-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'BvL':
            fpr,tpr,_ = metrics.roc_curve((matching_DeepCSV_targets==0) | (matching_DeepCSV_targets==1),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]))
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for bvl-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'BvC':
            fpr,tpr,_ = metrics.roc_curve((matching_DeepCSV_targets==0) | (matching_DeepCSV_targets==1),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]))
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for bvc-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'CvB':
            fpr,tpr,_ = metrics.roc_curve(matching_DeepCSV_targets==2,(matching_DeepCSV[:,2])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]))
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for cvb-tagging DeepCSV: {deepcsvauc}")
        if outdisc == 'CvL':
            fpr,tpr,_ = metrics.roc_curve(matching_DeepCSV_targets==2,(matching_DeepCSV[:,2])/(matching_DeepCSV[:,2]+matching_DeepCSV[:,3]))
            deepcsvauc = metrics.auc(fpr,tpr)
            if log_axis:
                ax.plot(tpr,fpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            else:
                ax.plot(fpr,tpr,label=f'DeepCSV, AUC = {deepcsvauc:.4f}',color='k')
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr,color='k')
                else:
                    ax2.plot(fpr,tpr,color='k')
            print(f"auc for cvl-tagging DeepCSV: {deepcsvauc}")
    del fpr
    del tpr
    gc.collect()

    
    if save_mode=='ROC':
        # .........................................................................................................
        #                                             style some more & save
        # .........................................................................................................
        leg = ax.legend(title=legtitle,loc=legloc,fontsize=20,title_fontsize=23)
        if 'right' in legloc:
            aligned = 'right'
        else:
            aligned = 'left'
        leg._legend_box.align = aligned
        ax.grid(which='minor', alpha=0.85)
        ax.grid(which='major', alpha=0.95, color='black')
        if add_axis:
            ax2.grid(which='minor', alpha=0.85)
            ax2.grid(which='major', alpha=0.95, color='black')

        if FGSM_setup != -1:
            if log_axis:
                ax.text(0,1.5e-3,fgsm_setup_text, fontsize=16)
            else:
                ax.text(0,0,fgsm_setup_text, fontsize=16)

        if plot_deepcsv:
            dcsv_text = '_with_deepcsv'
        else:
            dcsv_text = ''
        if log_axis:
            log_axis_text = '_logAx'
        else:
            log_axis_text = ''
        if add_axis:
            add_axis_text = '_addAx'
        else:
            add_axis_text = ''
        weighting_method = list(dict.fromkeys(wmets))
        at_epoch = list(dict.fromkeys(epochs))
        compare_setup = list(dict.fromkeys(setups))
        fgsm_setup_text = '' if fgsm_setup_text=='' else f'_FGSM{FGSM_setup}.'
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/compare/roc_{outdisc}_weighting_method{wmets}_at_epoch_{epochs}_setup_{setups}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/compare_new/roc_{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{dcsv_text}{log_axis_text}{add_axis_text}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{default}_{n_samples}.svg', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/compare_new/roc_{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{dcsv_text}{log_axis_text}{add_axis_text}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{default}_{n_samples}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
    else:
        auc_array = np.array(auc_list)
        weighting_method = list(dict.fromkeys(wmets))
        #at_epoch = list(dict.fromkeys(epochs))
        compare_setup = list(dict.fromkeys(setups))
        fgsm_setup_text = '' if fgsm_setup_text=='' else f'_FGSM{FGSM_setup}.'
        np.save(f'/home/um106329/aisafety/june_21/evaluate/auc/{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{default}_{n_samples}.npy',auc_array)
