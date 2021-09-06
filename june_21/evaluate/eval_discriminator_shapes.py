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

from scipy.stats import ks_2samp
from scipy.stats import entropy

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])

plt.ioff()

#oversampling = False  # deprecated, as WeightedRandomSampler will not be used

parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
#parser.add_argument("dominimal", help="Training done with minimal setup, i.e. 15 QCD, 5 TT files")
parser.add_argument("dominimal_eval", help="Only minimal number of files for evaluation (yes/no)")
#parser.add_argument("dofl", help="Use Focal Loss")  # pack focal loss into name of the weighting method
parser.add_argument("check_inputs", help="Check certain inputs in slices of Prob(udsg) (yes/no)")
args = parser.parse_args()

# weighting method can now be the simple names of the weighting method or the name with an additional _focalloss

# logic: only one epoch specified:
#    can test one weighting method alone (split by flavour), or multiple weighting methods (then the individual outputs will be stacked and compared with DeepCSV)
# more than epoch specified:
#    can test only for one weighting method, but get KS test between the first specified epoch and all specified following epochs per flavour and output node of the tagger plus comparison for the stacked histograms

NUM_DATASETS = args.files
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',_')]
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)

n_samples = args.jets
#do_minimal = args.dominimal
do_minimal_eval = args.dominimal_eval
compare_eps = True if len(epochs) > 1 else False
compare_wmets = True if len(wmets) > 1 else False
#do_FL = args.dofl
check_inputs = args.check_inputs

#if do_FL == 'yes':
#    fl_text = '_focalloss'
#else:
#    fl_text = ''

#if compare_eps:
#    # produce split by flav. histograms with one w.m.
# if compare_eps and compare_wmets:
#    print('Comparing epochs and weighting methods not supported. Choose one.')
#    sys.exit(0)
#if compare_wmets:
#    # Stack flavour per weighting method together

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
               '_ptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (F.L. $\gamma=$2.0)', 
               '_flatptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$2.0)', 
              }

more_text = [(f'_ptetaflavloss_focalloss_gamma{g}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g}'+r', $\alpha=$'+f'{a})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_alpha{a}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'2.0'+r', $\alpha=$'+f'{a})') for a in alphaparse] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'{g})') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'2.0)') for g in gamma] + \
            [(f'_ptetaflavloss_focalloss' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'2.0)') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}_alpha{a}' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'{g}'+r', $\alpha=$'+f'{a})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g}, $\epsilon=$'+f'{e})') for g, a, e in zip(gamma,alphaparse,epsilon)]

more_text_dict = {k:v for k, v in more_text}
wm_def_text =  {**wm_def_text, **more_text_dict}

#gamma = (weighting_method.split('_gamma')[-1]).split('_alpha')[0]
#alphaparse = (weighting_method.split('_gamma')[-1]).split('_alpha')[-1]
#if gamma != '': print('gamma',gamma)
#if alphaparse != '': print('alpha',alphaparse)
    
colorcode = ['firebrick','magenta','cyan','darkgreen']
colorcode_2 = ['#DA7479','#C89FD4','#63D8F1','#7DFDB4']  # from http://tristen.ca/hcl-picker/#/hlc/4/1/DA7479/7DFDB4
#wm_def_text = {'_noweighting': 'No weighting', 
#               '_ptetaflavloss' : r'$p_T, \eta$ reweighted',
#               '_flatptetaflavloss' : r'$p_T, \eta$ reweighted (Flat)',
#               '_ptetaflavloss_focalloss' : r'$p_T, \eta$ reweighted (F.L.)', 
#               '_flatptetaflavloss_focalloss' : r'$p_T, \eta$ reweighted (Flat, F.L.)',
#               f'_ptetaflavloss_focalloss_gamma{gamma}' : r'$p_T, \eta$ reweighted (F.L. $\gamma=$'+f'{gamma})', 
#               f'_ptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : r'$p_T, \eta$ reweighted (F.L. $\gamma=$'+f'{gamma}'+r',$\alpha=$'+f'{alphaparse})', 
#               f'_ptetaflavloss_focalloss_alpha{alphaparse}' : r'$p_T, \eta$ reweighted (F.L. $\gamma=$'+f'2.0'+r',$\alpha=$'+f'{alphaparse})', 
#               f'_flatptetaflavloss_focalloss_gamma{gamma}' : r'$p_T, \eta$ reweighted (Flat, F.L. $\gamma=$'+f'{gamma})', 
#               f'_flatptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : r'$p_T, \eta$ reweighted (Flat, F.L. $\gamma=$'+f'{gamma}'+r',$\alpha=$'+f'{alphaparse})',
#              }
wm_def_color = {'_noweighting': '#92638C', 
               '_ptetaflavloss' : '#F06644',
               '_flatptetaflavloss' : '#7AC7A3',
               '_ptetaflavloss_focalloss' : '#FEC55C', 
               '_flatptetaflavloss_focalloss' : '#4BC2D8',
               #f'_ptetaflavloss_focalloss_gamma{gamma}' : '#FEC55C', 
               #f'_ptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : '#FEC55C', 
               #f'_ptetaflavloss_focalloss_alpha{alphaparse}' : '#FEC55C', 
               #f'_flatptetaflavloss_focalloss_gamma{gamma}' : '#4BC2D8',
               #f'_flatptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}' : '#4BC2D8',
               #f'_flatptetaflavloss_focalloss_gamma{gamma}_alpha{alphaparse}_adv_tr_eps{epsilon}' : '#4BC2D8',
              }
more_color = [(f'_ptetaflavloss_focalloss_gamma{g}' , '#FEC55C') for g in gamma] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}' , '#FEC55C') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_alpha{a}' , '#FEC55C') for a in alphaparse] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}' , '#4BC2D8') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}_alpha{a}' , '#4BC2D8') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}' , '#FEC55C') for g, e in zip(gamma,epsilon)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}_adv_tr_eps{e}' , '#FEC55C') for g, a, e in zip(gamma,alphaparse,epsilon)] + \
            [(f'_ptetaflavloss_adv_tr_eps{e}' , '#FEC55C') for e in epsilon] + \
            [(f'_flatptetaflavloss_focalloss' , '#4BC2D8') for g in gamma] + \
            [(f'_ptetaflavloss_focalloss' , '#FEC55C') for g in gamma]

more_color_dict = {k:v for k, v in more_color}
wm_def_color =  {**wm_def_color, **more_color_dict}

# 51 bin edges betweeen 0 and 1 --> 50 bins of width 0.02, plus two additional bins at -0.05 and -0.025, as well as at 1.025 and 1.05
# in total: 54 bins, 55 bin edges
# ensures that there are bin edges at 0 and 1 ('exactly') with the option to plot DeepCSV defaults close to the other values
bins = np.append(np.insert(np.linspace(0,0.98,50),0,[-0.05,-0.025]),[1.00001,1.025,1.05])
#print(bins)
#sys.exit()
# Loading data will be necessary for all use cases

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


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len_test)

#test_target_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]
#DeepCSV_testset_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]

test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
# it might be that DeepCSV has some default outputs (-1 bin), plot them at the lower end of the histograms
DeepCSV_testset[:,0][DeepCSV_testset[:,0] < 0] = -0.045
DeepCSV_testset[:,1][DeepCSV_testset[:,1] < 0] = -0.045
DeepCSV_testset[:,2][DeepCSV_testset[:,2] < 0] = -0.045
DeepCSV_testset[:,3][DeepCSV_testset[:,3] < 0] = -0.045
# and if for whatever reason there are values larger than 1, this could also not be interpreted as probability
#DeepCSV_testset[:,0][DeepCSV_testset[:,0] > 1] = +0.99999
#DeepCSV_testset[:,1][DeepCSV_testset[:,1] > 1] = +0.99999
#DeepCSV_testset[:,2][DeepCSV_testset[:,2] > 1] = +0.99999
#DeepCSV_testset[:,3][DeepCSV_testset[:,3] > 1] = +0.99999
print('DeepCSV test done')

jetFlavour = test_targets+1


def calc_BvL(predictions):
    #matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
    #matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
    #matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
    matching_targets = test_targets
    matching_predictions = predictions
    matching_DeepCSV = DeepCSV_testset
    
    #matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
    matching_DeepCSV = np.where(np.tile(((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1), (4,1)).transpose(), matching_DeepCSV, (-1.0)*np.ones((len(matching_targets),4)))
    matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((len(matching_targets),4)))
    
    custom_BvL = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,3]), (-0.045)*np.ones(len(matching_targets)))
    
    DeepCSV_BvL = np.where(((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]), (-0.045)*np.ones(len(matching_targets)))
    
    return custom_BvL, DeepCSV_BvL

def calc_BvC(predictions):
    #matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
    #matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
    #matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
    matching_targets = test_targets
    matching_predictions = predictions
    matching_DeepCSV = DeepCSV_testset
    
    #matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
    matching_DeepCSV = np.where(np.tile(((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1), (4,1)).transpose(), matching_DeepCSV, (-1.0)*np.ones((len(matching_targets),4)))
    matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2] != 0) , (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((len(matching_targets),4)))
    
    custom_BvC = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1), (matching_predictions[:,0]+matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-0.045)*np.ones(len(matching_targets)))
    
    DeepCSV_BvC = np.where(((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]), (-0.045)*np.ones(len(matching_targets)))
    
    return custom_BvC, DeepCSV_BvC
    
def calc_CvB(predictions):
    #matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
    #matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
    #matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
    matching_targets = test_targets
    matching_predictions = predictions
    matching_DeepCSV = DeepCSV_testset
    
    #matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
    matching_DeepCSV = np.where(np.tile((((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)), (4,1)).transpose(), matching_DeepCSV, (-1.0)*np.ones((len(matching_targets),4)))
    matching_predictions = np.where(np.tile((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((len(matching_targets),4)))
    
    custom_CvB = np.where(((matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1), (matching_predictions[:,2])/(matching_predictions[:,0]+matching_predictions[:,1]+matching_predictions[:,2]), (-0.045)*np.ones(len(matching_targets)))
    
    DeepCSV_CvB = np.where(((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1),(matching_DeepCSV[:,2])/(matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,2]), (-0.045)*np.ones(len(matching_targets)))
    
    return custom_CvB, DeepCSV_CvB
    
def calc_CvL(predictions):
    #matching_targets = test_targets[(jetFlavour==3) | (jetFlavour==4)]
    #matching_predictions = predictions[(jetFlavour==3) | (jetFlavour==4)]
    #matching_DeepCSV = DeepCSV_testset[(jetFlavour==3) | (jetFlavour==4)]
    matching_targets = test_targets
    matching_predictions = predictions
    matching_DeepCSV = DeepCSV_testset
    
    #matching_DeepCSV_targets = matching_targets[((matching_DeepCSV[:,0]+matching_DeepCSV[:,1]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)]
    matching_DeepCSV = np.where(np.tile((((matching_DeepCSV[:,2]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1)), (4,1)).transpose(), matching_DeepCSV, (-1.0)*np.ones((len(matching_targets),4)))
    matching_predictions = np.where(np.tile((matching_predictions[:,2]+matching_predictions[:,3] != 0), (4,1)).transpose(), matching_predictions, (-1.0)*np.ones((len(matching_targets),4)))
    
    custom_CvL = np.where(((matching_predictions[:,2]+matching_predictions[:,3]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1), (matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]), (-0.045)*np.ones(len(matching_targets)))
    
    DeepCSV_CvL = np.where(((matching_DeepCSV[:,2]+matching_DeepCSV[:,3]) != 0) & (matching_DeepCSV[:,0] >= 0) & (matching_DeepCSV[:,0] <= 1) & (matching_DeepCSV[:,1] >= 0) & (matching_DeepCSV[:,1] <= 1),(matching_DeepCSV[:,2])/(matching_DeepCSV[:,2]+matching_DeepCSV[:,3]), (-0.045)*np.ones(len(matching_targets)))
    
    return custom_CvL, DeepCSV_CvL







# =============================================================================================================================
#
#
#                           New approach: allows to compare epochs or weighting methods
#
#
# -----------------------------------------------------------------------------------------------------------------------------

# create model, similar for all epochs / weighting methods and only load the weights during the loop
# everything is in evaluation mode, and no gradients are necessary in this script

with torch.no_grad():

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

    if compare_eps:

        KS_test_b_node  =  []
        KS_test_bb_node =  []
        KS_test_c_node  =  []
        KS_test_l_node  =  []
        
        KL_test_b_node  =  []
        KL_test_bb_node =  []
        KL_test_c_node  =  []
        KL_test_l_node  =  []

        for i,e in enumerate(epochs):
            # get predictions and create histograms & KS test to the first specified epoch

            checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{e}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
            model.load_state_dict(checkpoint["model_state_dict"])

            predictions = model(test_inputs).detach().numpy()

            mostprob = np.argmax(predictions, axis=-1)
            cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
            print(f'epoch {e}\n',cfm)
            with open(f'/home/um106329/aisafety/june_21/evaluate/confusion_matrices/weighting_method{weighting_method}_default_{default}_{n_samples}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_minieval_{do_minimal_eval}.npy', 'wb') as f:
                np.save(f, cfm)

            wm_text = wm_def_text[weighting_method]

            classifierHist = hist.Hist("Jets / 0.02 units",
                                hist.Cat("sample","sample name"),
                                hist.Cat("flavour","flavour of the jet"),
                                hist.Bin("probb","P(b)",bins),
                                hist.Bin("probbb","P(bb)",bins),
                                hist.Bin("probc","P(c)",bins),
                                hist.Bin("probudsg","P(udsg)",bins),
                             )

            classifierHist.fill(sample=wm_text,flavour='b-jets',probb=predictions[:,0][jetFlavour==1],probbb=predictions[:,1][jetFlavour==1],probc=predictions[:,2][jetFlavour==1],probudsg=predictions[:,3][jetFlavour==1])
            classifierHist.fill(sample=wm_text,flavour='bb-jets',probb=predictions[:,0][jetFlavour==2],probbb=predictions[:,1][jetFlavour==2],probc=predictions[:,2][jetFlavour==2],probudsg=predictions[:,3][jetFlavour==2])
            classifierHist.fill(sample=wm_text,flavour='c-jets',probb=predictions[:,0][jetFlavour==3],probbb=predictions[:,1][jetFlavour==3],probc=predictions[:,2][jetFlavour==3],probudsg=predictions[:,3][jetFlavour==3])
            classifierHist.fill(sample=wm_text,flavour='udsg-jets',probb=predictions[:,0][jetFlavour==4],probbb=predictions[:,1][jetFlavour==4],probc=predictions[:,2][jetFlavour==4],probudsg=predictions[:,3][jetFlavour==4])
            classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
            classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
            classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
            classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])

            # split per flavour discriminator shapes

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
            #plt.subplots_adjust(wspace=0.4)
            custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('sample','probbb','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linewidth':3})
            custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probc','probudsg'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':colorcode,'linewidth':3})
            custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probbb','probudsg'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':colorcode,'linewidth':3})
            custom_ax4 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probbb','probc'),overlay='flavour',ax=ax4,clear=False,line_opts={'color':colorcode,'linewidth':3})
            dcsv_ax1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
            dcsv_ax2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
            dcsv_ax3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
            dcsv_ax4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
            ax3.legend(loc='upper right',title='Outputs',ncol=1,fontsize=18,title_fontsize=19)
            ax1.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

            ax1.set_ylim(bottom=0, auto=True)
            ax2.set_ylim(bottom=0, auto=True)
            ax3.set_ylim(bottom=0, auto=True)
            ax4.set_ylim(bottom=0, auto=True)

            ax1.set_yscale('log')
            ax2.set_yscale('log')
            #ax3.set_yscale('log')
            #ax4.set_yscale('log')

            ax1.autoscale(True)
            ax2.autoscale(True)
            ax3.autoscale(True)
            ax4.autoscale(True)

            #ax1.ticklabel_format(scilimits=(-5,5))
            #ax2.ticklabel_format(scilimits=(-5,5))
            ax3.ticklabel_format(scilimits=(-5,5))
            ax4.ticklabel_format(scilimits=(-5,5))

            fig.suptitle(f'Classifier and DeepCSV outputs, {wm_text}\nAfter {e} epochs, evaluated on {len_test} jets, default {default}')
            fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/weighting_method{weighting_method}_default_{default}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
            fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/weighting_method{weighting_method}_default_{default}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
            fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/weighting_method{weighting_method}_default_{default}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
            gc.collect()
            plt.show(block=False)
            time.sleep(5)
            plt.clf()
            plt.cla()
            plt.close('all')
            gc.collect(2)

            # stacked discriminator shapes

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
            #plt.subplots_adjust(wspace=0.4)
            custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
            custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
            custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
            custom_ax4 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
            dcsv_ax1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
            dcsv_ax2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
            dcsv_ax3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
            dcsv_ax4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
            ax3.legend(loc='upper right',title='Outputs',ncol=1,fontsize=18,title_fontsize=19)
            ax1.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

            ax1.set_ylim(bottom=0, auto=True)
            ax2.set_ylim(bottom=0, auto=True)
            ax3.set_ylim(bottom=0, auto=True)
            ax4.set_ylim(bottom=0, auto=True)

            ax1.set_yscale('log')
            ax2.set_yscale('log')
            #ax3.set_yscale('log')
            #ax4.set_yscale('log')

            ax1.autoscale(True)
            ax2.autoscale(True)
            ax3.autoscale(True)
            ax4.autoscale(True)

            #ax1.ticklabel_format(scilimits=(-5,5))
            #ax2.ticklabel_format(scilimits=(-5,5))
            ax3.ticklabel_format(scilimits=(-5,5))
            ax4.ticklabel_format(scilimits=(-5,5))

            fig.suptitle(f'Classifier and DeepCSV outputs, {wm_text}\nAfter {e} epochs, evaluated on {len_test} jets, default {default}')
            fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
            fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
            fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{e}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
            gc.collect()
            plt.show(block=False)
            time.sleep(5)
            plt.clf()
            plt.cla()
            plt.close('all')
            gc.collect(2)





            # check P(b) histogram
            # had to learn how to access the values from the multid. histogram / coffea hist stuff
            #classifierHist[wm_text].sum('sample','probbb','probc','probudsg')['b-jets'].values()[()]
            #print(classifierHist[wm_text].sum('sample','probbb','probc','probudsg').dense_axes())
            #print(classifierHist[wm_text].sum('sample','probbb','probc','probudsg')['b-jets'])
            #print(classifierHist[wm_text].sum('flavour','probbb','probc','probudsg').values()[(wm_text,)])
            #sys.exit()
            
            # there are four tagger outputs and in each there will be the entries per flavour, or all flavours together
            # with sum one has to specify everything that is 'not' wanted individually
            probb_b     = classifierHist[wm_text].sum('sample', 'probbb','probc','probudsg').values()[('b-jets',)]
            probb_bb    = classifierHist[wm_text].sum('sample', 'probbb','probc','probudsg').values()[('bb-jets',)]
            probb_c     = classifierHist[wm_text].sum('sample', 'probbb','probc','probudsg').values()[('c-jets',)]
            probb_l     = classifierHist[wm_text].sum('sample', 'probbb','probc','probudsg').values()[('udsg-jets',)]
            probb_stack = classifierHist[wm_text].sum('flavour','probbb','probc','probudsg').values()[(wm_text,)]

            probbb_b     = classifierHist[wm_text].sum('sample', 'probb','probc','probudsg').values()[('b-jets',)]
            probbb_bb    = classifierHist[wm_text].sum('sample', 'probb','probc','probudsg').values()[('bb-jets',)]
            probbb_c     = classifierHist[wm_text].sum('sample', 'probb','probc','probudsg').values()[('c-jets',)]
            probbb_l     = classifierHist[wm_text].sum('sample', 'probb','probc','probudsg').values()[('udsg-jets',)]
            probbb_stack = classifierHist[wm_text].sum('flavour','probb','probc','probudsg').values()[(wm_text,)]

            probc_b     = classifierHist[wm_text].sum('sample', 'probb','probbb','probudsg').values()[('b-jets',)]
            probc_bb    = classifierHist[wm_text].sum('sample', 'probb','probbb','probudsg').values()[('bb-jets',)]
            probc_c     = classifierHist[wm_text].sum('sample', 'probb','probbb','probudsg').values()[('c-jets',)]
            probc_l     = classifierHist[wm_text].sum('sample', 'probb','probbb','probudsg').values()[('udsg-jets',)]
            probc_stack = classifierHist[wm_text].sum('flavour','probb','probbb','probudsg').values()[(wm_text,)]

            probudsg_b     = classifierHist[wm_text].sum('sample', 'probb','probbb','probc').values()[('b-jets',)]
            probudsg_bb    = classifierHist[wm_text].sum('sample', 'probb','probbb','probc').values()[('bb-jets',)]
            probudsg_c     = classifierHist[wm_text].sum('sample', 'probb','probbb','probc').values()[('c-jets',)]
            probudsg_l     = classifierHist[wm_text].sum('sample', 'probb','probbb','probc').values()[('udsg-jets',)]
            probudsg_stack = classifierHist[wm_text].sum('flavour','probb','probbb','probc').values()[(wm_text,)]
            
            
            if i >= 1:
            # calculate every KS test of the current epoch and the previous epoch, per node (KS is symmetric)
                KS_test_b_node.append([
                    np.asarray(ks_2samp(nth_probb_b     , probb_b    )),
                    np.asarray(ks_2samp(nth_probb_bb    , probb_bb   )),
                    np.asarray(ks_2samp(nth_probb_c     , probb_c    )),
                    np.asarray(ks_2samp(nth_probb_l     , probb_l    )),
                    np.asarray(ks_2samp(nth_probb_stack , probb_stack))
                                        ])
                KS_test_bb_node.append([
                    np.asarray(ks_2samp(nth_probbb_b     , probbb_b    )),
                    np.asarray(ks_2samp(nth_probbb_bb    , probbb_bb   )),
                    np.asarray(ks_2samp(nth_probbb_c     , probbb_c    )),
                    np.asarray(ks_2samp(nth_probbb_l     , probbb_l    )),
                    np.asarray(ks_2samp(nth_probbb_stack , probbb_stack))
                                        ])
                KS_test_c_node.append([
                    np.asarray(ks_2samp(nth_probc_b     , probc_b    )),
                    np.asarray(ks_2samp(nth_probc_bb    , probc_bb   )),
                    np.asarray(ks_2samp(nth_probc_c     , probc_c    )),
                    np.asarray(ks_2samp(nth_probc_l     , probc_l    )),
                    np.asarray(ks_2samp(nth_probc_stack , probc_stack))
                                        ])
                KS_test_l_node.append([
                    np.asarray(ks_2samp(nth_probudsg_b     , probudsg_b    )),
                    np.asarray(ks_2samp(nth_probudsg_bb    , probudsg_bb   )),
                    np.asarray(ks_2samp(nth_probudsg_c     , probudsg_c    )),
                    np.asarray(ks_2samp(nth_probudsg_l     , probudsg_l    )),
                    np.asarray(ks_2samp(nth_probudsg_stack , probudsg_stack))
                                        ])  
                
            # calculate every KL divergence of the current epoch and the previous epoch, per node (KL not symmetric, take Q as approx. of P)
                KL_test_b_node.append([
                    np.asarray(entropy(probb_b     , qk=nth_probb_b    )),
                    np.asarray(entropy(probb_bb    , qk=nth_probb_bb   )),
                    np.asarray(entropy(probb_c     , qk=nth_probb_c    )),
                    np.asarray(entropy(probb_l     , qk=nth_probb_l    )),
                    np.asarray(entropy(probb_stack , qk=nth_probb_stack))
                                        ])
                KL_test_bb_node.append([
                    np.asarray(entropy(probbb_b     , qk=nth_probbb_b    )),
                    np.asarray(entropy(probbb_bb    , qk=nth_probbb_bb   )),
                    np.asarray(entropy(probbb_c     , qk=nth_probbb_c    )),
                    np.asarray(entropy(probbb_l     , qk=nth_probbb_l    )),
                    np.asarray(entropy(probbb_stack , qk=nth_probbb_stack))
                                        ])
                KL_test_c_node.append([
                    np.asarray(entropy(probc_b     , qk=nth_probc_b    )),
                    np.asarray(entropy(probc_bb    , qk=nth_probc_bb   )),
                    np.asarray(entropy(probc_c     , qk=nth_probc_c    )),
                    np.asarray(entropy(probc_l     , qk=nth_probc_l    )),
                    np.asarray(entropy(probc_stack , qk=nth_probc_stack))
                                        ])
                KL_test_l_node.append([
                    np.asarray(entropy(probudsg_b     , qk=nth_probudsg_b    )),
                    np.asarray(entropy(probudsg_bb    , qk=nth_probudsg_bb   )),
                    np.asarray(entropy(probudsg_c     , qk=nth_probudsg_c    )),
                    np.asarray(entropy(probudsg_l     , qk=nth_probudsg_l    )),
                    np.asarray(entropy(probudsg_stack , qk=nth_probudsg_stack))
                                        ]) 
            
            
            # save the histograms of the current epoch n to compare this to epoch n+1 during the next iteration of the loop
            nth_probb_b     = probb_b    
            nth_probb_bb    = probb_bb   
            nth_probb_c     = probb_c    
            nth_probb_l     = probb_l    
            nth_probb_stack = probb_stack

            nth_probbb_b     = probbb_b     
            nth_probbb_bb    = probbb_bb    
            nth_probbb_c     = probbb_c     
            nth_probbb_l     = probbb_l     
            nth_probbb_stack = probbb_stack 

            nth_probc_b     = probc_b    
            nth_probc_bb    = probc_bb   
            nth_probc_c     = probc_c    
            nth_probc_l     = probc_l    
            nth_probc_stack = probc_stack

            nth_probudsg_b     = probudsg_b    
            nth_probudsg_bb    = probudsg_bb   
            nth_probudsg_c     = probudsg_c    
            nth_probudsg_l     = probudsg_l    
            nth_probudsg_stack = probudsg_stack

        #print(np.array(KS_test_l_node)[:,0,0])  # was used to debug / check if accessing the entries works correctly after reading the KS result as numpy array
        #sys.exit()
        
        # KS_test_X_node is for tagger output node X and contains (dim 0): entries per epoch; next dim: entries per flavour (b,bb,c,l,stacked); next dim: statistic and p-value of the KS-test

        # epoch --> x-axis, then different plots in same figure: 4 axes, in each will be the entries per output node X, and there should be five lines each (one per flav./stacked)
        # and this for both the statistic and the p-value, so save two figures in the end


        # =================================================================================================================
        # 
        #                                                 Plot KS statistics
        # 
        # -----------------------------------------------------------------------------------------------------------------


        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)

        statistic_b_ax1     = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,0,0],color=colorcode[0],label='b-jets')
        statistic_bb_ax1    = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,1,0],color=colorcode[1],label='bb-jets')
        statistic_c_ax1     = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,2,0],color=colorcode[2],label='c-jets')
        statistic_l_ax1     = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,3,0],color=colorcode[3],label='udsg-jets')
        statistic_stack_ax1 = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,4,0],color='orange',label='all jets')

        statistic_b_ax2     = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,0,0],color=colorcode[0],label='b-jets')
        statistic_bb_ax2    = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,1,0],color=colorcode[1],label='bb-jets')
        statistic_c_ax2     = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,2,0],color=colorcode[2],label='c-jets')
        statistic_l_ax2     = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,3,0],color=colorcode[3],label='udsg-jets')
        statistic_stack_ax2 = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,4,0],color='orange',label='all jets')

        statistic_b_ax3     = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,0,0],color=colorcode[0],label='b-jets')
        statistic_bb_ax3    = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,1,0],color=colorcode[1],label='bb-jets')
        statistic_c_ax3     = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,2,0],color=colorcode[2],label='c-jets')
        statistic_l_ax3     = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,3,0],color=colorcode[3],label='udsg-jets')
        statistic_stack_ax3 = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,4,0],color='orange',label='all jets')

        statistic_b_ax4     = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,0,0],color=colorcode[0],label='b-jets')
        statistic_bb_ax4    = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,1,0],color=colorcode[1],label='bb-jets')
        statistic_c_ax4     = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,2,0],color=colorcode[2],label='c-jets')
        statistic_l_ax4     = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,3,0],color=colorcode[3],label='udsg-jets')
        statistic_stack_ax4 = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,4,0],color='orange',label='all jets')



        ax2.legend(loc='upper right',title='Outputs',ncol=1)
        #ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_title('P(b)')
        ax2.set_title('P(bb)')
        ax3.set_title('P(c)')
        ax4.set_title('P(udsg)')

        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax3.set_xlabel('epoch')
        ax4.set_xlabel('epoch')

        ax1.set_ylabel('KS statistic')
        ax2.set_ylabel('KS statistic')
        ax3.set_ylabel('KS statistic')
        ax4.set_ylabel('KS statistic')

        #ax1.set_yscale('log')
        #ax2.set_yscale('log')
        #ax3.set_yscale('log')
        #ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)


        fig.suptitle(f'KS test statistic, {wm_text}\nAfter {e} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KS_test_nnplus1_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KS_test_nnplus1_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KS_test_nnplus1_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)



        # =================================================================================================================
        # 
        #                                                 Plot p-values
        # 
        # -----------------------------------------------------------------------------------------------------------------


        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)

        pvalue_b_ax1     = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,0,1],color=colorcode[0],label='b-jets')
        pvalue_bb_ax1    = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,1,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax1     = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,2,1],color=colorcode[2],label='c-jets')
        pvalue_l_ax1     = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,3,1],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax1 = ax1.plot(epochs[1:],np.array(KS_test_b_node)[:,4,1],color='orange',label='all jets')

        pvalue_b_ax2     = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,0,1],color=colorcode[0],label='b-jets')
        pvalue_bb_ax2    = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,1,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax2     = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,2,1],color=colorcode[2],label='c-jets')
        pvalue_l_ax2     = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,3,1],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax2 = ax2.plot(epochs[1:],np.array(KS_test_bb_node)[:,4,1],color='orange',label='all jets')

        pvalue_b_ax3     = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,0,1],color=colorcode[0],label='b-jets')
        pvalue_bb_ax3    = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,1,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax3     = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,2,1],color=colorcode[2],label='c-jets')
        pvalue_l_ax3     = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,3,1],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax3 = ax3.plot(epochs[1:],np.array(KS_test_c_node)[:,4,1],color='orange',label='all jets')

        pvalue_b_ax4     = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,0,1],color=colorcode[0],label='b-jets')
        pvalue_bb_ax4    = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,1,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax4     = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,2,1],color=colorcode[2],label='c-jets')
        pvalue_l_ax4     = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,3,1],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax4 = ax4.plot(epochs[1:],np.array(KS_test_l_node)[:,4,1],color='orange',label='all jets')



        ax2.legend(loc='upper right',title='Outputs',ncol=1)
        #ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_title('P(b)')
        ax2.set_title('P(bb)')
        ax3.set_title('P(c)')
        ax4.set_title('P(udsg)')

        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax3.set_xlabel('epoch')
        ax4.set_xlabel('epoch')

        ax1.set_ylabel('KS p-value')
        ax2.set_ylabel('KS p-value')
        ax3.set_ylabel('KS p-value')
        ax4.set_ylabel('KS p-value')

        #ax1.set_yscale('log')
        #ax2.set_yscale('log')
        #ax3.set_yscale('log')
        #ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)


        fig.suptitle(f'KS p-values, {wm_text}\nAfter {e} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KS_test_nnplus1_pvalues_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KS_test_nnplus1_pvalues_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KS_test_nnplus1_pvalues_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)

        
        
        # =================================================================================================================
        # 
        #                                                 Plot KL-divergences
        # 
        # -----------------------------------------------------------------------------------------------------------------


        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)

        pvalue_b_ax1     = ax1.plot(epochs[1:],np.array(KL_test_b_node)[:,0],color=colorcode[0],label='b-jets')
        pvalue_bb_ax1    = ax1.plot(epochs[1:],np.array(KL_test_b_node)[:,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax1     = ax1.plot(epochs[1:],np.array(KL_test_b_node)[:,2],color=colorcode[2],label='c-jets')
        pvalue_l_ax1     = ax1.plot(epochs[1:],np.array(KL_test_b_node)[:,3],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax1 = ax1.plot(epochs[1:],np.array(KL_test_b_node)[:,4],color='orange',label='all jets')

        pvalue_b_ax2     = ax2.plot(epochs[1:],np.array(KL_test_bb_node)[:,0],color=colorcode[0],label='b-jets')
        pvalue_bb_ax2    = ax2.plot(epochs[1:],np.array(KL_test_bb_node)[:,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax2     = ax2.plot(epochs[1:],np.array(KL_test_bb_node)[:,2],color=colorcode[2],label='c-jets')
        pvalue_l_ax2     = ax2.plot(epochs[1:],np.array(KL_test_bb_node)[:,3],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax2 = ax2.plot(epochs[1:],np.array(KL_test_bb_node)[:,4],color='orange',label='all jets')

        pvalue_b_ax3     = ax3.plot(epochs[1:],np.array(KL_test_c_node)[:,0],color=colorcode[0],label='b-jets')
        pvalue_bb_ax3    = ax3.plot(epochs[1:],np.array(KL_test_c_node)[:,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax3     = ax3.plot(epochs[1:],np.array(KL_test_c_node)[:,2],color=colorcode[2],label='c-jets')
        pvalue_l_ax3     = ax3.plot(epochs[1:],np.array(KL_test_c_node)[:,3],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax3 = ax3.plot(epochs[1:],np.array(KL_test_c_node)[:,4],color='orange',label='all jets')

        pvalue_b_ax4     = ax4.plot(epochs[1:],np.array(KL_test_l_node)[:,0],color=colorcode[0],label='b-jets')
        pvalue_bb_ax4    = ax4.plot(epochs[1:],np.array(KL_test_l_node)[:,1],color=colorcode[1],label='bb-jets')
        pvalue_c_ax4     = ax4.plot(epochs[1:],np.array(KL_test_l_node)[:,2],color=colorcode[2],label='c-jets')
        pvalue_l_ax4     = ax4.plot(epochs[1:],np.array(KL_test_l_node)[:,3],color=colorcode[3],label='udsg-jets')
        pvalue_stack_ax4 = ax4.plot(epochs[1:],np.array(KL_test_l_node)[:,4],color='orange',label='all jets')



        ax2.legend(loc='upper right',title='Outputs',ncol=1)
        #ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_title('P(b)')
        ax2.set_title('P(bb)')
        ax3.set_title('P(c)')
        ax4.set_title('P(udsg)')

        ax1.set_xlabel('epoch')
        ax2.set_xlabel('epoch')
        ax3.set_xlabel('epoch')
        ax4.set_xlabel('epoch')

        ax1.set_ylabel('KL divergence')
        ax2.set_ylabel('KL divergence')
        ax3.set_ylabel('KL divergence')
        ax4.set_ylabel('KL divergence')

        #ax1.set_yscale('log')
        #ax2.set_yscale('log')
        #ax3.set_yscale('log')
        #ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)


        fig.suptitle(f'KL divergences, {wm_text}\nAfter {e} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KL_test_nnplus1_pvalues_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KL_test_nnplus1_pvalues_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/KL_test_nnplus1_pvalues_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)
        
        
        # -----------------------------------------------------------------------------------------------------------------




    elif compare_wmets:

        wm_texts = []

        # stacked discriminator shapes
        classifierHist = hist.Hist("Jets / 0.02 units",
                                hist.Cat("sample","sample name"),
                                hist.Cat("flavour","flavour of the jet"),
                                hist.Bin("probb","P(b)",bins),
                                hist.Bin("probbb","P(bb)",bins),
                                hist.Bin("probc","P(c)",bins),
                                hist.Bin("probudsg","P(udsg)",bins),
                             )

        classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
        classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
        classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
        classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)

        dcsv_ax1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
        dcsv_ax2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
        dcsv_ax3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})
        dcsv_ax4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.6,'facecolor':'red'})

        for w in wmets:
            # get predictions and create histograms

            checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{w}_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted{w}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
            model.load_state_dict(checkpoint["model_state_dict"])

            predictions = model(test_inputs).detach().numpy()

            wm_text = wm_def_text[w]
            wm_texts.append(wm_text)

            classifierHist.fill(sample=wm_text,flavour='b-jets',probb=predictions[:,0][jetFlavour==1],probbb=predictions[:,1][jetFlavour==1],probc=predictions[:,2][jetFlavour==1],probudsg=predictions[:,3][jetFlavour==1])
            classifierHist.fill(sample=wm_text,flavour='bb-jets',probb=predictions[:,0][jetFlavour==2],probbb=predictions[:,1][jetFlavour==2],probc=predictions[:,2][jetFlavour==2],probudsg=predictions[:,3][jetFlavour==2])
            classifierHist.fill(sample=wm_text,flavour='c-jets',probb=predictions[:,0][jetFlavour==3],probbb=predictions[:,1][jetFlavour==3],probc=predictions[:,2][jetFlavour==3],probudsg=predictions[:,3][jetFlavour==3])
            classifierHist.fill(sample=wm_text,flavour='udsg-jets',probb=predictions[:,0][jetFlavour==4],probbb=predictions[:,1][jetFlavour==4],probc=predictions[:,2][jetFlavour==4],probudsg=predictions[:,3][jetFlavour==4])



            #plt.subplots_adjust(wspace=0.4)
            custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':wm_def_color[w],'linewidth':3})
            custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':wm_def_color[w],'linewidth':3})
            custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':wm_def_color[w],'linewidth':3})
            custom_ax4 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':wm_def_color[w],'linewidth':3})


        ax3.legend(loc='upper right',title='Outputs',ncol=1,fontsize=18,title_fontsize=19)
        ax1.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        #ax3.set_yscale('log')
        #ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)

        #ax1.ticklabel_format(scilimits=(-5,5))
        #ax2.ticklabel_format(scilimits=(-5,5))
        ax3.ticklabel_format(scilimits=(-5,5))
        ax4.ticklabel_format(scilimits=(-5,5))

        fig.suptitle(f'Classifier and DeepCSV outputs\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)

        # -----------------------------------------------------------------------------------------------------------------

    else:
        
        # this is just a very quick plot of one epoch, one weighting method only

        checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        predictions = model(test_inputs).detach().numpy()
        
        wm_text = wm_def_text[weighting_method]
        
        
        #mostprob = np.argmax(predictions, axis=-1)
        #cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
        #print(f'epoch {at_epoch}\n',cfm)
        #with open(f'/home/um106329/aisafety/june_21/evaluate/confusion_matrices/weighting_method{weighting_method}_default_{default}_{n_samples}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_minieval_{do_minimal_eval}.npy', 'wb') as f:
        #    np.save(f, cfm)


        classifierHist = hist.Hist("Jets / 0.02 units",
                            hist.Cat("sample","sample name"),
                            hist.Cat("flavour","flavour of the jet"),
                            hist.Bin("probb","P(b)",bins),
                            hist.Bin("probbb","P(bb)",bins),
                            hist.Bin("probc","P(c)",bins),
                            hist.Bin("probudsg","P(udsg)",bins),
                            #hist.Bin("bvl","B vs L",bins),
                            #hist.Bin("bvc","B vs C",bins),
                            #hist.Bin("cvb","C vs B",bins),
                            #hist.Bin("cvl","C vs L",bins),
                         )

        classifierHist.fill(sample=wm_text,flavour='b-jets',probb=predictions[:,0][jetFlavour==1],probbb=predictions[:,1][jetFlavour==1],probc=predictions[:,2][jetFlavour==1],probudsg=predictions[:,3][jetFlavour==1])
        classifierHist.fill(sample=wm_text,flavour='bb-jets',probb=predictions[:,0][jetFlavour==2],probbb=predictions[:,1][jetFlavour==2],probc=predictions[:,2][jetFlavour==2],probudsg=predictions[:,3][jetFlavour==2])
        classifierHist.fill(sample=wm_text,flavour='c-jets',probb=predictions[:,0][jetFlavour==3],probbb=predictions[:,1][jetFlavour==3],probc=predictions[:,2][jetFlavour==3],probudsg=predictions[:,3][jetFlavour==3])
        classifierHist.fill(sample=wm_text,flavour='udsg-jets',probb=predictions[:,0][jetFlavour==4],probbb=predictions[:,1][jetFlavour==4],probc=predictions[:,2][jetFlavour==4],probudsg=predictions[:,3][jetFlavour==4])
        classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
        classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
        classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
        classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])

        
        # =================================================================================================================
        # 
        #                                           Plot outputs (not stacked)
        # 
        # -----------------------------------------------------------------------------------------------------------------
        # split per flavour outputs

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
        #plt.subplots_adjust(wspace=0.4)
        custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('sample','probbb','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linewidth':3})
        custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probc','probudsg'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':colorcode,'linewidth':3})
        custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probbb','probudsg'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':colorcode,'linewidth':3})
        custom_ax4 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probbb','probc'),overlay='flavour',ax=ax4,clear=False,line_opts={'color':colorcode,'linewidth':3})
        dcsv_ax1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        dcsv_ax2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        dcsv_ax3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        dcsv_ax4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        #ax3.legend(loc='upper right',title=f'Outputs',ncol=1,fontsize=18,title_fontsize=19,facecolor='k', framealpha=0.3)
        #ax2.legend(loc='upper right',ncol=1,fontsize=18,facecolor='k', framealpha=0.3)
        # gamma25
        #ax2.legend(loc='upper right',ncol=1,fontsize=18)
        #ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()
        # gamma2
        ax1.legend(loc='upper center',ncol=1,fontsize=18)
        ax2.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)

        #ax1.ticklabel_format(scilimits=(-5,5))
        #ax2.ticklabel_format(scilimits=(-5,5))
        #ax3.ticklabel_format(scilimits=(-5,5))
        #ax4.ticklabel_format(scilimits=(-5,5))
        # for gamma25
        #ax1.text(0.25,1e6,f'{wm_text}, epoch {at_epoch}',fontsize=15)
        # for gamma2
        ax2.text(0.33,5e6,f'{wm_text}, epoch {at_epoch}',fontsize=15)
        #fig.suptitle(f'Classifier and DeepCSV outputs, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)
        
        # =================================================================================================================
        # 
        #                                           Plot outputs (stacked)
        # 
        # -----------------------------------------------------------------------------------------------------------------
        # stacked discriminator shapes

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
        #plt.subplots_adjust(wspace=0.4)
        custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
        custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
        custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
        custom_ax4 = hist.plot1d(classifierHist[wm_text].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':wm_def_color[weighting_method],'linewidth':3})
        dcsv_ax1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.5,'facecolor':'#404040'})
        dcsv_ax2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.5,'facecolor':'#404040'})
        dcsv_ax3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.5,'facecolor':'#404040'})
        dcsv_ax4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.5,'facecolor':'#404040'})
        ax3.legend(loc='upper right',title=f'Outputs',ncol=1,fontsize=18,title_fontsize=19)
        ax1.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        #ax3.set_yscale('log')
        #ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)

        #ax1.ticklabel_format(scilimits=(-5,5))
        #ax2.ticklabel_format(scilimits=(-5,5))
        ax3.ticklabel_format(scilimits=(-5,5))
        ax4.ticklabel_format(scilimits=(-5,5))

        fig.suptitle(f'Classifier and DeepCSV outputs, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)
        
        # =================================================================================================================
        # 
        #                                           Plot outputs (stacked v2)
        # 
        # -----------------------------------------------------------------------------------------------------------------
        # new way to display the stacked histograms
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
        #plt.subplots_adjust(wspace=0.4)
        custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('sample','probbb','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probc','probudsg'),overlay='flavour',ax=ax2,clear=False, fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probbb','probudsg'),overlay='flavour',ax=ax3,clear=False,fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        custom_ax4 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probbb','probc'),overlay='flavour',ax=ax4,clear=False,   fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        dcsv_ax1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'linewidth':1,'color':'black'})
        dcsv_ax2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False, line_opts={'linewidth':1,'color':'black'})
        dcsv_ax3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'linewidth':1,'color':'black'})
        dcsv_ax4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,   line_opts={'linewidth':1,'color':'black'})
        ax3.legend(loc='upper right',title=f'Outputs',ncol=1,fontsize=18,title_fontsize=19)
        ax1.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        #ax3.set_yscale('log')
        #ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)

        #ax1.ticklabel_format(scilimits=(-5,5))
        #ax2.ticklabel_format(scilimits=(-5,5))
        ax3.ticklabel_format(scilimits=(-5,5))
        ax4.ticklabel_format(scilimits=(-5,5))

        fig.suptitle(f'Classifier and DeepCSV outputs, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_v2_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_v2_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/stacked_v2_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)
        
        
        
        
        
        del classifierHist
        gc.collect()
        
        
        
        # =================================================================================================================
        # 
        #                                     Plot discriminator shapes (not stacked)
        # 
        # -----------------------------------------------------------------------------------------------------------------
        
        custom_BvL, DeepCSV_BvL = calc_BvL(predictions)
        custom_BvC, DeepCSV_BvC = calc_BvC(predictions)
        custom_CvB, DeepCSV_CvB = calc_CvB(predictions)
        custom_CvL, DeepCSV_CvL = calc_CvL(predictions)
        
        if check_inputs != 'yes':
            del predictions
            gc.collect()
        
        discriminatorHist = hist.Hist("Jets / 0.02 units",
                            hist.Cat("sample","sample name"),
                            hist.Cat("flavour","flavour of the jet"),
                            #hist.Bin("probb","P(b)",bins),
                            #hist.Bin("probbb","P(bb)",bins),
                            #hist.Bin("probc","P(c)",bins),
                            #hist.Bin("probudsg","P(udsg)",bins),
                            hist.Bin("bvl","B vs L",bins),
                            hist.Bin("bvc","B vs C",bins),
                            hist.Bin("cvb","C vs B",bins),
                            hist.Bin("cvl","C vs L",bins),
                         )
        
        discriminatorHist.fill(sample=wm_text,flavour='b-jets',bvl=custom_BvL[jetFlavour==1],bvc=custom_BvC[jetFlavour==1],cvb=custom_CvB[jetFlavour==1],cvl=custom_CvL[jetFlavour==1])
        discriminatorHist.fill(sample=wm_text,flavour='bb-jets',bvl=custom_BvL[jetFlavour==2],bvc=custom_BvC[jetFlavour==2],cvb=custom_CvB[jetFlavour==2],cvl=custom_CvL[jetFlavour==2])
        discriminatorHist.fill(sample=wm_text,flavour='c-jets',bvl=custom_BvL[jetFlavour==3],bvc=custom_BvC[jetFlavour==3],cvb=custom_CvB[jetFlavour==3],cvl=custom_CvL[jetFlavour==3])
        discriminatorHist.fill(sample=wm_text,flavour='udsg-jets',bvl=custom_BvL[jetFlavour==4],bvc=custom_BvC[jetFlavour==4],cvb=custom_CvB[jetFlavour==4],cvl=custom_CvL[jetFlavour==4])
        discriminatorHist.fill(sample="DeepCSV",flavour='b-jets',bvl=DeepCSV_BvL[jetFlavour==1],bvc=DeepCSV_BvC[jetFlavour==1],cvb=DeepCSV_CvB[jetFlavour==1],cvl=DeepCSV_CvL[jetFlavour==1])
        discriminatorHist.fill(sample="DeepCSV",flavour='bb-jets',bvl=DeepCSV_BvL[jetFlavour==2],bvc=DeepCSV_BvC[jetFlavour==2],cvb=DeepCSV_CvB[jetFlavour==2],cvl=DeepCSV_CvL[jetFlavour==2])
        discriminatorHist.fill(sample="DeepCSV",flavour='c-jets',bvl=DeepCSV_BvL[jetFlavour==3],bvc=DeepCSV_BvC[jetFlavour==3],cvb=DeepCSV_CvB[jetFlavour==3],cvl=DeepCSV_CvL[jetFlavour==3])
        discriminatorHist.fill(sample="DeepCSV",flavour='udsg-jets',bvl=DeepCSV_BvL[jetFlavour==4],bvc=DeepCSV_BvC[jetFlavour==4],cvb=DeepCSV_CvB[jetFlavour==4],cvl=DeepCSV_CvL[jetFlavour==4])


        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
        #plt.subplots_adjust(wspace=0.4)
        custom_ax1 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linewidth':3})
        custom_ax2 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':colorcode,'linewidth':3})
        custom_ax3 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':colorcode,'linewidth':3})
        custom_ax4 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb'),overlay='flavour',ax=ax4,clear=False,line_opts={'color':colorcode,'linewidth':3})
        dcsv_ax1 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvc','cvb','cvl'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        dcsv_ax2 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvl','cvb','cvl'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        dcsv_ax3 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvl','bvc','cvl'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        dcsv_ax4 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvl','bvc','cvb'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
        # gamma25
        #ax1.legend(loc=(0.67,0.7),ncol=1,fontsize=13.5)
        #ax3.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()
        # gamma2
        ax1.legend(loc='upper center',ncol=1,fontsize=13.5)
        ax3.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)

        #ax1.ticklabel_format(scilimits=(-5,5))
        #ax2.ticklabel_format(scilimits=(-5,5))
        #ax3.ticklabel_format(scilimits=(-5,5))
        #ax4.ticklabel_format(scilimits=(-5,5))
        # for adversarial training gamma25
        #ax4.text(0.49,5e5,f'{wm_text},\nepoch {at_epoch}',fontsize=14)
        # for basic training gamma25
        #ax4.text(0.59,5e5,f'{wm_text},\nepoch {at_epoch}',fontsize=14)
        # for basic training gamma2
        ax2.text(0.33,5e6,f'{wm_text}, epoch {at_epoch}',fontsize=14)
        #fig.suptitle(f'Classifier and DeepCSV discriminators, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/discriminators_versus_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/discriminators_versus_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/discriminators_versus_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)
        
        
        # =================================================================================================================
        # 
        #                                     Plot discriminator shapes (stacked)
        # 
        # -----------------------------------------------------------------------------------------------------------------
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
        #plt.subplots_adjust(wspace=0.4)
        custom_ax1 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl'),overlay='flavour',ax=ax1,clear=False,fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        custom_ax2 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl'),overlay='flavour',ax=ax2,clear=False,fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        custom_ax3 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl'),overlay='flavour',ax=ax3,clear=False,fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        custom_ax4 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb'),overlay='flavour',ax=ax4,clear=False,fill_opts={'facecolor':colorcode_2,'alpha':1},stack=True)
        dcsv_ax1 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvc','cvb','cvl'),ax=ax1,clear=False,line_opts={'linewidth':1,'color':'black'})
        dcsv_ax2 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvl','cvb','cvl'),ax=ax2,clear=False,line_opts={'linewidth':1,'color':'black'})
        dcsv_ax3 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvl','bvc','cvl'),ax=ax3,clear=False,line_opts={'linewidth':1,'color':'black'})
        dcsv_ax4 = hist.plot1d(discriminatorHist['DeepCSV'].sum('flavour','bvl','bvc','cvb'),ax=ax4,clear=False,line_opts={'linewidth':1,'color':'black'})
        ax1.legend(loc='upper center',title=f'{wm_text}, epoch {at_epoch}',ncol=1,fontsize=17,title_fontsize=17.5)
        ax3.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()

        ax1.set_ylim(bottom=0, auto=True)
        ax2.set_ylim(bottom=0, auto=True)
        ax3.set_ylim(bottom=0, auto=True)
        ax4.set_ylim(bottom=0, auto=True)

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')

        ax1.autoscale(True)
        ax2.autoscale(True)
        ax3.autoscale(True)
        ax4.autoscale(True)

        #ax1.ticklabel_format(scilimits=(-5,5))
        #ax2.ticklabel_format(scilimits=(-5,5))
        #ax3.ticklabel_format(scilimits=(-5,5))
        #ax4.ticklabel_format(scilimits=(-5,5))

        #fig.suptitle(f'Classifier and DeepCSV discriminators, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/discriminators_versus_stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=400)
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/discriminators_versus_stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.pdf', bbox_inches='tight')
        fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/discriminators_versus_stacked_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.svg', bbox_inches='tight')
        gc.collect()
        plt.show(block=False)
        time.sleep(5)
        plt.clf()
        plt.cla()
        plt.close('all')
        gc.collect(2)
        
        
        # =================================================================================================================
        # 
        #                    For the different bins of the tagger outputs, show interesting input variables
        # 
        # -----------------------------------------------------------------------------------------------------------------
        
        if check_inputs == 'yes':
        
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
        
               
            for b in range(2,len(bins)-3):


                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)

                # for variable in [0,1,2,3,4,5,6,7,8,9,10,11,12,63,64,65,66]:
                for i, variable in enumerate([0,1,12,64]):
                    scaler = torch.load(f'/hpcwork/um106329/june_21/scaler_{variable}_with_default_{default}.pt')
                    scaledback = scaler.inverse_transform(test_inputs[:,variable])

                    in_this_bin_b  = scaledback[(predictions[:,3] >= bins[b]) & (predictions[:,3] < bins[b+1]) & (jetFlavour.numpy() == 1)]
                    in_this_bin_bb = scaledback[(predictions[:,3] >= bins[b]) & (predictions[:,3] < bins[b+1]) & (jetFlavour.numpy() == 2)]
                    in_this_bin_c  = scaledback[(predictions[:,3] >= bins[b]) & (predictions[:,3] < bins[b+1]) & (jetFlavour.numpy() == 3)]
                    in_this_bin_l  = scaledback[(predictions[:,3] >= bins[b]) & (predictions[:,3] < bins[b+1]) & (jetFlavour.numpy() == 4)]

                    #if variable == 1:
                    #    exec("ax%s.hist(in_this_bin_b, bins=100, range=(0,1000), histtype='step', label='b-jets', linewidth=2)"%(i+1))
                    #    exec("ax%s.hist(in_this_bin_bb, bins=100, range=(0,1000), histtype='step', label='bb-jets', linewidth=2)"%(i+1))
                    #    exec("ax%s.hist(in_this_bin_c, bins=100, range=(0,1000), histtype='step', label='c-jets', linewidth=2)"%(i+1))
                    #    exec("ax%s.hist(in_this_bin_l, bins=100, range=(0,1000), histtype='step', label='udsg-jets', linewidth=2)"%(i+1))

                    #else:
                    exec("ax%s.hist(in_this_bin_b, bins=100, histtype='step', range=(min(scaledback),max(scaledback)), label='b-jets', linewidth=2)"%(i+1))
                    exec("ax%s.hist(in_this_bin_bb, bins=100, histtype='step',range=(min(scaledback),max(scaledback)), label='bb-jets', linewidth=2)"%(i+1))
                    exec("ax%s.hist(in_this_bin_c, bins=100, histtype='step', range=(min(scaledback),max(scaledback)), label='c-jets', linewidth=2)"%(i+1))
                    exec("ax%s.hist(in_this_bin_l, bins=100, histtype='step', range=(min(scaledback),max(scaledback)), label='udsg-jets', linewidth=2)"%(i+1))
                    exec("ax%s.set_xlabel(display_names[%s])"%((i+1),(variable)))
                    exec("ax%s.set_ylabel('Jets')"%(i+1))


                ax1.set_ylim(bottom=0, auto=True)
                ax2.set_ylim(bottom=0, auto=True)
                ax3.set_ylim(bottom=0, auto=True)
                ax4.set_ylim(bottom=0, auto=True)

                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax3.set_yscale('log')
                #ax4.set_yscale('log')

                ax1.autoscale(True)
                ax2.autoscale(True)
                ax3.autoscale(True)
                ax4.autoscale(True)

                #ax1.ticklabel_format(scilimits=(-5,5))
                #ax2.ticklabel_format(scilimits=(-5,5))
                #ax3.ticklabel_format(scilimits=(-5,5))
                ax4.ticklabel_format(scilimits=(-3,3))

                ax4.legend()

                fig.suptitle(f'Inputs in Prob(udsg) bin [{bins[b]:.2f},{bins[b+1]:.2f}], {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
                fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/input_histograms_by_tagger_outputs/weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}_Probudsg_bin_{b}.png', bbox_inches='tight', dpi=400)
                fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/input_histograms_by_tagger_outputs/weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}_Probudsg_bin_{b}.pdf', bbox_inches='tight')
                fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/discriminator_shapes/shapes_new/input_histograms_by_tagger_outputs/weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}_Probudsg_bin_{b}.svg', bbox_inches='tight')
                gc.collect()
                plt.show(block=False)
                time.sleep(5)
                plt.clf()
                plt.cla()
                plt.close('all')
                gc.collect(2)
