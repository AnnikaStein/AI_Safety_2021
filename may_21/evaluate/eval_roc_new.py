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


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use(hep.cms.style.ROOT)
colorcode = ['firebrick','magenta','cyan','darkgreen']


parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("dominimal", help="Only do training with minimal setup, i.e. 15 QCD, 5 TT files")
parser.add_argument("dominimal_eval", help="Only minimal number of files for evaluation")
#parser.add_argument("compare", help="Compare with earlier epochs", default='no')  # one can infer if user wants to compare epochs, if user put in more than one epoch
#parser.add_argument("dofl", help="Use Focal Loss")
args = parser.parse_args()

NUM_DATASETS = args.files
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',')]
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
    
n_samples = args.jets
do_minimal = args.dominimal
do_minimal_eval = args.dominimal_eval
compare = True if len(epochs) > 1 else False
#do_FL = args.dofl

#if do_FL == 'yes':
#    fl_text = '_focalloss'
#else:
#    fl_text = ''
    
print(f'Evaluate training at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')

wm_def_text = {'_noweighting': 'No weighting', 
               '_ptetaflavloss' : 'Loss weighting',
               '_flatptetaflavloss' : 'Loss weighting (Flat)',
               '_ptetaflavloss_focalloss' : 'Loss weighting (Focal Loss)', 
               '_flatptetaflavloss_focalloss' : 'Loss weighting (Flat, Focal Loss)'
              }
wm_def_color = {'_noweighting': 'yellow', 
               '_ptetaflavloss' : 'orange',
               '_flatptetaflavloss' : 'brown',
               '_ptetaflavloss_focalloss' : 'cyan', 
               '_flatptetaflavloss_focalloss' : 'blue'
              }

if do_minimal_eval == 'no':
    test_input_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    test_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    DeepCSV_testset_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    
        
        
if do_minimal_eval == 'yes':
    test_input_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,5)]
    test_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/test_targets_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,5)]
    DeepCSV_testset_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,5)]
    
        
if do_minimal_eval == 'medium':
    rng = np.random.default_rng(12345)
    some_files = rng.integers(low=0, high=278, size=50)
    test_input_file_paths = np.array([f'/hpcwork/um106329/may_21/scaled_QCD/test_inputs_%d_with_default_{default}.pt' % k for k in range(229)] + [f'/hpcwork/um106329/may_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(49)])[some_files]
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
print('DeepCSV test done')

jetFlavour = test_targets+1


#do_noweighting   = True if ( weighting_method == '_all' or weighting_method == '_noweighting'   ) else False
#do_ptetaflavloss = True if ( weighting_method == '_all' or weighting_method == '_ptetaflavloss' ) else False
#do_flatptetaflavloss = True if ( weighting_method == '_all' or weighting_method == '_flatptetaflavloss' ) else False


plt.ioff()

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
    
    if compare == False:
        
        checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        predictions = model(test_inputs).detach().numpy()
        predictions[:,0][predictions[:,0] > 0.999999] = 0.999999
        predictions[:,1][predictions[:,1] > 0.999999] = 0.999999
        predictions[:,2][predictions[:,2] > 0.999999] = 0.999999
        predictions[:,3][predictions[:,3] > 0.999999] = 0.999999
        predictions[:,0][predictions[:,0] < 0.000001] = 0.000001
        predictions[:,1][predictions[:,1] < 0.000001] = 0.000001
        predictions[:,2][predictions[:,2] < 0.000001] = 0.000001
        predictions[:,3][predictions[:,3] < 0.000001] = 0.000001
        
        wm_text = wm_def_text[weighting_method]
        
        '''       
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==0, torch.ones(len_test), torch.zeros(len_test)),predictions[:,0])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for b-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==0, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,0])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for b-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: {wm_text}, AUC = {customauc}', f'DeepCSV, AUC = {deepcsvauc}'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC b tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==1, torch.ones(len_test), torch.zeros(len_test)),predictions[:,1])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for bb-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==1, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,1])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for bb-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: {wm_text}, AUC = {customauc}', f'DeepCSV, AUC = {deepcsvauc}'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC bb tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_bb_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==2, torch.ones(len_test), torch.zeros(len_test)),predictions[:,2])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for c-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==2, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,2])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for c-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: {wm_text}, AUC = {customauc}', f'DeepCSV, AUC = {deepcsvauc}'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC c tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        #fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len_test)],predictions_new_flat[:,3])
        # trying out new way to slice targets instead of looping over them
        #torch.where(m<0,m,n)
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==3, torch.ones(len_test), torch.zeros(len_test)),predictions[:,3])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for udsg-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==3, torch.ones(len_test), torch.zeros(len_test)),DeepCSV_testset[:,3])
        plt.plot(fpr,tpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for udsg-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend([f'Classifier: {wm_text}, AUC = {customauc}', f'DeepCSV, AUC = {deepcsvauc}'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC udsg tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        '''


        '''
            B vs Light jets
        '''
        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del gc.garbage[:]
        #del test_inputs
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del predictions_new_flat
        
        matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==4)]
        #del DeepCSV_testset
        gc.collect()

        len_BvsUDSG = len(matching_targets)


        
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,2]))
        plt.plot(tpr,fpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for B vs UDSG {wm_text}: {customauc}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(1-matching_DeepCSV[:,2]))
        plt.plot(tpr,fpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for B vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        plt.ylim(bottom=1e-4)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}'+', AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        '''
            B vs C jets
        '''
        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del gc.garbage[:]
        #del test_inputs
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del predictions_new_flat
        
        matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del DeepCSV_testset
        gc.collect()

        len_BvsC = len(matching_targets)


        
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsC), torch.zeros(len_BvsC)),(matching_predictions[:,0]+matching_predictions[:,1])/(1-matching_predictions[:,3]))
        plt.plot(tpr,fpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for B vs C {wm_text}: {customauc}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==0) | (matching_targets==1), torch.ones(len_BvsC), torch.zeros(len_BvsC)),(matching_DeepCSV[:,0]+matching_DeepCSV[:,1])/(1-matching_DeepCSV[:,3]))
        plt.plot(tpr,fpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for B vs C DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        plt.title(f'ROC for b vs. c\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-4)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}'+', AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)           
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        '''
            C vs B jets
        '''
        #matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del gc.garbage[:]
        #del test_inputs
        #matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del test_targets
        
        #matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del predictions_new_flat
        
        #matching_DeepCSV = DeepCSV_testset[(jetFlavour==1) | (jetFlavour==2) | (jetFlavour==3)]
        #del DeepCSV_testset
        #gc.collect()

        len_CvsB = len_BvsC
       
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==2), torch.ones(len_CvsB), torch.zeros(len_CvsB)),(matching_predictions[:,2])/(1-matching_predictions[:,3]))
        plt.plot(tpr,fpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for C vs B {wm_text}: {customauc}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==2), torch.ones(len_CvsB), torch.zeros(len_CvsB)),(matching_DeepCSV[:,2])/(1-matching_DeepCSV[:,3]))
        plt.plot(tpr,fpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for C vs B DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        plt.title(f'ROC for c vs. b\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-4)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}'+', AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400) 
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)         
         
        '''
            C vs Light jets
        '''
        matching_inputs = test_inputs[(jetFlavour==3) | (jetFlavour==4)]
        #del gc.garbage[:]
        #del test_inputs
        matching_targets = test_targets[(jetFlavour==3) | (jetFlavour==4)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==3) | (jetFlavour==4)]
        #del predictions_new_flat
        
        matching_DeepCSV = DeepCSV_testset[(jetFlavour==3) | (jetFlavour==4)]
        #del DeepCSV_testset
        gc.collect()

        len_CvsUDSG = len(matching_targets)


        
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==2), torch.ones(len_CvsUDSG), torch.zeros(len_CvsUDSG)),(matching_predictions[:,2])/(matching_predictions[:,2]+matching_predictions[:,3]))
        plt.plot(tpr,fpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for C vs UDSG {wm_text}: {customauc}")
        fpr,tpr,thresholds = metrics.roc_curve(torch.where((matching_targets==2), torch.ones(len_CvsUDSG), torch.zeros(len_CvsUDSG)),(matching_DeepCSV[:,2])/(matching_DeepCSV[:,2]+matching_DeepCSV[:,3]))
        plt.plot(tpr,fpr)
        deepcsvauc = metrics.auc(fpr,tpr)
        print(f"auc for C vs UDSG DeepCSV: {deepcsvauc}")
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        plt.title(f'ROC for c vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-4)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}'+', AUC = {:.4f}'.format(customauc), f'DeepCSV'+', AUC = {:.4f}'.format(deepcsvauc)],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
                 
            
    else:
        #epochs = [at_epoch - 100, at_epoch - 75, at_epoch - 50, at_epoch - 25, at_epoch]  # can be controlled manually with argparser
        '''
        B vs Light jets
        '''

        BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==2],test_inputs[jetFlavour==4]))
        del gc.garbage[:]
        del test_inputs
        BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
        del test_targets
        len_BvsUDSG = len(BvsUDSG_targets)

        for i in epochs:
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

            checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/{weighting_method}{fl_text}_{NUM_DATASETS}_{default}_{n_samples}/model_{i}_epochs_v10_GPU_weighted{weighting_method}{fl_text}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
            model.load_state_dict(checkpoint["model_state_dict"])

            model.to(device)


            #evaluate network on inputs
            model.eval()
            with torch.no_grad():
                BvsUDSG_predictions = model(BvsUDSG_inputs).detach().numpy()

            fig = plt.figure(figsize=[15,15],num=40)
            fpr,tpr,thresholds = metrics.roc_curve(torch.where((BvsUDSG_targets==0) | (BvsUDSG_targets==1), torch.ones(len_BvsUDSG), torch.zeros(len_BvsUDSG)),BvsUDSG_predictions[:,0]+BvsUDSG_predictions[:,1])
            plt.plot(fpr,tpr, label=f'{i} epochs')
            plt.xlabel('mistag rate')
            plt.ylabel('efficiency')
            plt.xlim(-0.05,1.05)
            plt.ylim(-0.05,1.05)

        if weighting_method == '_noweighting':
            text_wm = 'No weighting'
        if weighting_method == '_ptetaflavloss':
            text_wm = 'Loss weighting'
        if weighting_method == '_flatptetaflavloss':
            text_wm = 'Loss weighting (flat)'

        plt.legend()
        plt.title(f'ROC for b vs. udsg ({text_wm})\nEvaluated on {len_test} jets ({NUM_DATASETS} files, default {default})', y=1.02)
        fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/roc_curves/roc_BvL_weighting_method{weighting_method}{fl_text}_compare_at_epochs_{epochs}_{len_test}_jets_training_{NUM_DATASETS}_{default}_{n_samples}.png', bbox_inches='tight', dpi=400)