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
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
parser.add_argument("wm", help="Weighting method: _as_is, _new or _both")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("compare", help="Compare with earlier epochs", default='no')
args = parser.parse_args()

NUM_DATASETS = args.files
at_epoch = args.prevep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
compare = args.compare

print(f'Evaluate training at epoch {at_epoch}')




'''

    Load inputs and targets
    
'''
test_input_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/test_inputs_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len_test)

test_target_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/test_targets_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/DeepCSV_testset_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]

test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
print('DeepCSV test done')

jetFlavour = test_targets+1


plt.ioff()

if compare == 'no':
    '''

        Predictions: Without weighting

    '''
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



    checkpoint = torch.load(f'/hpcwork/um106329/april_21/saved_models/TT_as_is_{NUM_DATASETS}_{default}/model_all_TT_{at_epoch}_epochs_v10_GPU_weighted_as_is_{NUM_DATASETS}_datasets_with_default_{default}.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)




    #evaluate network on inputs
    model.eval()
    with torch.no_grad():
        predictions_as_is = model(test_inputs).detach().numpy()
    print('predictions without weighting done')


    '''

        Predictions: With new weighting method

    '''
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



    checkpoint = torch.load(f'/hpcwork/um106329/april_21/saved_models/TT_new_{NUM_DATASETS}_{default}/model_all_TT_{at_epoch}_epochs_v10_GPU_weighted_new_{NUM_DATASETS}_datasets_with_default_{default}.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    #evaluate network on inputs
    model.eval()
    with torch.no_grad():
        predictions_new = model(test_inputs).detach().numpy()
    print('predictions with loss weighting done')


    
    if weighting_method == '_both':
        #plot some ROC curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,17],num=4)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_as_is[:,0])
        ax1.plot(fpr,tpr)
        print(f"auc for b-tagging without weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_new[:,0])
        ax1.plot(fpr,tpr)
        print(f"auc for b-tagging with new weighting method: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,0])
        ax1.plot(fpr,tpr)
        print(f"auc for b-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        ax1.legend(['Classifier: No weighting','Classifier: Loss weighting','DeepCSV'])
        ax1.set_xlabel('false positive rate')
        ax1.set_ylabel('true positive rate')
        ax1.set_title('b tagging')
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_as_is[:,1])
        ax2.plot(fpr,tpr)
        print(f"auc for bb-tagging without weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_new[:,1])
        ax2.plot(fpr,tpr)
        print(f"auc for bb-tagging with new weighting method: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,1])
        ax2.plot(fpr,tpr)
        print(f"auc for bb-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        ax2.legend(['Classifier: No weighting','Classifier: Loss weighting','DeepCSV'])
        ax2.set_xlabel('false positive rate')
        ax2.set_ylabel('true positive rate')
        ax2.set_title('bb tagging')
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_as_is[:,2])
        ax3.plot(fpr,tpr)
        print(f"auc for c-tagging as is: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_new[:,2])
        ax3.plot(fpr,tpr)
        print(f"auc for c-tagging new: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,2])
        ax3.plot(fpr,tpr)
        print(f"auc for c-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        ax3.legend(['Classifier: No weighting','Classifier: Loss weighting','DeepCSV'])
        ax3.set_xlabel('false positive rate')
        ax3.set_ylabel('true positive rate')
        ax3.set_title('c tagging')
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_as_is[:,3])
        ax4.plot(fpr,tpr)
        print(f"auc for udsg-tagging as is: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_new[:,3])
        ax4.plot(fpr,tpr)
        print(f"auc for udsg-tagging new: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,3])
        ax4.plot(fpr,tpr)
        print(f"auc for udsg-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        ax4.set_xlabel('false positive rate')
        ax4.set_ylabel('true positive rate')
        ax4.set_title('udsg- tagging')
        ax1.get_legend().remove(), ax2.get_legend().remove(), ax3.get_legend().remove()
        ax4.legend(['Classifier: No weighting','Classifier: Loss weighting','DeepCSV'],loc='lower right')
        fig.suptitle(f'ROCs for b, bb, c and light jets\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/compare_roc_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        gc.collect(2)




    '''
        No weighting alone (for DPG, where b and c alone will be shown)
    '''
    if weighting_method == '_as_is':
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_as_is[:,0])
        plt.plot(fpr,tpr)
        print(f"auc for b-tagging without weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,0])
        plt.plot(fpr,tpr)
        print(f"auc for b-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: No weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC b tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_b_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_as_is[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for bb-tagging without weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for bb-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: No weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC bb tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_bb_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_as_is[:,2])
        plt.plot(fpr,tpr)
        print(f"auc for c-tagging without weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,2])
        plt.plot(fpr,tpr)
        print(f"auc for c-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: No weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC c tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_c_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_as_is[:,3])
        plt.plot(fpr,tpr)
        print(f"auc for udsg-tagging without weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,3])
        plt.plot(fpr,tpr)
        print(f"auc for udsg-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: No weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC udsg tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_udsg_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)


    '''
        Loss weighting alone
    '''
    if weighting_method == '_new':
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_new[:,0])
        plt.plot(fpr,tpr)
        print(f"auc for b-tagging with loss weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,0])
        plt.plot(fpr,tpr)
        print(f"auc for b-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: Loss weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC b tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_b_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_new[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for bb-tagging with loss weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for bb-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: Loss weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC bb tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_bb_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_new[:,2])
        plt.plot(fpr,tpr)
        print(f"auc for c-tagging with loss weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,2])
        plt.plot(fpr,tpr)
        print(f"auc for c-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: Loss weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC c tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_c_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_new[:,3])
        plt.plot(fpr,tpr)
        print(f"auc for udsg-tagging with loss weighting: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,3])
        plt.plot(fpr,tpr)
        print(f"auc for udsg-tagging DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.legend(['Classifier: Loss weighting','DeepCSV'],loc='lower right')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title(f'ROC udsg tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_udsg_tagging_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)





    '''
        B vs Light jets
    '''

    BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==2],test_inputs[jetFlavour==4]))
    del gc.garbage[:]
    del test_inputs
    BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
    del test_targets
    BvsUDSG_predictions_as_is = np.concatenate((predictions_as_is[jetFlavour==1],predictions_as_is[jetFlavour==2],predictions_as_is[jetFlavour==4]))
    del predictions_as_is
    BvsUDSG_predictions_new = np.concatenate((predictions_new[jetFlavour==1],predictions_new[jetFlavour==2],predictions_new[jetFlavour==4]))
    del predictions_new
    BvsUDSG_DeepCSV = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==2],DeepCSV_testset[jetFlavour==4]))
    del DeepCSV_testset
    gc.collect()




    if weighting_method == '_both':
        # plot ROC BvsUDSG
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[:,0]+BvsUDSG_predictions_as_is[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG as is: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[:,0]+BvsUDSG_predictions_new[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG new: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0]+BvsUDSG_DeepCSV[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG DeepCSV: {metrics.auc(fpr,tpr)}")

        plt.xlabel('mistag rate')
        plt.ylabel('efficiency')
        plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')

        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)

        plt.legend(['Classifier: No weighting', 'Classifier: Loss weighting','DeepCSV'],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/compare_roc_BvL_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)



    '''
        No weighting alone
    '''
    if weighting_method == '_as_is':
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[:,0]+BvsUDSG_predictions_as_is[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG as is: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0]+BvsUDSG_DeepCSV[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.xlabel('mistag rate')
        plt.ylabel('efficiency')
        plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)
        plt.legend(['Classifier: No weighting', 'DeepCSV'],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_BvL_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)


    '''
        Loss weighting alone
    '''
    if weighting_method == '_new':
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[:,0]+BvsUDSG_predictions_new[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG as is: {metrics.auc(fpr,tpr)}")
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0]+BvsUDSG_DeepCSV[:,1])
        plt.plot(fpr,tpr)
        print(f"auc for B vs UDSG DeepCSV: {metrics.auc(fpr,tpr)}")
        plt.xlabel('mistag rate')
        plt.ylabel('efficiency')
        plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files, default {default})')
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)
        plt.legend(['Classifier: Loss weighting', 'DeepCSV'],loc='lower right')
        fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_BvL_weighting_method{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)
        
else:
    epochs = [at_epoch - 100, at_epoch - 75, at_epoch - 50, at_epoch - 25, at_epoch]
    '''
    B vs Light jets
    '''

    BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==2],test_inputs[jetFlavour==4]))
    del gc.garbage[:]
    del test_inputs
    BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==2],test_targets[jetFlavour==4]))
    del test_targets
    
    
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



        checkpoint = torch.load(f'/hpcwork/um106329/april_21/saved_models/TT{weighting_method}_{NUM_DATASETS}_{default}/model_all_TT_{i}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}.pt', map_location=torch.device(device))
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)


        #evaluate network on inputs
        model.eval()
        with torch.no_grad():
            BvsUDSG_predictions = model(BvsUDSG_inputs).detach().numpy()
                
        fig = plt.figure(figsize=[15,15],num=40)
        fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 or BvsUDSG_targets[i]==1  else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions[:,0]+BvsUDSG_predictions[:,1])
        plt.plot(fpr,tpr, label=f'{i} epochs')
        plt.xlabel('mistag rate')
        plt.ylabel('efficiency')
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)
    
    if weighting_method == '_as_is':
        text_wm = 'No weighting'
    if weighting_method == '_new':
        text_wm = 'Loss weighting'
        
    plt.legend()
    plt.title(f'ROC for b vs. udsg ({text_wm})\nEvaluated on {len_test} jets ({NUM_DATASETS} files, default {default})', y=1.02)
    fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/roc_BvL_weighting_method{weighting_method}_default_{default}_compare_at_epochs_{epochs}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=400)