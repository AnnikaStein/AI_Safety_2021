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

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use(hep.cms.style.ROOT)
colorcode = ['firebrick','magenta','cyan','darkgreen']

#oversampling = False  # deprecated, as WeightedRandomSampler will not be used

parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss or _compare")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("dominimal", help="Only do training with minimal setup, i.e. 15 QCD, 5 TT files")
parser.add_argument("dominimal_eval", help="Only minimal number of files for evaluation")
args = parser.parse_args()

NUM_DATASETS = args.files
at_epoch = args.prevep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)

n_samples = args.jets
do_minimal = args.dominimal
do_minimal_eval = args.dominimal_eval

print(f'Evaluate training at epoch {at_epoch}')

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





do_noweighting   = True if ( weighting_method == '_compare' or weighting_method == '_noweighting'   ) else False
do_ptetaflavloss = True if ( weighting_method == '_compare' or weighting_method == '_ptetaflavloss' ) else False
do_flatptetaflavloss = True if ( weighting_method == '_compare' or weighting_method == '_flatptetaflavloss' ) else False



'''

    Predictions: Without weighting
    
'''
if do_noweighting:
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



    checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/_noweighting_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted_noweighting_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)




    #evaluate network on inputs
    model.eval()
    predictions_as_is = model(test_inputs).detach().numpy()

    # to find out if the values are in realistic bounds between 0 and 1, for debugging if needed
    '''
    hist, bin_edges = np.histogram(predictions_as_is[:,0],bins=20)
    print('Flavour b predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()
    hist, bin_edges = np.histogram(predictions_as_is[:,1],bins=20)
    print('Flavour bb predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()
    hist, bin_edges = np.histogram(predictions_as_is[:,2],bins=20)
    print('Flavour c predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()
    hist, bin_edges = np.histogram(predictions_as_is[:,3],bins=20)
    print('Flavour udsg predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()


    print(np.unique(predictions_as_is))
    predictions_as_is[:,0][predictions_as_is[:,0] > 1.] = 0.999999
    predictions_as_is[:,1][predictions_as_is[:,1] > 1.] = 0.999999
    predictions_as_is[:,2][predictions_as_is[:,2] > 1.] = 0.999999
    predictions_as_is[:,3][predictions_as_is[:,3] > 1.] = 0.999999
    predictions_as_is[:,0][predictions_as_is[:,0] < 0.] = 0.000001
    predictions_as_is[:,1][predictions_as_is[:,1] < 0.] = 0.000001
    predictions_as_is[:,2][predictions_as_is[:,2] < 0.] = 0.000001
    predictions_as_is[:,3][predictions_as_is[:,3] < 0.] = 0.000001
    print(np.unique(predictions_as_is))


    hist, bin_edges = np.histogram(predictions_as_is[:,0],bins=20)
    print('Flavour b predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()
    hist, bin_edges = np.histogram(predictions_as_is[:,1],bins=20)
    print('Flavour bb predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()
    hist, bin_edges = np.histogram(predictions_as_is[:,2],bins=20)
    print('Flavour c predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()
    hist, bin_edges = np.histogram(predictions_as_is[:,3],bins=20)
    print('Flavour udsg predictions: bin_edges and histogram')
    print(bin_edges)
    print(hist)
    del hist
    del bin_edges
    gc.collect()


    sys.exit()
    '''
    print('predictions without weighting done')

    mostprob = np.argmax(predictions_as_is, axis=-1)
    cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
    print(cfm)
    with open(f'/home/um106329/aisafety/may_21/evaluate/confusion_matrices/weighting_method_noweighting_default_{default}_{n_samples}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_minieval_{do_minimal_eval}.npy', 'wb') as f:
        np.save(f, cfm)


'''

    Predictions: With new weighting method
    
'''
if do_ptetaflavloss:
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



    checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/_ptetaflavloss_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted_ptetaflavloss_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    #evaluate network on inputs
    model.eval()
    predictions_new = model(test_inputs).detach().numpy()
    '''
    print(np.unique(predictions_new))
    predictions_new[:,0][predictions_new[:,0] > 1.] = 0.999999
    predictions_new[:,1][predictions_new[:,1] > 1.] = 0.999999
    predictions_new[:,2][predictions_new[:,2] > 1.] = 0.999999
    predictions_new[:,3][predictions_new[:,3] > 1.] = 0.999999
    predictions_new[:,0][predictions_new[:,0] < 0.] = 0.000001
    predictions_new[:,1][predictions_new[:,1] < 0.] = 0.000001
    predictions_new[:,2][predictions_new[:,2] < 0.] = 0.000001
    predictions_new[:,3][predictions_new[:,3] < 0.] = 0.000001
    print(np.unique(predictions_new))
    '''

    print('predictions with loss weighting done')


    #sys.exit()


    mostprob = np.argmax(predictions_new, axis=-1)
    cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
    print(cfm)
    with open(f'/home/um106329/aisafety/may_21/evaluate/confusion_matrices/weighting_method_ptetaflavloss_default_{default}_{n_samples}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_minieval_{do_minimal_eval}.npy', 'wb') as f:
        np.save(f, cfm)

'''

    Predictions: With new weighting method FLAT
    
'''
if do_flatptetaflavloss:
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



    checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/_flatptetaflavloss_{NUM_DATASETS}_{default}_{n_samples}/model_{at_epoch}_epochs_v10_GPU_weighted_flatptetaflavloss_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    #evaluate network on inputs
    model.eval()
    predictions_new_flat = model(test_inputs).detach().numpy()
    '''
    print(np.unique(predictions_new_flat))
    predictions_new_flat[:,0][predictions_new_flat[:,0] > 1.] = 0.999999
    predictions_new_flat[:,1][predictions_new_flat[:,1] > 1.] = 0.999999
    predictions_new_flat[:,2][predictions_new_flat[:,2] > 1.] = 0.999999
    predictions_new_flat[:,3][predictions_new_flat[:,3] > 1.] = 0.999999
    predictions_new_flat[:,0][predictions_new_flat[:,0] < 0.] = 0.000001
    predictions_new_flat[:,1][predictions_new_flat[:,1] < 0.] = 0.000001
    predictions_new_flat[:,2][predictions_new_flat[:,2] < 0.] = 0.000001
    predictions_new_flat[:,3][predictions_new_flat[:,3] < 0.] = 0.000001
    print(np.unique(predictions_new_flat))
    '''

    print('predictions with loss weighting done')


    #sys.exit()


    mostprob = np.argmax(predictions_new_flat, axis=-1)
    cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
    print(cfm)
    with open(f'/home/um106329/aisafety/may_21/evaluate/confusion_matrices/weighting_method_flatptetaflavloss_default_{default}_{n_samples}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_minieval_{do_minimal_eval}.npy', 'wb') as f:
        np.save(f, cfm)
        
        
def sum_hist():
    plt.ioff()    
    classifierHist = hist.Hist("Jets",
                            hist.Cat("sample","sample name"),
                            hist.Cat("flavour","flavour of the jet"),
                            hist.Bin("probb","P(b)",50,-0.05,1.05),
                            hist.Bin("probbb","P(bb)",50,-0.05,1.05),
                            hist.Bin("probc","P(c)",50,-0.05,1.05),
                            hist.Bin("probudsg","P(udsg)",50,-0.05,1.05),
                         )

   
    classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
    classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
    classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
    classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])
    
    if do_noweighting:
        classifierHist.fill(sample="No weighting",flavour='b-jets',probb=predictions_as_is[:,0][jetFlavour==1],probbb=predictions_as_is[:,1][jetFlavour==1],probc=predictions_as_is[:,2][jetFlavour==1],probudsg=predictions_as_is[:,3][jetFlavour==1])
        classifierHist.fill(sample="No weighting",flavour='bb-jets',probb=predictions_as_is[:,0][jetFlavour==2],probbb=predictions_as_is[:,1][jetFlavour==2],probc=predictions_as_is[:,2][jetFlavour==2],probudsg=predictions_as_is[:,3][jetFlavour==2])
        classifierHist.fill(sample="No weighting",flavour='c-jets',probb=predictions_as_is[:,0][jetFlavour==3],probbb=predictions_as_is[:,1][jetFlavour==3],probc=predictions_as_is[:,2][jetFlavour==3],probudsg=predictions_as_is[:,3][jetFlavour==3])
        classifierHist.fill(sample="No weighting",flavour='udsg-jets',probb=predictions_as_is[:,0][jetFlavour==4],probbb=predictions_as_is[:,1][jetFlavour==4],probc=predictions_as_is[:,2][jetFlavour==4],probudsg=predictions_as_is[:,3][jetFlavour==4])
    
    if do_ptetaflavloss:
        classifierHist.fill(sample="Loss weighting",flavour='b-jets',probb=predictions_new[:,0][jetFlavour==1],probbb=predictions_new[:,1][jetFlavour==1],probc=predictions_new[:,2][jetFlavour==1],probudsg=predictions_new[:,3][jetFlavour==1])
        classifierHist.fill(sample="Loss weighting",flavour='bb-jets',probb=predictions_new[:,0][jetFlavour==2],probbb=predictions_new[:,1][jetFlavour==2],probc=predictions_new[:,2][jetFlavour==2],probudsg=predictions_new[:,3][jetFlavour==2])
        classifierHist.fill(sample="Loss weighting",flavour='c-jets',probb=predictions_new[:,0][jetFlavour==3],probbb=predictions_new[:,1][jetFlavour==3],probc=predictions_new[:,2][jetFlavour==3],probudsg=predictions_new[:,3][jetFlavour==3])
        classifierHist.fill(sample="Loss weighting",flavour='udsg-jets',probb=predictions_new[:,0][jetFlavour==4],probbb=predictions_new[:,1][jetFlavour==4],probc=predictions_new[:,2][jetFlavour==4],probudsg=predictions_new[:,3][jetFlavour==4])
    if do_flatptetaflavloss:
        classifierHist.fill(sample="Loss weighting (flat)",flavour='b-jets',probb=predictions_new_flat[:,0][jetFlavour==1],probbb=predictions_new_flat[:,1][jetFlavour==1],probc=predictions_new_flat[:,2][jetFlavour==1],probudsg=predictions_new_flat[:,3][jetFlavour==1])
        classifierHist.fill(sample="Loss weighting (flat)",flavour='bb-jets',probb=predictions_new_flat[:,0][jetFlavour==2],probbb=predictions_new_flat[:,1][jetFlavour==2],probc=predictions_new_flat[:,2][jetFlavour==2],probudsg=predictions_new_flat[:,3][jetFlavour==2])
        classifierHist.fill(sample="Loss weighting (flat)",flavour='c-jets',probb=predictions_new_flat[:,0][jetFlavour==3],probbb=predictions_new_flat[:,1][jetFlavour==3],probc=predictions_new_flat[:,2][jetFlavour==3],probudsg=predictions_new_flat[:,3][jetFlavour==3])
        classifierHist.fill(sample="Loss weighting (flat)",flavour='udsg-jets',probb=predictions_new_flat[:,0][jetFlavour==4],probbb=predictions_new_flat[:,1][jetFlavour==4],probc=predictions_new_flat[:,2][jetFlavour==4],probudsg=predictions_new_flat[:,3][jetFlavour==4])


    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
    #plt.subplots_adjust(wspace=0.4)
    dcsv1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    dcsv2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    dcsv3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    dcsv4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    
    #max1, max2, max3, max4 = dcsv1.get_ylim()[1], dcsv2.get_ylim()[1], dcsv3.get_ylim()[1], dcsv4.get_ylim()[1]
    if do_noweighting:
        no_w1 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'blue','linewidth':3})
        no_w2 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'blue','linewidth':3})
        no_w3 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'blue','linewidth':3})
        no_w4 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'blue','linewidth':3})
    
    #max1, max2, max3, max4 = max(max1,no_w1.get_ylim()[1]), max(max2,no_w2.get_ylim()[1]), max(max3,no_w3.get_ylim()[1]), max(max4,no_w4.get_ylim()[1])
    if do_ptetaflavloss:
        loss_w1 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'green','linewidth':3})
        loss_w2 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'green','linewidth':3})
        loss_w3 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'green','linewidth':3})
        loss_w4 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'green','linewidth':3})
    if do_flatptetaflavloss:
        loss_flat_w1 = hist.plot1d(classifierHist['Loss weighting (flat)'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'green','linewidth':3})
        loss_flat_w2 = hist.plot1d(classifierHist['Loss weighting (flat)'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'green','linewidth':3})
        loss_flat_w3 = hist.plot1d(classifierHist['Loss weighting (flat)'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'green','linewidth':3})
        loss_flat_w4 = hist.plot1d(classifierHist['Loss weighting (flat)'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'green','linewidth':3})    
    
    
    ax2.legend(loc='upper right',title='Outputs',ncol=1)
    ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()
    
    # just leaving all those trials to get correct y-limits here, as what I did in the end finally worked with pure matplotlib functions...
    
    '''
    ax1_y_limit = max(max(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg').values()),
                  max(classifierHist['No weighting'].sum('flavour','probbb','probc','probudsg').values()),
                  max(classifierHist['Loss weighting'].sum('flavour','probbb','probc','probudsg').values())) 
    ax2_y_limit = max(max(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg').values()),
                  max(classifierHist['No weighting'].sum('flavour','probb','probc','probudsg').values()),
                  max(classifierHist['Loss weighting'].sum('flavour','probb','probc','probudsg').values())) 
    ax3_y_limit = max(max(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg').values()),
                  max(classifierHist['No weighting'].sum('flavour','probb','probbb','probudsg').values()),
                  max(classifierHist['Loss weighting'].sum('flavour','probb','probbb','probudsg').values())) 
    ax4_y_limit = max(max(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc').values()),
                  max(classifierHist['No weighting'].sum('flavour','probb','probbb','probc').values()),
                  max(classifierHist['Loss weighting'].sum('flavour','probb','probbb','probc').values())) 
    '''
    #ax1_y_limit, ax2_y_limit, ax3_y_limit, ax4_y_limit = max(max1,loss_w1.get_ylim()[1]), max(max2,loss_w2.get_ylim()[1]), max(max3,loss_w3.get_ylim()[1]), max(max4,loss_w4.get_ylim()[1])
    #print(ax1_y_limit)
    #print(np.max(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg').values().values(),classifierHist['No weighting'].sum('flavour','probbb','probc','probudsg').values().values(),classifierHist['Loss weighting'].sum('flavour','probbb','probc','probudsg').values().values()))
    #print(np.max(classifierHist['No weighting'].sum('flavour','probbb','probc','probudsg').values()['No weighting']))
    '''
    # recompute the ax.dataLim
    ax1.relim()
    ax2.relim()
    ax3.relim()
    ax4.relim()
    # update ax.viewLim using the new dataLim
    ax1.autoscale_view(True,True,True)
    ax2.autoscale_view(True,True,True)
    ax3.autoscale_view(True,True,True)
    ax4.autoscale_view(True,True,True)
    
    ax1.set_ymargin(0.1)
    ax2.set_ymargin(0.1)
    ax3.set_ymargin(0.1)
    ax4.set_ymargin(0.1)
    '''
    
    # to have the y-limit adapt to the maximum of EACH bin or overlay and scale automatically, without having to define the maximum myself
    
    ax1.set_ylim(bottom=0, auto=True)
    ax2.set_ylim(bottom=0, auto=True)
    ax3.set_ylim(bottom=0, auto=True)
    ax4.set_ylim(bottom=0, auto=True)
    
    ax1.autoscale(True)
    ax2.autoscale(True)
    ax3.autoscale(True)
    ax4.autoscale(True)
    
    # this is to make sure also for the smaller number of jets there will be scientific notation on the y-axis (this ensures the width of the subfigures together with labels will be the same
    # for both 49 and 10 files, so they can be compared, at least qualitatively and have the same aspect ratios etc.)
    
    ax1.ticklabel_format(scilimits=(-5,5))
    ax2.ticklabel_format(scilimits=(-5,5))
    ax3.ticklabel_format(scilimits=(-5,5))
    ax4.ticklabel_format(scilimits=(-5,5))
    
    fig.suptitle(f'Classifier and DeepCSV outputs\n After {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
    fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/discriminator_shapes/{weighting_method}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=300)
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)


    
def flavsplit_hist(wm):
    plt.ioff()
    if wm == '_noweighting':
        global predictions_as_is
        predictions = predictions_as_is
        del predictions_as_is
        gc.collect()
        method = 'No weighting applied'
    elif wm == '_ptetaflavloss':
        global predictions_new
        predictions = predictions_new
        del predictions_new
        gc.collect()
        method = 'Loss weighting'
    else:
        global predictions_new_flat
        predictions = predictions_new_flat
        del predictions_new_flat
        gc.collect()
        method = 'Loss weighting (flat)'    
    classifierHist = hist.Hist("Jets",
                            hist.Cat("sample","sample name"),
                            hist.Cat("flavour","flavour of the jet"),
                            hist.Bin("probb","P(b)",50,-0.05,1.05),
                            hist.Bin("probbb","P(bb)",50,-0.05,1.05),
                            hist.Bin("probc","P(c)",50,-0.05,1.05),
                            hist.Bin("probudsg","P(udsg)",50,-0.05,1.05),
                         )

    classifierHist.fill(sample="Classifier",flavour='b-jets',probb=predictions[:,0][jetFlavour==1],probbb=predictions[:,1][jetFlavour==1],probc=predictions[:,2][jetFlavour==1],probudsg=predictions[:,3][jetFlavour==1])
    classifierHist.fill(sample="Classifier",flavour='bb-jets',probb=predictions[:,0][jetFlavour==2],probbb=predictions[:,1][jetFlavour==2],probc=predictions[:,2][jetFlavour==2],probudsg=predictions[:,3][jetFlavour==2])
    classifierHist.fill(sample="Classifier",flavour='c-jets',probb=predictions[:,0][jetFlavour==3],probbb=predictions[:,1][jetFlavour==3],probc=predictions[:,2][jetFlavour==3],probudsg=predictions[:,3][jetFlavour==3])
    classifierHist.fill(sample="Classifier",flavour='udsg-jets',probb=predictions[:,0][jetFlavour==4],probbb=predictions[:,1][jetFlavour==4],probc=predictions[:,2][jetFlavour==4],probudsg=predictions[:,3][jetFlavour==4])
    classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
    classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
    classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
    classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])


    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
    #plt.subplots_adjust(wspace=0.4)
    hist.plot1d(classifierHist['Classifier'].sum('sample','probbb','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linewidth':3})
    hist.plot1d(classifierHist['Classifier'].sum('sample','probb','probc','probudsg'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':colorcode,'linewidth':3})
    hist.plot1d(classifierHist['Classifier'].sum('sample','probb','probbb','probudsg'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':colorcode,'linewidth':3})
    hist.plot1d(classifierHist['Classifier'].sum('sample','probb','probbb','probc'),overlay='flavour',ax=ax4,clear=False,line_opts={'color':colorcode,'linewidth':3})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'orange'})
    ax2.legend(loc='upper right',title='Outputs',ncol=1)
    ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()
    
    ax1.set_ylim(bottom=0, auto=True)
    ax2.set_ylim(bottom=0, auto=True)
    ax3.set_ylim(bottom=0, auto=True)
    ax4.set_ylim(bottom=0, auto=True)
    
    ax1.autoscale(True)
    ax2.autoscale(True)
    ax3.autoscale(True)
    ax4.autoscale(True)
    
    ax1.ticklabel_format(scilimits=(-5,5))
    ax2.ticklabel_format(scilimits=(-5,5))
    ax3.ticklabel_format(scilimits=(-5,5))
    ax4.ticklabel_format(scilimits=(-5,5))
    
    fig.suptitle(f'Classifier and DeepCSV outputs, {method}\n After {at_epoch} epochs, evaluated on {len_test} jets, default {default}')
    fig.savefig(f'/home/um106329/aisafety/may_21/evaluate/discriminator_shapes/weighting_method{wm}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files_{n_samples}_samples_minieval_{do_minimal_eval}.png', bbox_inches='tight', dpi=300)
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)
    
    
if weighting_method == '_compare':
    sum_hist()
    flavsplit_hist('_noweighting')
    flavsplit_hist('_ptetaflavloss')
    flavsplit_hist('_flatptetaflavloss')
else:
    sum_hist()
    flavsplit_hist(weighting_method)
