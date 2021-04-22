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

#oversampling = False  # deprecated, as WeightedRandomSampler will not be used

parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
parser.add_argument("wm", help="Weighting method: _as_is, _new or _both")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
args = parser.parse_args()

NUM_DATASETS = args.files
at_epoch = args.prevep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)

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
predictions_as_is = model(test_inputs).detach().numpy()
print('predictions without weighting done')


mostprob = np.argmax(predictions_as_is, axis=-1)
cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
print(cfm)
with open(f'/home/um106329/aisafety/april_21/eval_attack/plots/confusion_matrix_weighting_method_as_is_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.npy', 'wb') as f:
    np.save(f, cfm)


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
predictions_new = model(test_inputs).detach().numpy()
print('predictions with loss weighting done')

mostprob = np.argmax(predictions_new, axis=-1)
cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
print(cfm)
with open(f'/home/um106329/aisafety/april_21/eval_attack/plots/confusion_matrix_weighting_method_new_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.npy', 'wb') as f:
    np.save(f, cfm)


def compare_hist():
    plt.ioff()    
    classifierHist = hist.Hist("Jets",
                            hist.Cat("sample","sample name"),
                            hist.Cat("flavour","flavour of the jet"),
                            hist.Bin("probb","P(b)",50,0,1),
                            hist.Bin("probbb","P(bb)",50,0,1),
                            hist.Bin("probc","P(c)",50,0,1),
                            hist.Bin("probudsg","P(udsg)",50,0,1),
                         )

   
    classifierHist.fill(sample="DeepCSV",flavour='b-jets',probb=DeepCSV_testset[:,0][jetFlavour==1],probbb=DeepCSV_testset[:,1][jetFlavour==1],probc=DeepCSV_testset[:,2][jetFlavour==1],probudsg=DeepCSV_testset[:,3][jetFlavour==1])
    classifierHist.fill(sample="DeepCSV",flavour='bb-jets',probb=DeepCSV_testset[:,0][jetFlavour==2],probbb=DeepCSV_testset[:,1][jetFlavour==2],probc=DeepCSV_testset[:,2][jetFlavour==2],probudsg=DeepCSV_testset[:,3][jetFlavour==2])
    classifierHist.fill(sample="DeepCSV",flavour='c-jets',probb=DeepCSV_testset[:,0][jetFlavour==3],probbb=DeepCSV_testset[:,1][jetFlavour==3],probc=DeepCSV_testset[:,2][jetFlavour==3],probudsg=DeepCSV_testset[:,3][jetFlavour==3])
    classifierHist.fill(sample="DeepCSV",flavour='udsg-jets',probb=DeepCSV_testset[:,0][jetFlavour==4],probbb=DeepCSV_testset[:,1][jetFlavour==4],probc=DeepCSV_testset[:,2][jetFlavour==4],probudsg=DeepCSV_testset[:,3][jetFlavour==4])
    
    classifierHist.fill(sample="No weighting",flavour='b-jets',probb=predictions_as_is[:,0][jetFlavour==1],probbb=predictions_as_is[:,1][jetFlavour==1],probc=predictions_as_is[:,2][jetFlavour==1],probudsg=predictions_as_is[:,3][jetFlavour==1])
    classifierHist.fill(sample="No weighting",flavour='bb-jets',probb=predictions_as_is[:,0][jetFlavour==2],probbb=predictions_as_is[:,1][jetFlavour==2],probc=predictions_as_is[:,2][jetFlavour==2],probudsg=predictions_as_is[:,3][jetFlavour==2])
    classifierHist.fill(sample="No weighting",flavour='c-jets',probb=predictions_as_is[:,0][jetFlavour==3],probbb=predictions_as_is[:,1][jetFlavour==3],probc=predictions_as_is[:,2][jetFlavour==3],probudsg=predictions_as_is[:,3][jetFlavour==3])
    classifierHist.fill(sample="No weighting",flavour='udsg-jets',probb=predictions_as_is[:,0][jetFlavour==4],probbb=predictions_as_is[:,1][jetFlavour==4],probc=predictions_as_is[:,2][jetFlavour==4],probudsg=predictions_as_is[:,3][jetFlavour==4])
    
    
    classifierHist.fill(sample="Loss weighting",flavour='b-jets',probb=predictions_new[:,0][jetFlavour==1],probbb=predictions_new[:,1][jetFlavour==1],probc=predictions_new[:,2][jetFlavour==1],probudsg=predictions_new[:,3][jetFlavour==1])
    classifierHist.fill(sample="Loss weighting",flavour='bb-jets',probb=predictions_new[:,0][jetFlavour==2],probbb=predictions_new[:,1][jetFlavour==2],probc=predictions_new[:,2][jetFlavour==2],probudsg=predictions_new[:,3][jetFlavour==2])
    classifierHist.fill(sample="Loss weighting",flavour='c-jets',probb=predictions_new[:,0][jetFlavour==3],probbb=predictions_new[:,1][jetFlavour==3],probc=predictions_new[:,2][jetFlavour==3],probudsg=predictions_new[:,3][jetFlavour==3])
    classifierHist.fill(sample="Loss weighting",flavour='udsg-jets',probb=predictions_new[:,0][jetFlavour==4],probbb=predictions_new[:,1][jetFlavour==4],probc=predictions_new[:,2][jetFlavour==4],probudsg=predictions_new[:,3][jetFlavour==4])


    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
    #plt.subplots_adjust(wspace=0.4)
    dcsv1 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    dcsv2 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    dcsv3 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    dcsv4 = hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    
    #max1, max2, max3, max4 = dcsv1.get_ylim()[1], dcsv2.get_ylim()[1], dcsv3.get_ylim()[1], dcsv4.get_ylim()[1]
    
    no_w1 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'blue','linewidth':3})
    no_w2 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'blue','linewidth':3})
    no_w3 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'blue','linewidth':3})
    no_w4 = hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'blue','linewidth':3})
    
    #max1, max2, max3, max4 = max(max1,no_w1.get_ylim()[1]), max(max2,no_w2.get_ylim()[1]), max(max3,no_w3.get_ylim()[1]), max(max4,no_w4.get_ylim()[1])
    
    loss_w1 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'green','linewidth':3})
    loss_w2 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'green','linewidth':3})
    loss_w3 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'green','linewidth':3})
    loss_w4 = hist.plot1d(classifierHist['Loss weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'green','linewidth':3})
    
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
    fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/compare_discriminator_shapes_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=300)
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)


    
def do_hist(wm):
    plt.ioff()
    if wm == '_as_is':
        global predictions_as_is
        predictions = predictions_as_is
        del predictions_as_is
        gc.collect()
        method = 'No weighting applied'
    else:
        global predictions_new
        predictions = predictions_new
        del predictions_new
        gc.collect()
        method = 'Loss weighting'
        
    classifierHist = hist.Hist("Jets",
                            hist.Cat("sample","sample name"),
                            hist.Cat("flavour","flavour of the jet"),
                            hist.Bin("probb","P(b)",50,0,1),
                            hist.Bin("probbb","P(bb)",50,0,1),
                            hist.Bin("probc","P(c)",50,0,1),
                            hist.Bin("probudsg","P(udsg)",50,0,1),
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
    fig.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/discriminator_shapes_weighting_method{wm}_default_{default}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_files.png', bbox_inches='tight', dpi=300)
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)
    
    
if weighting_method == '_both':
    compare_hist()
    do_hist('_as_is')
    do_hist('_new')
else:
    do_hist(weighting_method)
