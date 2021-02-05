import uproot4 as uproot
import numpy as np
import awkward1 as ak

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
colorcode = ['firebrick','magenta','cyan','darkgreen']



at_epoch = 1

NUM_DATASETS = 200

print(f'Evaluate training at epoch {at_epoch}')




'''

    Load inputs and targets
    
'''
test_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]


test_inputs = torch.cat(tuple(torch.load(ti) for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len_test)

test_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/test_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]
DeepCSV_testset_file_paths = ['/work/um106329/MA/cleaned/preprocessed/DeepCSV_testset_%d.pt' % k for k in range(0,NUM_DATASETS)]

test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths)).float()
print('test targets done')
DeepCSV_testset = np.concatenate([torch.load(ti) for ti in DeepCSV_testset_file_paths])
print('DeepCSV test done')

jetFlavour = test_targets+1



'''

    Predictions: Without weighting
    
'''
criterion = nn.CrossEntropyLoss()



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



checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted_as_is.pt' % NUM_DATASETS, map_location=torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)




#evaluate network on inputs
model.eval()
predictions_as_is = model(test_inputs).detach().numpy()
print('predictions without weighting done')




'''

    Predictions: With first weighting method
    
'''

# as calculated in dataset_info.ipynb
allweights = [0.9393934969162745, 0.9709644530642717, 0.8684253665882813, 0.2212166834311725]
class_weights = torch.FloatTensor(allweights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)



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



checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted.pt' % NUM_DATASETS, map_location=torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)




#evaluate network on inputs
model.eval()
predictions_old = model(test_inputs).detach().numpy()
print('predictions with first weighting method done')




'''

    Predictions: With new weighting method
    
'''

# as calculated in dataset_info.ipynb
allweights = [0.27580367992004956, 0.5756907770526237, 0.1270419391956182, 0.021463603831708488]
class_weights = torch.FloatTensor(allweights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)



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



checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{at_epoch}_epochs_v13_GPU_weighted_new.pt' % NUM_DATASETS, map_location=torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)




#evaluate network on inputs
model.eval()
predictions_new = model(test_inputs).detach().numpy()
print('predictions with new weighting method done')


def compare_hist():
        
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
    
    classifierHist.fill(sample="1 - rel. freq. weighting",flavour='b-jets',probb=predictions_old[:,0][jetFlavour==1],probbb=predictions_old[:,1][jetFlavour==1],probc=predictions_old[:,2][jetFlavour==1],probudsg=predictions_old[:,3][jetFlavour==1])
    classifierHist.fill(sample="1 - rel. freq. weighting",flavour='bb-jets',probb=predictions_old[:,0][jetFlavour==2],probbb=predictions_old[:,1][jetFlavour==2],probc=predictions_old[:,2][jetFlavour==2],probudsg=predictions_old[:,3][jetFlavour==2])
    classifierHist.fill(sample="1 - rel. freq. weighting",flavour='c-jets',probb=predictions_old[:,0][jetFlavour==3],probbb=predictions_old[:,1][jetFlavour==3],probc=predictions_old[:,2][jetFlavour==3],probudsg=predictions_old[:,3][jetFlavour==3])
    classifierHist.fill(sample="1 - rel. freq. weighting",flavour='udsg-jets',probb=predictions_old[:,0][jetFlavour==4],probbb=predictions_old[:,1][jetFlavour==4],probc=predictions_old[:,2][jetFlavour==4],probudsg=predictions_old[:,3][jetFlavour==4])
    
    classifierHist.fill(sample="1 / rel. freq. weighting",flavour='b-jets',probb=predictions_new[:,0][jetFlavour==1],probbb=predictions_new[:,1][jetFlavour==1],probc=predictions_new[:,2][jetFlavour==1],probudsg=predictions_new[:,3][jetFlavour==1])
    classifierHist.fill(sample="1 / rel. freq. weighting",flavour='bb-jets',probb=predictions_new[:,0][jetFlavour==2],probbb=predictions_new[:,1][jetFlavour==2],probc=predictions_new[:,2][jetFlavour==2],probudsg=predictions_new[:,3][jetFlavour==2])
    classifierHist.fill(sample="1 / rel. freq. weighting",flavour='c-jets',probb=predictions_new[:,0][jetFlavour==3],probbb=predictions_new[:,1][jetFlavour==3],probc=predictions_new[:,2][jetFlavour==3],probudsg=predictions_new[:,3][jetFlavour==3])
    classifierHist.fill(sample="1 / rel. freq. weighting",flavour='udsg-jets',probb=predictions_new[:,0][jetFlavour==4],probbb=predictions_new[:,1][jetFlavour==4],probc=predictions_new[:,2][jetFlavour==4],probudsg=predictions_new[:,3][jetFlavour==4])



    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    hist.plot1d(classifierHist['DeepCSV'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,fill_opts={'alpha':.7,'facecolor':'red'})
    
    hist.plot1d(classifierHist['No weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'blue','linewidth':3})
    hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'blue','linewidth':3})
    hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'blue','linewidth':3})
    hist.plot1d(classifierHist['No weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'blue','linewidth':3})
    
    hist.plot1d(classifierHist['1 - rel. freq. weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'orange','linewidth':3})
    hist.plot1d(classifierHist['1 - rel. freq. weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'orange','linewidth':3})
    hist.plot1d(classifierHist['1 - rel. freq. weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'orange','linewidth':3})
    hist.plot1d(classifierHist['1 - rel. freq. weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'orange','linewidth':3})
    
    hist.plot1d(classifierHist['1 / rel. freq. weighting'].sum('flavour','probbb','probc','probudsg'),ax=ax1,clear=False,line_opts={'color':'green','linewidth':3})
    hist.plot1d(classifierHist['1 / rel. freq. weighting'].sum('flavour','probb','probc','probudsg'),ax=ax2,clear=False,line_opts={'color':'green','linewidth':3})
    hist.plot1d(classifierHist['1 / rel. freq. weighting'].sum('flavour','probb','probbb','probudsg'),ax=ax3,clear=False,line_opts={'color':'green','linewidth':3})
    hist.plot1d(classifierHist['1 / rel. freq. weighting'].sum('flavour','probb','probbb','probc'),ax=ax4,clear=False,line_opts={'color':'green','linewidth':3})
    
    ax2.legend(loc='upper right',title='Outputs',ncol=1)
    ax1.get_legend().remove(), ax3.get_legend().remove(), ax4.get_legend().remove()
    #ax2.set_ylim(0, 2.25e7)
    fig.suptitle(f'Classifier and DeepCSV outputs\n After {at_epoch} epochs, evaluated on {len_test} jets')
    fig.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/compare_hist_all.png', bbox_inches='tight', dpi=300)
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)


    
def do_hist(weighting = 1):
    
    if weighting == 0:
        predictions = predictions_as_is
        method = 'No weighting applied'
    elif weighting == 1:
        predictions = predictions_old
        method = '1 - rel. freq. weighting'
    else:
        predictions = predictions_new
        method = '1 / rel. freq. weighting'
        
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
    ax2.set_ylim(0, 2.25e7)
    fig.suptitle(f'Classifier and DeepCSV outputs, {method}\n After {at_epoch} epochs, evaluated on {len_test} jets')
    fig.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/compare_hist_{weighting}.png', bbox_inches='tight', dpi=300)
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)
    
    
do_hist(0)
do_hist(1)
do_hist(2)
compare_hist()


'''
#plot some ROC curves
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,17],num=4)
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_as_is[:,0])
ax1.plot(fpr,tpr)
print(f"auc for b-tagging without weighting: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_old[:,0])
ax1.plot(fpr,tpr)
print(f"auc for b-tagging with first weighting method: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],predictions_new[:,0])
ax1.plot(fpr,tpr)
print(f"auc for b-tagging with new weighting method: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==0 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,0])
ax1.plot(fpr,tpr)
ax1.legend(['Classifier: Without Weighting', 'Classifier: 1-rel.freq. weighting', 'Classifier: 1/rel.freq. weighting','DeepCSV'])
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
ax1.set_title('b tagging')
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_as_is[:,1])
ax2.plot(fpr,tpr)
print(f"auc for bb-tagging without weighting: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_old[:,1])
ax2.plot(fpr,tpr)
print(f"auc for bb-tagging with first weighting method: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],predictions_new[:,1])
ax2.plot(fpr,tpr)
print(f"auc for bb-tagging with new weighting method: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==1 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,1])
ax2.plot(fpr,tpr)
ax2.legend(['Classifier: Without Weighting', 'Classifier: 1-rel.freq. weighting', 'Classifier: 1/rel.freq. weighting','DeepCSV'])
ax2.set_xlabel('false positive rate')
ax2.set_ylabel('true positive rate')
ax2.set_title('bb tagging')
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_as_is[:,2])
ax3.plot(fpr,tpr)
print(f"auc for c-tagging as is: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_old[:,2])
ax3.plot(fpr,tpr)
print(f"auc for c-tagging old: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],predictions_new[:,2])
ax3.plot(fpr,tpr)
print(f"auc for c-tagging new: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==2 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,2])
ax3.plot(fpr,tpr)
ax3.legend(['Classifier: Without Weighting', 'Classifier: 1-rel.freq. weighting', 'Classifier: 1/rel.freq. weighting','DeepCSV'])
ax3.set_xlabel('false positive rate')
ax3.set_ylabel('true positive rate')
ax3.set_title('c tagging')
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_as_is[:,3])
ax4.plot(fpr,tpr)
print(f"auc for udsg-tagging as is: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_old[:,3])
ax4.plot(fpr,tpr)
print(f"auc for udsg-tagging old: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],predictions_new[:,3])
ax4.plot(fpr,tpr)
print(f"auc for udsg-tagging new: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if test_targets[i]==3 else 0) for i in range(len(test_targets))],DeepCSV_testset[:,3])
ax4.plot(fpr,tpr)
ax4.legend(['Classifier: Without Weighting', 'Classifier: 1-rel.freq. weighting', 'Classifier: 1/rel.freq. weighting','DeepCSV'])
ax4.set_xlabel('false positive rate')
ax4.set_ylabel('true positive rate')
ax4.set_title('udsg- tagging')
ax1.get_legend().remove(), ax2.get_legend().remove(), ax3.get_legend().remove()
ax4.legend(['Classifier: Without Weighting', 'Classifier: 1-rel.freq. weighting', 'Classifier: 1/rel.freq. weighting','DeepCSV'],loc='lower right')
fig.suptitle(f'ROCs for b, bb, c and light jets\n After {at_epoch} epochs, evaluated on {len_test} jets')
fig.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/compare_roc.png', bbox_inches='tight', dpi=300)
gc.collect(2)



BvsUDSG_inputs = torch.cat((test_inputs[jetFlavour==1],test_inputs[jetFlavour==4]))
del gc.garbage[:]
del test_inputs
BvsUDSG_targets = torch.cat((test_targets[jetFlavour==1],test_targets[jetFlavour==4]))
del test_targets
BvsUDSG_predictions_as_is = np.concatenate((predictions_as_is[jetFlavour==1],predictions_as_is[jetFlavour==4]))
del predictions_as_is
BvsUDSG_predictions = np.concatenate((predictions_old[jetFlavour==1],predictions_old[jetFlavour==4]))
del predictions
BvsUDSG_predictions_new = np.concatenate((predictions_new[jetFlavour==1],predictions_new[jetFlavour==4]))
del predictions_new
BvsUDSG_DeepCSV = np.concatenate((DeepCSV_testset[jetFlavour==1],DeepCSV_testset[jetFlavour==4]))
del DeepCSV_testset
gc.collect()



# plot ROC BvsUDSG

fig = plt.figure(figsize=[15,15],num=40)
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[:,0])
plt.plot(fpr,tpr)
print(f"auc for B vs UDSG as is: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions[:,0])
plt.plot(fpr,tpr)
print(f"auc for B vs UDSG: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[:,0])
plt.plot(fpr,tpr)
print(f"auc for B vs UDSG new: {metrics.auc(fpr,tpr)}")
fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0])
plt.plot(fpr,tpr)

plt.xlabel('mistag rate')
plt.ylabel('efficiency')
plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets')

plt.xlim(-0.05,1.05)
plt.ylim(-0.05,1.05)
#plt.plot([0,0.1,0.1,0,0],[0.4,0.4,0.9,0.9,0.4],'--',color='black')
#plt.plot([0.1,1.01],[0.9,0.735],'--',color='grey')
#plt.plot([0,0.3],[0.4,0.0],'--',color='grey')
#ax = plt.axes([.37, .16, .5, .5])

#fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_as_is[:,0])
#plt.plot(fpr,tpr)
#fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions[:,0])
#plt.plot(fpr,tpr)
#fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_predictions_new[:,0])
#plt.plot(fpr,tpr)
#fpr,tpr,thresholds = metrics.roc_curve([(1 if BvsUDSG_targets[i]==0 else 0) for i in range(len(BvsUDSG_targets))],BvsUDSG_DeepCSV[:,0])
#plt.plot(fpr,tpr)

plt.legend(['Classifier: Without Weighting', 'Classifier: 1-rel.freq. weighting', 'Classifier: 1/rel.freq. weighting','DeepCSV'],loc='lower right')
#plt.xlim(0,0.1)
#plt.ylim(0.4,0.9)
fig.savefig(f'/home/um106329/aisafety/models/weighted/compare/after_{at_epoch}/compare_roc_BvsUDSG.png', bbox_inches='tight', dpi=300)
'''
