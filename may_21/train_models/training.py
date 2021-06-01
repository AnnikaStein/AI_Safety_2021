#import uproot4 as uproot
import numpy as np
#import awkward1 as ak

#from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt
#import mplhep as hep

#import coffea.hist as hist

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset, WeightedRandomSampler, DataLoader


import time
import random
import gc

import argparse
#import ast

import os

from jet_reweighting import FlavEtaPtSampler
#from jet_reweighting import FlavEtaPtDataLoader

#plt.style.use(hep.cms.style.ROOT)

# depending on what's available, or force cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

#C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
#colorcode = ['firebrick','magenta','cyan','darkgreen']


start = time.time()



parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
parser.add_argument("addep", type=int, help="Number of additional epochs for this training")
parser.add_argument("wm", help="Weighting method")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("dominimal", help="Only do training with minimal setup, i.e. 15 QCD, 5 TT files")
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
epochs = args.addep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
#minima = np.load('/home/um106329/aisafety/april_21/from_Nik/default_value_studies_minima.npy')  # this should only be needed when applying the attack
#defaults = minima - default
#defaults = -999*np.ones(100)
n_samples = args.jets
do_minimal = args.dominimal

'''
    Available weighting methods:
        '_noweighting' :  apply no weighting factors at all (will converge to a nice result performance-wise, but the discriminator shapes are basically just two bins (0 or 1, almost nothing in between))
        '_ptetaflavsampler' : should be uniform in flavour, and reweighted in pt & eta such that shapes per flavour are equal (absolute) -- uses over- and undersampling
        '_ptetaflavloss' : should be uniform in flavour, and reweighted in pt & eta such that shapes per flavour are equal (absolute) -- uses sample weights that will be multiplied with the loss
'''

print(f'weighting method: {weighting_method}')   

# for the initial setup, reduce sources of randomness (e.g. if different methods will be used, they should start with the same initial weights), but later, using deterministic algs etc. would just slow things down without providing any benefit
if prev_epochs == 0:
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    parent_dir_1 = '/home/um106329/aisafety/may_21/train_models/saved_models'
    directory_1 = f'{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}'
    path_1 = os.path.join(parent_dir_1, directory_1)
    if not os.path.exists(path_1):
        os.mkdir(path_1)
    parent_dir_2 = '/hpcwork/um106329/may_21/saved_models'
    directory_2 = f'{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}'
    path_2 = os.path.join(parent_dir_2, directory_2)
    if not os.path.exists(path_2):
        os.mkdir(path_2)
        

# Parameters for the training and validation    
#bsize = 1000000     # this might seem large, but for comparison: bsize of 250000 for 86M training inputs
bsize = 1000000 
lrate = 0.0001     # initial learning rate, only for first epoch


print(f'starting to train the model after {prev_epochs} epochs that were already done')
print(f'learning rate for this script: {lrate}')
print(f'batch size for this script: {bsize}')
    



with open(f"/home/um106329/aisafety/may_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_default_{default}_{n_samples}_jets.txt", "a") as log:
    if NUM_DATASETS > 229 or do_minimal == 'yes':
        log.write(f"Setup: QCD and TT to Semileptonic samples with default value {default}, weighting method {weighting_method}, so far {prev_epochs} epochs done. Use lr={lrate} and bsize={bsize}. {n_samples} jets (-1 stands for all jets).\n")
    else:
        log.write(f"Setup: QCD samples with default value {default}, weighting method {weighting_method}, so far {prev_epochs} epochs done. Use lr={lrate} and bsize={bsize}. {n_samples} jets (-1 stands for all jets).\n")

        
        
# number of datasets is now the total number (QCD and TT combined), e.g. NUM_DATASETS = 5 --> take 0,1,2,3,4 of QCD (because min(5,229)=5) and take nothing from TT (because max(5-229,0)=0)
# other example: 229 --> 0-228 from QCD, and again max(0,0)=0 from TT
# only when NUM_DATASETS > 229 --> take all from QCD (229) and take those that remain from TT, e.g. 230 would mean take max(230-229,0)=1 TT dataset, this is the one with index 0
# maximal possible: 278 (=229+49) datasets in total
if do_minimal == 'no':
    train_input_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_inputs_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/train_inputs_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    train_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_targets_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/train_targets_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    val_input_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/val_inputs_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/val_inputs_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    val_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/val_targets_%d_with_default_{default}.pt' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/val_targets_%d_with_default_{default}.pt' % k for k in range(0,max(NUM_DATASETS-229,0))]
    
    if weighting_method == '_ptetaflavloss':
        train_sample_weights_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_sample_weights_%d_with_default_{default}.npy' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/train_sample_weights_%d_with_default_{default}.npy' % k for k in range(0,max(NUM_DATASETS-229,0))]
    elif weighting_method == '_ptetaflavsampler':
        train_bins_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_pt_eta_bins_%d_with_default_{default}.npy' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/train_pt_eta_bins_%d_with_default_{default}.npy' % k for k in range(0,max(NUM_DATASETS-229,0))]
#val_bins_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/val_pt_eta_bins_%d_with_default_{default}.npy' % k for k in range(0,min(NUM_DATASETS,229))] + [f'/hpcwork/um106329/may_21/scaled_TT/val_pt_eta_bins_%d_with_default_{default}.npy' % k for k in range(0,max(NUM_DATASETS-229,0))]
# after the paths have been defined, no need to keep track of qcd and tt paths anymore

# to imitate the performance of the model on new, unseen jets, the weighting is not used for the validation set; there the class imbalance and different distributions for the flavours in pt and eta are
# just kept as is, just like for future applications when no truth information about the flavour is available

# there were 229 QCD and 49 TT files, distribution of jets is 709686958 to 224235788 jets in total (take 0.8 x 0.9 of these for training), a combination of the first 15 QCD and 5 TT samples yields
# approximately the same distribution of jets, and can therefore be used to perform a faster training on less jets with similar behavior (total: train,val,tes = 760900950)

if do_minimal == 'yes':
    train_input_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_inputs_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/train_inputs_%d_with_default_{default}.pt' % k for k in range(0,5)]
    train_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_targets_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/train_targets_%d_with_default_{default}.pt' % k for k in range(0,5)]
    val_input_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/val_inputs_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/val_inputs_%d_with_default_{default}.pt' % k for k in range(0,5)]
    val_target_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/val_targets_%d_with_default_{default}.pt' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/val_targets_%d_with_default_{default}.pt' % k for k in range(0,5)] 
    
    if weighting_method == '_ptetaflavloss':
        train_sample_weights_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_sample_weights_%d_with_default_{default}.npy' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/train_sample_weights_%d_with_default_{default}.npy' % k for k in range(0,5)]
    elif weighting_method == '_ptetaflavsampler':
        train_bins_file_paths = [f'/hpcwork/um106329/may_21/scaled_QCD/train_pt_eta_bins_%d_with_default_{default}.npy' % k for k in range(0,15)] + [f'/hpcwork/um106329/may_21/scaled_TT/train_pt_eta_bins_%d_with_default_{default}.npy' % k for k in range(0,5)]

##### LOAD TRAINING SAMPLES #####

pre = time.time()
if weighting_method == '_ptetaflavloss':
    # if the loss shall be multiplied with sample weights after the calculation, one needs to add these as an additional column to the dataset inputs (otherwise the indices would not match up when using the dataloader)
    # adapted from https://stackoverflow.com/a/66375624/14558181
    allin = ConcatDataset([TensorDataset(torch.cat((torch.load(train_input_file_paths[f]), torch.from_numpy(np.load(train_sample_weights_file_paths[f])).to(torch.float16).unsqueeze(1)), dim=1) , torch.load(train_target_file_paths[f])) for f in range(NUM_DATASETS)])
else:
    allin = ConcatDataset([TensorDataset(torch.load(train_input_file_paths[f]), torch.load(train_target_file_paths[f])) for f in range(NUM_DATASETS)])
#allin = TensorDataset(train_inputs, train_targets)



post = time.time()
print(f"Time to load train: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")



##### LOAD VAL SAMPLES #####

pre = time.time()

allval = ConcatDataset([TensorDataset(torch.load(val_input_file_paths[f]), torch.load(val_target_file_paths[f])) for f in range(NUM_DATASETS)])
#allval = TensorDataset(val_inputs, val_targets)


post = time.time()
print(f"Time to load val: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")


##### LOAD TRAIN & VAL BINS #####

if weighting_method == '_ptetaflavsampler':
    train_bins = np.hstack([np.load(train_bin_path) for train_bin_path in train_bins_file_paths])
    #val_bins = np.hstack([np.load(val_bin_path) for val_bin_path in val_bins_file_paths])


pre = time.time()
'''
# deprecated, use FlavEtaPtSampler in the future
if weighting_method == '_wrs':
    #weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    #samples_weights = weights[torch.concat([torch.load(train_target_file_paths[f]) for f in range(NUM_DATASETS)])]
    ts = torch.cat([torch.load(train_target_file_paths[f]) for f in range(NUM_DATASETS)]).numpy()
    #ts = train_targets.numpy()
    class_weights = compute_class_weight(
           'balanced',
            classes=np.array([0,1,2,3]), 
            y=ts)
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(allin),
        replacement=True)
    trainloader = torch.utils.data.DataLoader(allin, batch_size=bsize, sampler=sampler, num_workers=0)
    post = time.time()
    print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(allval),
        replacement=True)
    valloader = torch.utils.data.DataLoader(allval, batch_size=bsize, sampler=sampler, num_workers=0)
    post = time.time()
    print(f"Time to create valloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
    
else:
    trainloader = torch.utils.data.DataLoader(allin, batch_size=bsize, shuffle=True, num_workers=0)
    post = time.time()
    print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
    
    valloader = torch.utils.data.DataLoader(allval, batch_size=bsize, shuffle=False, num_workers=0)
    post = time.time()
    print(f"Time to create valloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
'''
draw_n = None if n_samples == -1 else n_samples
if weighting_method == '_ptetaflavsampler':
    alltargets = torch.cat([torch.load(t) for t in train_target_file_paths])
    trainloader = DataLoader(allin, batch_size=bsize, sampler=FlavEtaPtSampler(alltargets,train_bins,num_samples=draw_n),num_workers=0, pin_memory=False)
    post = time.time()
    print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
#if weighting_method == '_ptetaflavloss':
#    trainloader = DataLoader(allin, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=False)
#    post = time.time()
#    print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
else:
    trainloader = DataLoader(allin, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=False)
    post = time.time()
    print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
    
valloader = DataLoader(allval, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=False)
postval = time.time()
print(f"Time to create valloader: {np.floor((postval-post)/60)} min {np.ceil((postval-post)%60)} s")


total_len_train = len(trainloader)
total_n_train = len(trainloader.dataset)
print(total_n_train,'\ttraining samples')



total_len_val = len(valloader)
total_n_val = len(valloader.dataset)
print(total_n_val,'\tvalidation samples')



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


if prev_epochs > 0:
    checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{prev_epochs}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt')
    model.load_state_dict(checkpoint["model_state_dict"])

print(model)

'''
# In principle, one can use multiple gpu devices with DataParallel. However, in this particular case, the cons are that the data transfer takes longer, and loss / weights will be averaged
# when both results are merged. The model might be too simple to profit from this method. First tests took longer than with just one device, and the loss did not go down as fast.
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
'''
model.to(device)

'''
# deprecated
# Choose the parameter for weighting in the loss function, according to the choice above
if weighting_method == '':
    # as calculated in dataset_info.ipynb
    allweights = [0.7158108642980718, 0.9962305696752469, 0.9099623138559123, 0.37799625217076893]   # will not be used from now on, was 1 - rel. freq. weighting, deprecated!!
    class_weights = torch.FloatTensor(allweights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
elif weighting_method == '_new':
    train_targets = torch.cat(tuple(torch.load(vi) for vi in train_target_file_paths))
    allweights = compute_class_weight(
           'balanced',
            classes=np.array([0,1,2,3]), 
            y=train_targets.numpy())
    class_weights = torch.FloatTensor(allweights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()
'''    

if weighting_method == '_ptetaflavloss':
    # loss weighting happens after loss for each sample has been calculated (see later in main train loop), will multiply with sample weights and only afterwards calculate the mean, therefore reduction has to be set to 'none' (performs no dim. red. and gives tensor of length n_batch)
    criterion = nn.CrossEntropyLoss(reduction='none')
else:
    # with this weighting, loss weighting is not necessary anymore (the imbalanced classes are already handled with weights for the cusotm sampler)
    criterion = nn.CrossEntropyLoss()
    
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)


if prev_epochs > 0:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    '''
    # deprecated, learning rate now controlled below for each epoch
    # update the learning rate to the new one
    for g in optimizer.param_groups:
        print('lr: ', g['lr'], 'prev run')
        g['lr'] = lrate
        print('lr: ', g['lr'], 'after update')
    '''
else:
    print(f'Learning rate (initial): {lrate}')

def new_learning_rate(ep):
    for g in optimizer.param_groups:
        #print('lr: ', g['lr'], 'prev epoch')
        # not recursive, just iterative calculation based on the initial learning rate stored in the variable lrate and the currect epoch ep
        g['lr'] = lrate/(1+ep/30)                      # decaying learning rate (the larger the number, e.g. 50, the slower the decay, think: after x epochs, the learning rate has been halved)
        #g['lr'] = 0.00001                             # change lr (to a new constant)
        print('lr: ', g['lr'], 'after update')
        
#The training algorithm

tic = time.time()
loss_history, val_loss_history = [], []
stale_epochs, min_loss = 0, 10
max_stale_epochs = 100

# epochs to be trained with the current script (on top of the prev_epochs)
#epochs = 1    # this is now controlled by the parser above
times = []


with open(f"/home/um106329/aisafety/may_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_default_{default}_{n_samples}_jets.txt", "a") as log:
    log.write(f"{np.floor((tic-start)/60)} min {np.ceil((tic-start)%60)} s"+' Everything prepared for main training loop.\n')




for e in range(epochs):
    times.append(time.time())
    if prev_epochs+e >= 1:  # this is to keep track of the total number of epochs, if the training is started again multiple times after some epochs that were already done
        new_learning_rate(prev_epochs+e)  # and if it's not the first epoch, decrease the learning rate a tiny bit (see function above), leading to a (hopefully) better convergence, that takes large steps at the beginning, and small ones close to the end
    running_loss = 0
    model.train()
    for b, (i,j) in enumerate(trainloader):
        if weighting_method == '_ptetaflavloss':
            sample_weights = i[:, -1].to(device, non_blocking=True)
            i = i[:, :-1]
        if e == 0 and b == 1:
            tb1 = time.time()
            print('first batch done')
            print(f"Time for first batch: {np.floor((tb1-times[0])/60)} min {((tb1-times[0])%60)} s")
            
            with open(f"/home/um106329/aisafety/may_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_default_{default}_{n_samples}_jets.txt", "a") as log:
                log.write(f"{np.floor((tb1-start)/60)} min {np.ceil((tb1-start)%60)} s"+' First batch done!\n')
            
        i = i.to(device, non_blocking=True)
        j = j.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(i.float())
        loss = criterion(output, j)         
        del i
        del j
        gc.collect()
        if weighting_method == '_ptetaflavloss':
            # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/4
            loss = (loss * sample_weights / sample_weights.sum()).sum()
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()
        loss = loss.item()
        running_loss += loss
        del output
        gc.collect()
    else:
        del loss
        gc.collect()
        if e == 0:
            tep1 = time.time()
            print('first training epoch done, now starting first evaluation')
            
            with open(f"/home/um106329/aisafety/may_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_default_{default}_{n_samples}_jets.txt", "a") as log:
                log.write(f"{np.floor((tep1-start)/60)} min {np.ceil((tep1-start)%60)} s"+' First training epoch done! Start first evaluation.\n')
            
        with torch.no_grad():
            model.eval()
            if e > 0:
                del vloss
                del val_output
                gc.collect()
            running_val_loss = 0
            for i,j in valloader:
                i = i.to(device, non_blocking=True)
                j = j.to(device, non_blocking=True)
                val_output = model(i.float())
                vloss = criterion(val_output, j)
                del i
                del j
                gc.collect()
                if weighting_method == '_ptetaflavloss':
                    vloss = vloss.mean().item()
                else:
                    vloss = vloss.item()
                running_val_loss += vloss
            '''
            # Old method to calc validation loss
            val_output = model(prepro_val_inputs)
            val_loss = criterion(val_output, prepro_val_targets)
            '''
            val_loss_history.append(running_val_loss/total_len_val)
            
            

            if stale_epochs > max_stale_epochs:
                print(f'training stopped by reaching {max_stale_epochs} stale epochs.                                                              ')
                break
            if running_val_loss < min_loss:
                min_loss = running_val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            # e+1 to count from "1" instead of "0"
            print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {running_val_loss/total_len_val}",end='\n')
        loss_history.append(running_loss/total_len_train)
        #if (e+1)%np.floor(epochs/10)==0:
        #    print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss/total_len_val}")
            
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": running_val_loss/total_len_val}, f'/hpcwork/um106329/may_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt')
        # I'm now saving the model both on /hpcwork (fast and mounted for the batch system) and /home (slow, but there are backups)
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": running_val_loss/total_len_val}, f'/home/um106329/aisafety/may_21/train_models/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt')
toc = time.time()
#print(f"{(e+1)/epochs*100}% done. Epoch: {e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {running_val_loss/total_len_val}\nTraining complete. Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"used {NUM_DATASETS} files, {prev_epochs+epochs} epochs, dropout 0.1 4x, learning rate {lrate}")


torch.save(model, f'/hpcwork/um106329/may_21/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}_justmodel.pt')
torch.save(model, f'/home/um106329/aisafety/may_21/train_models/saved_models/{weighting_method}_{NUM_DATASETS}_{default}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}_justmodel.pt')

times.append(toc)
for p in range(epochs):
    print(f"Time for epoch {prev_epochs+p}: {np.floor((times[p+1]-times[p])/60)} min {((times[p+1]-times[p])%60)} s")
end = time.time()
print(f"Total time for whole script: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")


