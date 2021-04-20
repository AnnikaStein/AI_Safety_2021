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
from torch.utils.data import TensorDataset, ConcatDataset, WeightedRandomSampler


import time
import random
import gc

import argparse
#import ast

import os


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
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
epochs = args.addep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
#minima = np.load('/home/um106329/aisafety/april_21/from_Nik/minima.npy')  # this should only be needed when applying the attack
#defaults = minima - default
#defaults = -999*np.ones(100)

'''
    Available weighting methods:
        '_as_is'  :  apply no weighting factors at all (will converge to a nice result performance-wise, but the discriminator shapes are basically just two bins (0 or 1, almost nothing in between))
        ''        :  with factor 1 - relative frequency per flavour category (----> deprecated <----, as the weights have not been recalculated for the new samples)
        '_new'    :  n_samples / (n_classes * n_class_count) per flavour category in loss function (this is the recommended weighting method, if there is a class imbalance, and one does not need to create the weights beforehand (nice!))
        '_wrs'    :  using WeightedRandomSampler with n_samples / (n_classes * n_class_count) (does work, but the performance is not that good)
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
        
    parent_dir_1 = '/home/um106329/aisafety/april_21/train_models/saved_models'
    directory_1 = f'TT{weighting_method}_{NUM_DATASETS}_{default}'
    path_1 = os.path.join(parent_dir_1, directory_1)
    if not os.path.exists(path_1):
        os.mkdir(path_1)
    parent_dir_2 = '/hpcwork/um106329/april_21/saved_models'
    directory_2 = f'TT{weighting_method}_{NUM_DATASETS}_{default}'
    path_2 = os.path.join(parent_dir_2, directory_2)
    if not os.path.exists(path_2):
        os.mkdir(path_2)
        

# Parameters for the training and validation    
bsize = 1000000     # this might seem large, but for comparison: bsize of 250000 for 86M training inputs
lrate = 0.0001     # initial learning rate, only for first epoch


print(f'starting to train the model after {prev_epochs} epochs that were already done')
print(f'learning rate for this script: {lrate}')
print(f'batch size for this script: {bsize}')
    



with open(f"/home/um106329/aisafety/april_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_with_default_{default}.txt", "a") as log:
    log.write(f"Setup: TT to Semileptonic samples with default value {default}, weighting method {weighting_method}, so far {prev_epochs} epochs done. Use lr={lrate} and bsize={bsize}.\n")


train_input_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/train_inputs_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]
train_target_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/train_targets_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)] 
val_input_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/val_inputs_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]
val_target_file_paths = [f'/hpcwork/um106329/april_21/scaled_TT/val_targets_%d_with_default_{default}.pt' % k for k in range(0,NUM_DATASETS)]    



'''
# Old way to load validation samples, deprecated

# The new method for validation inputs is needed because the total file size for validation is too big to fit on a single gpu (16GB) that is already occupied by the model / loss 
# computation graph etc..
pre = time.time()


prepro_val_inputs = torch.cat(tuple(torch.load(vi).to(device) for vi in val_input_file_paths)).float()
print('prepro val inputs done')
prepro_val_targets = torch.cat(tuple(torch.load(vi).to(device) for vi in val_target_file_paths))
print('prepro val targets done')

post = time.time()
print(f"Time to load val: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
'''


##### LOAD TRAINING SAMPLES #####

pre = time.time()

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





pre = time.time()

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
    checkpoint = torch.load(f'/hpcwork/um106329/april_21/saved_models/TT{weighting_method}_{NUM_DATASETS}_{default}/model_all_TT_{prev_epochs}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}.pt')
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
    
    
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)


if prev_epochs > 0:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    '''
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
        g['lr'] = lrate/(1+ep/10)                      # adaptive learning rate
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


with open(f"/home/um106329/aisafety/april_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_with_default_{default}.txt", "a") as log:
    log.write(f"{np.floor((tic-start)/60)} min {np.ceil((tic-start)%60)} s"+' Everything prepared for main training loop.\n')




for e in range(epochs):
    times.append(time.time())
    if prev_epochs+e >= 1:
        new_learning_rate(prev_epochs+e)
    running_loss = 0
    model.train()
    for b, (i,j) in enumerate(trainloader):
        if e == 0 and b == 1:
            tb1 = time.time()
            print('first batch done')
            print(f"Time for first batch: {np.floor((tb1-times[0])/60)} min {((tb1-times[0])%60)} s")
            
            with open(f"/home/um106329/aisafety/april_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_with_default_{default}.txt", "a") as log:
                log.write(f"{np.floor((tb1-start)/60)} min {np.ceil((tb1-start)%60)} s"+' First batch done!\n')
            
        i = i.to(device, non_blocking=True)
        j = j.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(i.float())
        loss = criterion(output, j)
        del i
        del j
        gc.collect()
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
            
            with open(f"/home/um106329/aisafety/april_21/train_models/status_logfiles/logfile{weighting_method}_{NUM_DATASETS}_files_with_default_{default}.txt", "a") as log:
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
            
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": running_val_loss/total_len_val}, f'/hpcwork/um106329/april_21/saved_models/TT{weighting_method}_{NUM_DATASETS}_{default}/model_all_TT_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}.pt')
        # I'm now saving the model both on /hpcwork (fast and mounted for the batch system) and /home (slow, but there are backups)
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": running_val_loss/total_len_val}, f'/home/um106329/aisafety/april_21/train_models/saved_models/TT{weighting_method}_{NUM_DATASETS}_{default}/model_all_TT_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}.pt')
toc = time.time()
#print(f"{(e+1)/epochs*100}% done. Epoch: {e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {running_val_loss/total_len_val}\nTraining complete. Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"used {NUM_DATASETS} files, {prev_epochs+epochs} epochs, dropout 0.1 4x, learning rate {lrate}")


torch.save(model, f'/hpcwork/um106329/april_21/saved_models/TT{weighting_method}_{NUM_DATASETS}_{default}/model_all_TT_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_justmodel.pt')
torch.save(model, f'/home/um106329/aisafety/april_21/train_models/saved_models/TT{weighting_method}_{NUM_DATASETS}_{default}/model_all_TT_{prev_epochs+(e + 1)}_epochs_v10_GPU_weighted{weighting_method}_{NUM_DATASETS}_datasets_with_default_{default}_justmodel.pt')

times.append(toc)
for p in range(epochs):
    print(f"Time for epoch {prev_epochs+p}: {np.floor((times[p+1]-times[p])/60)} min {((times[p+1]-times[p])%60)} s")
end = time.time()
print(f"Total time for whole script: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")


