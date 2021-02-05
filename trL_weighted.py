import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset


import time
import random
import gc

start = time.time()


# depending on what's available, or force cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')



plt.style.use(hep.cms.style.ROOT)


'''
    Weighting method:
        '_as_is'  :  apply no weighting factors at all
        ''        :  with factor 1 - relative frequency per flavour category
        '_new'    :  with factor 1 / relative frequency per flavour category        
'''
weighting_method = '_new'   
print(f'weighting method: {weighting_method}')    

# Parameters for the training and validation    
bsize = 250000    
lrate = 0.00001
prev_epochs = 40

# Manually update the file path to the latest training job message
print(f'starting to train the model after {prev_epochs} epochs that were already done')
print(f'learning rate for this script: {lrate}')
print(f'batch size for this script: {bsize}')
    
NUM_DATASETS = 200

train_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)] 
val_input_file_paths = ['/work/um106329/MA/cleaned/preprocessed/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
val_target_file_paths = ['/work/um106329/MA/cleaned/preprocessed/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]    

pre = time.time()


prepro_val_inputs = torch.cat(tuple(torch.load(vi).to(device) for vi in val_input_file_paths)).float()
print('prepro val inputs done')
prepro_val_targets = torch.cat(tuple(torch.load(vi).to(device) for vi in val_target_file_paths))
print('prepro val targets done')

post = time.time()
print(f"Time to load val: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")


pre = time.time()

allin = ConcatDataset([TensorDataset(torch.load(train_input_file_paths[f]), torch.load(train_target_file_paths[f])) for f in range(NUM_DATASETS)])

post = time.time()
print(f"Time to load train: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")


pre = time.time()

trainloader = torch.utils.data.DataLoader(allin, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)

post = time.time()
print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")

total_len_train = len(trainloader)

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
    checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{prev_epochs}_epochs_v13_GPU_weighted{weighting_method}.pt' % NUM_DATASETS)
    model.load_state_dict(checkpoint["model_state_dict"])

print(model)


model.to(device)


# Choose the parameter for weighting in the loss function, according to the choice above
if weighting_method == '':
    # as calculated in dataset_info.ipynb
    allweights = [0.9393934969162745, 0.9709644530642717, 0.8684253665882813, 0.2212166834311725]
    class_weights = torch.FloatTensor(allweights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
elif weighting_method == '_new':
    allweights = [0.27580367992004956, 0.5756907770526237, 0.1270419391956182, 0.021463603831708488]
    class_weights = torch.FloatTensor(allweights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
else:
    criterion = nn.CrossEntropyLoss()
    
    
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)


if prev_epochs > 0:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # update the learning rate to the new one
    for g in optimizer.param_groups:
        print('lr: ', g['lr'], 'prev run')
        g['lr'] = lrate
        print('lr: ', g['lr'], 'after update')


#The training algorithm

tic = time.time()
loss_history, val_loss_history = [], []
stale_epochs, min_loss = 0, 10
max_stale_epochs = 100

# epochs to be trained with the current script (on top of the prev_epochs)
epochs = 10
times = []


for e in range(epochs):
    times.append(time.time())
    
    running_loss = 0
    model.train()
    for b, (i,j) in enumerate(trainloader):
        if e == 0 and b == 1:
            print('first batch done')
            print(f"Time for first batch: {np.floor((time.time()-times[0])/60)} min {((time.time()-times[0])%60)} s")
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
            print('first training epoch done, now starting first evaluation')
        with torch.no_grad():
            model.eval()
            if e > 0:
                del val_loss
                del val_output
                gc.collect()
            val_output = model(prepro_val_inputs)
            val_loss = criterion(val_output, prepro_val_targets)
            val_loss_history.append(val_loss)
            
            

            if stale_epochs > max_stale_epochs:
                print(f'training stopped by reaching {max_stale_epochs} stale epochs.                                                              ')
                break
            if val_loss.item() < min_loss:
                min_loss = val_loss.item()
                stale_epochs = 0
            else:
                stale_epochs += 1
            # e+1 to count from "1" instead of "0"
            print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}",end='\r')
        loss_history.append(running_loss/total_len_train)
        #if (e+1)%np.floor(epochs/10)==0:
        #    print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}")
            
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": val_loss}, f'/home/um106329/aisafety/models/weighted/%d_full_files_{prev_epochs+(e + 1)}_epochs_v13_GPU_weighted{weighting_method}.pt' % NUM_DATASETS)
toc = time.time()
#print(f"{(e+1)/epochs*100}% done. Epoch: {e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}\nTraining complete. Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"used {NUM_DATASETS} files, {prev_epochs+epochs} epochs, dropout 0.1 4x, learning rate {lrate}")


torch.save(model, f'/home/um106329/aisafety/models/weighted/%d_full_files_{prev_epochs+epochs}_epochs_v13_GPU_justModel_weighted{weighting_method}.pt' % NUM_DATASETS)

#Plot loss and validation loss over epochs
#plt.figure(1,figsize=[15,7.5])
#plt.plot(loss_history,color='forestgreen')
#plt.plot(val_loss_history,color='lawngreen')
#plt.title(f"Training history with {NUM_DATASETS} files, {prev_epochs+epochs} epochs")
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend(['training loss','validation loss'])
#plt.savefig(f'/home/um106329/aisafety/models/weighted/%d_full_files_{prev_epochs+epochs}_epochs_v13_GPU_weighted{weighting_method}.png' % NUM_DATASETS)

times.append(toc)
for p in range(epochs):
    print(f"Time for epoch {prev_epochs+p}: {np.floor((times[p+1]-times[p])/60)} min {((times[p+1]-times[p])%60)} s")
end = time.time()
print(f"Total time for whole script: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")

