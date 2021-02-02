# this one kind of creates a "new branch"
# starting from the previous epoch 1, 
# now with a lower learning rate than 0,001
# but only half the batch_size (500K instead of 1M)

#import uproot4 as uproot
#import uproot as uproot3
import numpy as np
#import awkward1 as ak

import matplotlib.pyplot as plt
import mplhep as hep


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset

#from sys import getsizeof

import time
import random
import gc

start = time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
#torch.multiprocessing.set_start_method('spawn', force=True)

#This is just some plot styling
plt.style.use(hep.cms.style.ROOT)
#C = ['firebrick', 'darkgreen', 'darkblue', 'grey', 'cyan','magenta']
#path = 'Figures/'

#class SingleFile(torch.utils.data.Dataset):
    #def __init__(self, file_path):
    #    #self.data = torch.load(file_path, map_location=device)[:1000]
    #    #self.data = torch.load(file_path)[:1000]
    #    self.input = torch.load('/work/um106329/MA/preprocessed_files/train_inputs_0.pt').to(device, non_blocking=False)[:2000]
    #    self.target = torch.load('/work/um106329/MA/preprocessed_files/train_targets_0.pt').to(device, non_blocking=False)[:2000]
    #    self.len = len(self.target)
#    def __init__(self, f):
        #self.data = torch.load(file_path, map_location=device)[:1000]
        #self.data = torch.load(file_path)[:1000]   
#        self.input = torch.load('/work/um106329/MA/preprocessed_files/train_inputs_%d.pt' % f)     
#        self.target = torch.load('/work/um106329/MA/preprocessed_files/train_targets_%d.pt' % f)
#        self.len = len(self.target)
#    def __getitem__(self, index):
        #inputs = self.data[index][0]
        #labels = self.data[index][1]
#        inputs = self.input[index]
#        labels = self.target[index]
#        return inputs, labels

#    def __len__(self):
        #return len(self.target)
#        return self.len
    
bsize = 350000    
lrate = 0.00001
prev_epochs = 91
print(f'starting to train the model after {prev_epochs} epochs that were already done, see aisafety/output.18976501.txt')
print(f'learning rate for this script: {lrate}')
print(f'batch size for this script: {bsize}')
    
NUM_DATASETS = 200
#with open("/home/um106329/aisafety/models/all_events_per_%d_files_100_epochs_v8.txt" % NUM_DATASETS, "w+") as text_file:
#    print(f'will start training with {NUM_DATASETS} datasets', file=text_file)
#trainset_file_paths = ['/work/um106329/MA/preprocessed_files/trainset_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_input_file_paths = ['/work/um106329/MA/preprocessed_files/train_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
train_target_file_paths = ['/work/um106329/MA/preprocessed_files/train_targets_%d.pt' % k for k in range(0,NUM_DATASETS)] 
val_input_file_paths = ['/work/um106329/MA/preprocessed_files/val_inputs_%d.pt' % k for k in range(0,NUM_DATASETS)]
val_target_file_paths = ['/work/um106329/MA/preprocessed_files/val_targets_%d.pt' % k for k in range(0,NUM_DATASETS)]    

pre = time.time()

# perform cat and .float on gpu (hopefully), first version did all that on cpu
#prepro_val_inputs = torch.cat(tuple(torch.load(vi) for vi in val_input_file_paths)).float().to(device)
prepro_val_inputs = torch.cat(tuple(torch.load(vi).to(device) for vi in val_input_file_paths)).float()
#prepro_val_inputs = torch.cat(tuple(torch.load(vi) for vi in val_input_file_paths)).float()
#print('prepro val inputs done')
print('prepro val inputs done')
#prepro_val_targets = torch.cat(tuple(torch.load(vi) for vi in val_target_file_paths)).to(device)
prepro_val_targets = torch.cat(tuple(torch.load(vi).to(device) for vi in val_target_file_paths))
#prepro_val_targets = torch.cat(tuple(torch.load(vi) for vi in val_target_file_paths))
print('prepro val targets done')

post = time.time()
print(f"Time to load val: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")


pre = time.time()

allin = ConcatDataset([TensorDataset(torch.load(train_input_file_paths[f]),torch.load(train_target_file_paths[f])) for f in range(NUM_DATASETS)])

post = time.time()
print(f"Time to load train: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")


pre = time.time()

trainloader = torch.utils.data.DataLoader(allin, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)

post = time.time()
print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")

total_len_train = len(trainloader)

model5 = nn.Sequential(nn.Linear(67, 100),
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

checkpoint = torch.load(f'/home/um106329/aisafety/models/%d_full_files_{prev_epochs}_epochs_v12_GPU_new.pt' % NUM_DATASETS)
model5.load_state_dict(checkpoint["model_state_dict"])

print(model5)


model5.to(device)

#print('located at', prepro_val_inputs.device)

criterion5 = nn.CrossEntropyLoss()
optimizer5 = torch.optim.Adam(model5.parameters(), lr=lrate)

optimizer5.load_state_dict(checkpoint["optimizer_state_dict"])


for g in optimizer5.param_groups:
    print('lr: ', g['lr'], 'prev run')
    g['lr'] = lrate
    print('lr: ', g['lr'], 'after update')


#The training algorithm

tic = time.time()
loss_history, val_loss_history = [], []
stale_epochs, min_loss = 0, 10
max_stale_epochs = 100
# epochs to be trained with the current script (on top of the prev_epochs)
epochs = 9
times = []

#files = list(range(NUM_DATASETS))

for e in range(epochs):
    #if e == 1:
        #e1time = time.time() 
    #if e == 2:
        #e2time = time.time() 
    times.append(time.time())
    #random.shuffle(files)
    running_loss = 0
    #total_len_train = 0
    #for f in files:
        #onefile = SingleFile(trainset_file_paths[f])
        #onefile = SingleFile(f)
        #onefile = TensorDataset(torch.load(train_input_file_paths[f]),torch.load(train_target_file_paths[f]))
        #trainloader = torch.utils.data.DataLoader(onefile, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
        #if e == 0 and total_len_train == 0:
        #    tac = time.time()
        #    print('first trainloader done, now starting first training')
        #    print(f"Time to create first trainloader: {np.floor((tac-tic)/60)} min {np.ceil((tac-tic)%60)} s")
            #with open("/home/um106329/aisafety/models/all_events_per_%d_files_100_epochs_v8.txt" % NUM_DATASETS, "w+") as text_file:
            #    print('first trainloader done, now starting first training', file=text_file)
        #running_loss_f = 0
    model5.train()
    for b, (i,j) in enumerate(trainloader):
        if e == 0 and b == 1:
            print('first batch done')
            print(f"Time for first batch: {np.floor((time.time()-times[0])/60)} min {((time.time()-times[0])%60)} s")
        i = i.to(device, non_blocking=True)
        j = j.to(device, non_blocking=True)
        #print(i)
        #print(j)  
        #if b == 0:
        #    print('first batch loaded')
        optimizer5.zero_grad()
        #if b == 0:
        #    print('first zero_grad done')
        output = model5(i.float())
        #if b == 0:
        #    print('first train inputs in model')          
        loss = criterion5(output, j)
        #if b == 0:
        #    print('first loss calc done')
        del i
        del j
        gc.collect()
        loss.backward()
        #if b == 0:
        #    print('first loss backward')
        optimizer5.step()
        #if b == 0:
        #    print('first optimizer step')
        loss = loss.item()
        running_loss += loss
        #if b == 0:
        #    print('first batch trained')
    #else:
     #   this_train_len = len(trainloader)
     #   running_loss += running_loss_f
     #   total_len_train += this_train_len
        #del onefile
        #del trainloader
        del output
        gc.collect()
        # for the next tests: try out del gc.garbage[:]
    else:
        del loss
        gc.collect()
        if e == 0:
            print('first training epoch done, now starting first evaluation')
            #with open("/home/um106329/aisafety/models/all_events_per_%d_files_100_epochs_v8.txt" % NUM_DATASETS, "w+") as text_file:
            #    print('first training epoch done, now starting first evaluation', file=text_file)
        with torch.no_grad():
            model5.eval()
            if e > 0:
                del val_loss
                del val_output
                gc.collect()
            val_output = model5(prepro_val_inputs)
            val_loss = criterion5(val_output, prepro_val_targets)
            val_loss_history.append(val_loss)
            
            

            if stale_epochs > max_stale_epochs:
                print(f'training stopped by reaching {max_stale_epochs} stale epochs.                                                              ')
                #with open("/home/um106329/aisafety/models/all_events_per_%d_files_100_epochs_v8.txt" % NUM_DATASETS, "w+") as text_file:
                #    print(f'training stopped by reaching {max_stale_epochs} stale epochs.                                                              ', file=text_file)
                break
            if val_loss.item() < min_loss:
                min_loss = val_loss.item()
                stale_epochs = 0
            else:
                stale_epochs += 1
            # e+1 to count from "1" instead of "0"
            print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}",end='\r')
            #with open("/home/um106329/aisafety/models/all_events_per_%d_files_100_epochs_v8.txt" % NUM_DATASETS, "w+") as text_file:
            #    print(f"{(e+1)/epochs*100}% done. Epoch: {e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}", file=text_file)
        loss_history.append(running_loss/total_len_train)
        if (e+1)%np.floor(epochs/10)==0:
            print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}")
            
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model5.state_dict(), "optimizer_state_dict": optimizer5.state_dict(), "loss": running_loss/total_len_train, "val_loss": val_loss}, f'/home/um106329/aisafety/models/%d_full_files_{prev_epochs+(e + 1)}_epochs_v12_GPU_new.pt' % NUM_DATASETS)
toc = time.time()
#print(f"{(e+1)/epochs*100}% done. Epoch: {e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}\nTraining complete. Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"used {NUM_DATASETS} files, {prev_epochs+epochs} epochs, dropout 0.1 4x, learning rate {lrate}")
#with open("/home/um106329/aisafety/models/all_events_per_%d_files_100_epochs_v8.txt" % NUM_DATASETS, "w+") as text_file:
    #print(f"{(e+1)/epochs*100}% done. Epoch: {e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {val_loss}\nTraining complete. Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s", file=text_file)
    #print(f"used {NUM_DATASETS} files, {epochs} epochs, dropout 0.1 4x, learing rate 0.00001", file=text_file)


torch.save(model5, f'/home/um106329/aisafety/models/%d_full_files_{prev_epochs+epochs}_epochs_v12_GPU_justModel_new.pt' % NUM_DATASETS)

#Plot loss and validation loss over epochs
#plt.figure(1,figsize=[15,7.5])
#plt.plot(loss_history,color='forestgreen')
#plt.plot(val_loss_history,color='lawngreen')
#plt.title(f"Training history with {NUM_DATASETS} files, {prev_epochs+epochs} epochs")
#plt.xlabel('epoch')
#plt.ylabel('loss')
#plt.legend(['training loss','validation loss'])
#plt.savefig(f'/home/um106329/aisafety/models/%d_full_files_{prev_epochs+epochs}_epochs_v12_GPU_new.png' % NUM_DATASETS)

times.append(toc)
#print(f"Time for first epoch: {np.floor((e1time-tic)/60)} min {np.ceil((e1time-tic)%60)} s")
#print(f"Time for second epoch: {np.floor((e2time-e1time)/60)} min {np.ceil((e2time-e1time)%60)} s")
for p in range(epochs):
    print(f"Time for epoch {prev_epochs+p}: {np.floor((times[p+1]-times[p])/60)} min {((times[p+1]-times[p])%60)} s")
end = time.time()
#print(f"Time for epoch {epochs-1}: {np.floor((end-times[epochs-1])/60)} min {np.ceil((end-times[epochs-1])%60)} s")
print(f"Total time for whole script: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")


