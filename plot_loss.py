import torch

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.cms.style.ROOT)




prev_epochs_0 = 120
prev_epochs_1 = 80
prev_epochs_2 = 120



NUM_DATASETS = 200


weighting_method_0 = '_as_is'

all_tr_losses_0 = []
all_val_losses_0 = []



for i in range(1, prev_epochs_0+1):
    checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{i}_epochs_v13_GPU_weighted{weighting_method_0}.pt' % NUM_DATASETS)
    loss = checkpoint['loss']
    all_tr_losses_0.append(loss)
    val_loss = checkpoint['val_loss']
    all_val_losses_0.append(val_loss)
    
    
    
    
weighting_method_1 = ''

all_tr_losses_1 = []
all_val_losses_1 = []



for i in range(1, prev_epochs_1+1):
    checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{i}_epochs_v13_GPU_weighted{weighting_method_1}.pt' % NUM_DATASETS)
    loss = checkpoint['loss']
    all_tr_losses_1.append(loss)
    val_loss = checkpoint['val_loss']
    all_val_losses_1.append(val_loss)
    
    
    
    
weighting_method_2 = '_new'

all_tr_losses_2 = []
all_val_losses_2 = []



for i in range(1, prev_epochs_2+1):
    checkpoint = torch.load(f'/home/um106329/aisafety/models/weighted/%d_full_files_{i}_epochs_v13_GPU_weighted{weighting_method_2}.pt' % NUM_DATASETS)
    loss = checkpoint['loss']
    all_tr_losses_2.append(loss)
    val_loss = checkpoint['val_loss']
    all_val_losses_2.append(val_loss)
    
    
    
    
plt.figure(1,figsize=[15,7.5])
plt.plot(all_tr_losses_0,color='midnightblue')
plt.plot(all_val_losses_0,color='royalblue')
plt.plot(all_tr_losses_1,color='saddlebrown')
plt.plot(all_val_losses_1,color='orange')
plt.plot(all_tr_losses_2,color='forestgreen')
plt.plot(all_val_losses_2,color='lawngreen')
plt.title(f"Training history with {NUM_DATASETS} files, {max(prev_epochs_0,prev_epochs_1,prev_epochs_2)} epochs")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training loss 0','validation loss 0','training loss 1','validation loss 1','training loss 2','validation loss 2'])
plt.savefig(f'/home/um106329/aisafety/models/weighted/%d_full_files_{max(prev_epochs_0,prev_epochs_1,prev_epochs_2)}_epochs_v13_GPU_train_history.png' % NUM_DATASETS)
