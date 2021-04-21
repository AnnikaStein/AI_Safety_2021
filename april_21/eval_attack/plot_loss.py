import torch

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.cms.style.ROOT)

import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
#parser.add_argument("wm", help="Weighting method")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
#weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)



weighting_method_0 = '_new'

all_tr_losses_0 = []
all_val_losses_0 = []



for i in range(1, prev_epochs+1):
    checkpoint = torch.load(f'/hpcwork/um106329/april_21/saved_models/TT{weighting_method_0}_{NUM_DATASETS}_{default}/model_all_TT_{i}_epochs_v10_GPU_weighted{weighting_method_0}_{NUM_DATASETS}_datasets_with_default_{default}.pt', map_location=torch.device(device))
    loss = checkpoint['loss']
    all_tr_losses_0.append(loss)
    val_loss = checkpoint['val_loss']
    all_val_losses_0.append(val_loss)
    
    
    
    
weighting_method_1 = '_as_is'

all_tr_losses_1 = []
all_val_losses_1 = []



for i in range(1, prev_epochs+1):
    checkpoint = torch.load(f'/hpcwork/um106329/april_21/saved_models/TT{weighting_method_1}_{NUM_DATASETS}_{default}/model_all_TT_{i}_epochs_v10_GPU_weighted{weighting_method_1}_{NUM_DATASETS}_datasets_with_default_{default}.pt', map_location=torch.device(device))
    loss = checkpoint['loss']
    all_tr_losses_1.append(loss)
    val_loss = checkpoint['val_loss']
    all_val_losses_1.append(val_loss)
    

plt.ioff()    
    
plt.figure(1,figsize=[15,9.5])
plt.plot(all_tr_losses_0,color='midnightblue',label='Training loss (loss weighting)')
plt.plot(all_val_losses_0,color='royalblue',label='Validation loss (loss weighting)')
plt.plot(all_tr_losses_1,color='saddlebrown',label='Training loss (no weighting)')
plt.plot(all_val_losses_1,color='orange',label='Validation loss (no weighting)')
plt.title(f"Training history with {NUM_DATASETS} files, {prev_epochs} epochs, each with default {default}")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f'/home/um106329/aisafety/april_21/eval_attack/plots/{NUM_DATASETS}_full_files_{prev_epochs}_epochs_v13_GPU_train_history_wm{weighting_method_0}{weighting_method_1}_default_{default}.png', bbox_inches='tight', dpi=600, facecolor='w', transparent=False)
