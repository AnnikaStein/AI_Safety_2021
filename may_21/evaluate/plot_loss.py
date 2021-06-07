import torch

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.cms.style.ROOT)

import argparse

import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
parser.add_argument("wm", help="Weighting method")  # '_noweighting', '_ptetaflavloss' or '_compare'
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("dominimal", help="Only do training with minimal setup, i.e. 15 QCD, 5 TT files")
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)

n_samples = args.jets
do_minimal = args.dominimal


do_noweighting   = True if ( weighting_method == '_compare' or weighting_method == '_noweighting'   ) else False
do_ptetaflavloss = True if ( weighting_method == '_compare' or weighting_method == '_ptetaflavloss' ) else False

if do_noweighting:

    all_tr_losses_noweighting = []
    all_val_losses_noweighting = []
    
      
    max_epoch_noweighting = 5 if NUM_DATASETS == 278 else 180 if NUM_DATASETS == 20 else prev_epochs
    
    for i in range(1, max_epoch_noweighting+1):
        checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/_noweighting_{NUM_DATASETS}_{default}_{n_samples}/model_{i}_epochs_v10_GPU_weighted_noweighting_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
        all_tr_losses_noweighting.append(checkpoint['loss'])
        all_val_losses_noweighting.append(checkpoint['val_loss'])
        
        
if do_ptetaflavloss:

    all_tr_losses_ptetaflavloss = []
    all_val_losses_ptetaflavloss = []



    for i in range(1, prev_epochs+1):
        checkpoint = torch.load(f'/hpcwork/um106329/may_21/saved_models/_ptetaflavloss_{NUM_DATASETS}_{default}_{n_samples}/model_{i}_epochs_v10_GPU_weighted_ptetaflavloss_{NUM_DATASETS}_datasets_with_default_{default}_{n_samples}.pt', map_location=torch.device(device))
        all_tr_losses_ptetaflavloss.append(checkpoint['loss'])
        all_val_losses_ptetaflavloss.append(checkpoint['val_loss'])

# one could also add ptetaflavsampler in the future, but it is so slow that I don't expect any comparable results

    
n_samples_text = f', with {n_samples} samples' if (n_samples != -1) else ''    
  

plt.ioff()    
    
plt.figure(1,figsize=[10,8])
if do_ptetaflavloss:   
    all_epochs = np.arange(1,prev_epochs+1)  
    plt.plot(all_epochs, all_tr_losses_ptetaflavloss,color='midnightblue',label=r'Training loss ($p_T,\eta$ loss weighting)')
    plt.plot(all_epochs, all_val_losses_ptetaflavloss,color='royalblue',label=r'Validation loss ($p_T,\eta$ loss weighting)')
if do_noweighting:
    few_epochs = np.arange(1,max_epoch_noweighting+1) 
    plt.plot(few_epochs, all_tr_losses_noweighting,color='saddlebrown',label='Training loss (no weighting)')
    plt.plot(few_epochs, all_val_losses_noweighting,color='orange',label='Validation loss (no weighting)')
plt.title(f"Training history with {NUM_DATASETS} files, {prev_epochs} epochs{n_samples_text}", y=1.02)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig(f'/home/um106329/aisafety/may_21/evaluate/loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{weighting_method}_default_{default}_{n_samples}_samples.png', bbox_inches='tight', dpi=400, facecolor='w', transparent=False)
