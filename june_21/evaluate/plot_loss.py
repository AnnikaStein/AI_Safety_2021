import torch

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])

import argparse

import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple weighting methods is possible (split by +)")
parser.add_argument("default", type=float, help="Default value")  # new, based on Nik's work
parser.add_argument("jets", help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type _-1 (using multiple: split them by , like so: _-1,_-1,_-1)")
args = parser.parse_args()

# example: python plot_loss.py 278 '_ptetaflavloss_focalloss_gamma25.0+_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.005+_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01' '0.001' '_-1,_-1,_-1'


NUM_DATASETS = args.files
weighting_method = args.wm
wmets = [w for w in weighting_method.split('+')]
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)

n_samples = args.jets
all_n_samples = [n[1:] for n in n_samples.split(',')]
print(all_n_samples)

gamma = [((weighting_method.split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0] for weighting_method in wmets]
alphaparse = [((weighting_method.split('_gamma')[-1]).split('_alpha')[-1]).split('_adv_tr_eps')[0] for weighting_method in wmets]
epsilon = [(weighting_method.split('_adv_tr_eps')[-1]) for weighting_method in wmets]
print('gamma',gamma)
print('alpha',alphaparse)
print('epsilon',epsilon)

wm_def_text = {'_noweighting': 'No weighting', 
               '_ptetaflavloss' : r'$p_T, \eta$ rew.',
               '_flatptetaflavloss' : r'$p_T, \eta$ rew. (Flat)',
               '_ptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (F.L. $\gamma=$2.0)', 
               '_flatptetaflavloss_focalloss' : r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$2.0)', 
              }

more_text = [(f'_ptetaflavloss_focalloss_gamma{g}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g}'+r', $\alpha=$'+f'{a})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_alpha{a}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'2.0'+r', $\alpha=$'+f'{a})') for a in alphaparse] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'{g})') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'2.0)') for g in gamma] + \
            [(f'_ptetaflavloss_focalloss' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'2.0)') for g in gamma] + \
            [(f'_flatptetaflavloss_focalloss_gamma{g}_alpha{a}' , r'$p_T, \eta$ rew. (Flat, F.L. $\gamma=$'+f'{g}'+r', $\alpha=$'+f'{a})') for g, a in zip(gamma,alphaparse)] + \
            [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}' , r'$p_T, \eta$ rew. (F.L. $\gamma=$'+f'{g}, $\epsilon=$'+f'{e})') for g, a, e in zip(gamma,alphaparse,epsilon)]

more_text_dict = {k:v for k, v in more_text}
wm_def_text =  {**wm_def_text, **more_text_dict}

colorcode = ['darkblue', 'royalblue', 'forestgreen', 'limegreen', 'maroon','red','darkolivegreen','yellow', 'darkcyan', 'cyan']

wm_epochs_so_far = {
    '_ptetaflavloss_focalloss_gamma40.0_alpha0.05,0.05,0.05,0.85' : 50,
    '_ptetaflavloss_focalloss_gamma30.0_alpha0.05,0.05,0.05,0.85' : 50,
    '_ptetaflavloss_focalloss_gamma25.0' : 200,
    '_flatptetaflavloss_focalloss_gamma25.0' : 200,
    '_flatptetaflavloss_focalloss' : 230,
    '_ptetaflavloss_focalloss' : 250,
    '_ptetaflavloss_focalloss_gamma13.0_adv_tr_eps0.005' : 15,
    '_ptetaflavloss_focalloss_gamma13.0' : 15,
    '_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.005' : 47,
    '_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01' : 200,
    
}

plt.ioff()    
    
plt.figure(1,figsize=[13,10])

for k,wm in enumerate(wmets):
    
    all_tr_losses = []
    all_val_losses = []
        
    for i in range(1, wm_epochs_so_far[wm]+1):
        checkpoint = torch.load(f'/hpcwork/um106329/june_21/saved_models/{wm}_{NUM_DATASETS}_{default}_{all_n_samples[k]}/model_{i}_epochs_v10_GPU_weighted{wm}_{NUM_DATASETS}_datasets_with_default_{default}_{all_n_samples[k]}.pt', map_location=torch.device(device))
        all_tr_losses.append(checkpoint['loss'])
        all_val_losses.append(checkpoint['val_loss'])
        
    all_epochs = np.arange(1,wm_epochs_so_far[wm]+1)  
    plt.plot(all_epochs, all_tr_losses,color=colorcode[k*2],label=f'Training loss ({wm_def_text[wm]})')
    plt.plot(all_epochs, all_val_losses,color=colorcode[k*2+1],label=f'Validation loss ({wm_def_text[wm]})')
        
plt.title(f"Training history", y=1.02)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
prev_epochs = [wm_epochs_so_far[wm] for wm in wmets]
plt.savefig(f'/home/um106329/aisafety/june_21/evaluate/loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{wmets}_default_{default}_{all_n_samples}_samples.png', bbox_inches='tight', dpi=400, facecolor='w', transparent=False)
plt.savefig(f'/home/um106329/aisafety/june_21/evaluate/loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{wmets}_default_{default}_{all_n_samples}_samples.svg', bbox_inches='tight')
plt.savefig(f'/home/um106329/aisafety/june_21/evaluate/loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{wmets}_default_{default}_{all_n_samples}_samples.pdf', bbox_inches='tight')
