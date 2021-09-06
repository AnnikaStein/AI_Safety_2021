#!/usr/bin/python3
## file: submit_auc.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os,sys
import subprocess

import numpy as np

import argparse


parser = argparse.ArgumentParser(description="Setup for AUC evaluation")
parser.add_argument('-f',"--files", help="Number of files for training",default='278')
parser.add_argument('-p',"--prevep", help="Number of previously trained epochs, can be a comma-separated list",default='1')
parser.add_argument('-c',"--comparesetup", help="Setup for comparison, examples: BvL_raw, CvB_sigma0.01, bb_eps0.01, can be a comma-separated list",default='BvL_raw')
parser.add_argument('-deep',"--plotdeepcsv", help="Plot DeepCSV for comparison",default='no')
parser.add_argument('-w',"--wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible",default='_ptetaflavloss_focalloss_gamma25.0')
parser.add_argument('-d',"--default", help="Default value",default='0.001')  # new, based on Nik's work
parser.add_argument('-j',"--jets", help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1",default='-1')
parser.add_argument('-me',"--dominimal_eval", help="Only minimal number of files for evaluation",default='yes')
parser.add_argument('-aa',"--add_axis", help="Add a second axis as inset to the plot",default='no')
parser.add_argument('-fgsm',"--FGSM_setup", help="If FGSM, create adversarial inputs from individual models (-1) or only for a given setup (specify this setup, e.g. type 2 for the 0,1,2 --> second index of all requested methods (counted from 0).)",default='-1')
parser.add_argument('-la',"--log_axis", help="Flip axis and use log scale (yes/no)",default='no')
parser.add_argument('-s',"--save_mode", help="Save AUC only, or save plots of ROC curves (ROC/AUC), only useful for multiple specified setups (compare).",default='AUC')
args = parser.parse_args()

NUM_DATASETS = args.files
at_epoch = args.prevep
compare_setup = args.comparesetup
plot_deepcsv = args.plotdeepcsv
weighting_method = args.wm
default = args.default  # new, based on Nik's work
    
n_samples = args.jets
do_minimal_eval = args.dominimal_eval
add_axis = args.add_axis
FGSM_setup = args.FGSM_setup
log_axis = args.log_axis
save_mode = args.save_mode

home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/june_21/evaluate/"
print('Shell script is located at:\t',shPath)



epochs = [int(e) for e in at_epoch.split(',')]
# example: 11,50 --> should yield 11,20, 21,30, 31,40, 41,50 (four packages = four jobs)
package_of_tens = (epochs[1]-epochs[0]+9) // 10
starts          = np.arange(epochs[0],epochs[0]+package_of_tens*10,10)
ends            = starts + 9
print(epochs)
print(package_of_tens)
print(starts)
print(ends)

time = 33

#sys.exit()

for index in range(len(ends)):
    prevep_str = str(starts[index])+','+str(ends[index])
    s = starts[index]
    e = ends[index]
    submit_command = ("sbatch "
            "--time=00:{13}:00 "
            #"--time=00:{4}:00 "
            "--job-name=AUC_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12} "
            #"--job-name=AUC_{0}_{1}_{2}_{3} "
            "--mem-per-cpu=75G "
            "--export=FILES={0},START={1},END={2},COMPARESETUP={3},PLOTDEEPCSV={4},WM={5},DEFAULT={6},JETS={7},DOMINIMAL_EVAL={8},ADD_AXIS={9},FGSM_SETUP={10},LOG_AXIS={11},SAVE_MODE={12} {14}eval_auc.sh").format(NUM_DATASETS,s,e,compare_setup,plot_deepcsv,weighting_method,default,n_samples,do_minimal_eval,add_axis,FGSM_setup,log_axis,save_mode, time, shPath)
            #"--export=START={0},END={1},COMPARESETUP={2},WM={3} {5}eval_auc.sh").format(s,e,compare_setup,weighting_method,time, shPath)
    
    print(submit_command)
    userinput = input("Submit job? (y/n) ").lower() == 'y'
    if userinput:
        exit_status = subprocess.call(submit_command, shell=True)
        if exit_status==1:  # Check to make sure the job submitted
            print("Job {0} failed to submit".format(submit_command))
        print("Done submitting jobs!")
