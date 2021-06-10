#!/usr/bin/python3
## file: submit_training.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess

import numpy as np

import argparse


parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument('-f',"--files", type=int, help="Number of files for training", default=278)
parser.add_argument('-p',"--prevep", type=int, help="Number of previously trained epochs", default=0)
parser.add_argument('-a',"--addep", type=int, help="Number of additional epochs for this training", default=30)
parser.add_argument('-w',"--wm", help="Weighting method", default="_ptetaflavloss")
parser.add_argument('-d',"--default", type=float, help="Default value", default='0.001')  # new, based on Nik's work
parser.add_argument('-j',"--jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1", default=-1)
parser.add_argument('-m',"--dominimal", help="Only do training with minimal setup, i.e. 15 QCD, 5 TT files", default='no')
parser.add_argument('-l',"--dofastdl", help="Use fast DataLoader", default='yes')
parser.add_argument('-fl',"--dofl", help="Use Focal Loss", default='yes')
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
epochs = args.addep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
n_samples = args.jets
do_minimal = args.dominimal
do_fastdataloader = args.dofastdl
do_FL = args.dofl
    
    
home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/may_21/train_models/"
print('Shell script is located at:\t',shPath)

if NUM_DATASETS == 20:
    time = 4
    mem = 16
else:
    time = 9
    mem = 182
    factor_FILES = NUM_DATASETS / 278.0

factor_EPOCHS = epochs / 40.0

if NUM_DATASETS == 20:
    time = int(np.rint(time * factor_EPOCHS) + 0.5)
    
elif NUM_DATASETS == 278:
    time = int(np.rint(time * factor_EPOCHS) + 0.5)

else:
    time = int(np.rint(time * factor_FILES * factor_EPOCHS) + 2)

    mem = min(int(np.rint(mem * factor_FILES) + 24),182)

    
submit_command = ("sbatch "
        "--time={6}:00:00 "
        "--mem-per-cpu={5}G "
        "--job-name=tr_{0}_{1}_{2}{3}_{4}_{8}_{9}_{10}_{11} "
        "--export=FILES={0},PREVEP={1},ADDEP={2},WM={3},DEFAULT={4},NJETS={8},DOMINIMAL={9},FASTDATALOADER={10},FOCALLOSS={11} {7}training.sh").format(NUM_DATASETS, prev_epochs, epochs, weighting_method, default, mem, time, shPath, n_samples, do_minimal, do_fastdataloader, do_FL)

print(submit_command)
userinput = input("Submit job? (y/n) ").lower() == 'y'
if userinput:
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status==1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))
    print("Done submitting jobs!")
