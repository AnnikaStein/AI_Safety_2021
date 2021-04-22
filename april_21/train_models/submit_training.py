#!/usr/bin/python3
## file: submit_training.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess

import numpy as np

import argparse


parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument('-f',"--files", type=int, help="Number of files for training", default=49)
parser.add_argument('-p',"--prevep", type=int, help="Number of previously trained epochs", default=0)
parser.add_argument('-a',"--addep", type=int, help="Number of additional epochs for this training", default=30)
parser.add_argument('-w',"--wm", help="Weighting method", default="_new")
parser.add_argument('-d',"--default", type=float, help="Default value", default='0.001')  # new, based on Nik's work
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
epochs = args.addep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
    
    
home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/april_21/train_models/"
print('Shell script is located at:\t',shPath)

time = 11
mem = 30

factor_FILES = NUM_DATASETS / 49.0
factor_EPOCHS = epochs / 30.0

if NUM_DATASETS < 49:
    time = int(np.rint(time * factor_FILES * factor_EPOCHS) + 2)

    mem = int(np.rint(mem * factor_FILES) + 8)
else:
    time = int(np.rint(time * factor_EPOCHS) + 2)

    
submit_command = ("sbatch "
        "--time={6}:30:00 "
        "--mem-per-cpu={5}G "
        "--job-name=TTtr_{0}_{1}_{2}{3}_{4} "
        "--export=FILES={0},PREVEP={1},ADDEP={2},WM={3},DEFAULT={4} {7}training.sh").format(NUM_DATASETS, prev_epochs, epochs, weighting_method, default, mem, time, shPath)

print(submit_command)
exit_status = subprocess.call(submit_command, shell=True)
if exit_status==1:  # Check to make sure the job submitted
    print("Job {0} failed to submit".format(submit_command))
print("Done submitting jobs!")
