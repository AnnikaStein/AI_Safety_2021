#!/usr/bin/python3
## file: submit_eval_roc.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess

import numpy as np

import argparse


parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument('-f',"--files", type=int, help="Number of files for training", default=49)
parser.add_argument('-p',"--prevep", type=int, help="Number of previously trained epochs", default=0)
parser.add_argument('-w',"--wm", help="Weighting method", default="_new")
parser.add_argument('-d',"--default", type=float, help="Default value", default='0.001')  # new, based on Nik's work
#parser.add_argument('-c',"--compare", help="Compare with earlier epochs", default='no')
parser.add_argument('-fl',"--dofl", help="Use Focal Loss", default='yes')
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
if default == int(default):
    default = int(default)
#compare = args.compare
do_FL = args.dofl
    
    
home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/april_21/eval_attack/"
print('Shell script is located at:\t',shPath)

time = 1
mem = 80

factor_FILES = NUM_DATASETS / 49.0

if NUM_DATASETS < 49:
    time = int(np.rint(time * factor_FILES) + 1)

    mem = int(np.rint(mem * factor_FILES) + 8)
    
if compare == 'yes':
    time *= 4
    mem += 10

    
submit_command = ("sbatch "
        "--time={5}:30:00 "
        "--mem-per-cpu={4}G "
        "--job-name=ROC_{0}_{1}{2}_{3} "
        "--export=FILES={0},PREVEP={1},WM={2},DEFAULT={3},FOCALLOSS={7} {6}eval_roc.sh").format(NUM_DATASETS, prev_epochs, weighting_method, default, mem, time, shPath, do_FL)

print(submit_command)
exit_status = subprocess.call(submit_command, shell=True)
if exit_status==1:  # Check to make sure the job submitted
    print("Job {0} failed to submit".format(submit_command))
print("Done submitting jobs!")
