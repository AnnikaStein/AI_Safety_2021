#!/usr/bin/python3
## file: submit_eval_inputs.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os,sys
import subprocess

import numpy as np

import argparse


parser = argparse.ArgumentParser(description="Setup for input evaluation")
parser.add_argument('-v',"--variable", type=int, help="Index of input variable",default=0)
parser.add_argument('-a',"--attack", help="The type of the attack, noise or fgsm",default='fgsm')
parser.add_argument('-r',"--fixRange", help="Use predefined range (yes) or just as is (no)",default='no')
parser.add_argument('-pa',"--para", help="Parameter for attack or noise (epsilon or sigma), can be comma-separated.",default='0.01,0.02')
parser.add_argument('-f',"--files", help="Number of files for training",default='278')
parser.add_argument('-p',"--prevep", help="Number of previously trained epochs, can be a comma-separated list",default='1')
parser.add_argument('-w',"--wm", help="Weighting method: _noweighting, _ptetaflavloss, _flatptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible",default='_ptetaflavloss_focalloss_gamma25.0')
parser.add_argument('-d',"--default", help="Default value",default='0.001')  # new, based on Nik's work
parser.add_argument('-j',"--jets", help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1",default='-1')
parser.add_argument('-me',"--dominimal_eval", help="Only minimal number of files for evaluation",default='yes')


args = parser.parse_args()
variable = args.variable
attack = args.attack
fixRange = args.fixRange
para = args.para
NUM_DATASETS = args.files
at_epoch = args.prevep
weighting_method = args.wm
default = args.default  # new, based on Nik's work
    
n_samples = args.jets
do_minimal_eval = args.dominimal_eval

home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/june_21/evaluate/"
print('Shell script is located at:\t',shPath)


variables = [int(i) for i in variable.split(',')]


time = 33

#sys.exit()

for index in variables:
    submit_command = ("sbatch "
            "--time=00:{9}:00 "
            "--job-name=Inputs_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8} "
            "--mem-per-cpu=75G "
            "--export=VARI={0},ATTACK={1},FIXRANGE={2},PARA={3},FILES={4},WM={5},DEFAULT={6},JETS={7},DOMINIMAL_EVAL={8} {10}eval_inputs.sh").format(index,attack,fixRange,para,NUM_DATASETS,weighting_method,default,n_samples,do_minimal_eval, time, shPath)
    
    print(submit_command)
    userinput = input("Submit job? (y/n) ").lower() == 'y'
    if userinput:
        exit_status = subprocess.call(submit_command, shell=True)
        if exit_status==1:  # Check to make sure the job submitted
            print("Job {0} failed to submit".format(submit_command))
        print("Done submitting jobs!")

