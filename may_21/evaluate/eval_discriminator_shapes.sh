#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=36G

#SBATCH --cpus-per-task=1

#SBATCH --job-name=EVAL_DISCRIMINATOR_SHAPES

#SBATCH --output=output.%J.txt

#SBATCH --time=04:10:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety/may_21/evaluate
# module unload intelmpi; module switch intel gcc
# module load cuda/110
# module load cudnn
source ~/miniconda3/bin/activate
conda activate my-env
python3 eval_discriminator_shapes.py 20 100 '_compare' '0.001' '-1' 'yes' 'yes'
