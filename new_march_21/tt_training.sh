#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=2

#SBATCH --mem-per-cpu=40G

#SBATCH --cpus-per-task=2

#SBATCH --job-name=TTtr49_0_5_new

#SBATCH --output=output.%J.txt

#SBATCH --time=64:30:00

#SBATCH --account=rwth0583

# with gpu: 

#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety/new_march_21/code
module unload intelmpi; module switch intel gcc
module load cuda/110
module load cudnn
source ~/miniconda3/bin/activate
conda activate my-env
python3 tt_training.py 49 120 30 '_new'
