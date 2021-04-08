#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=2

#SBATCH --mem-per-cpu=40G

#SBATCH --cpus-per-task=2

#SBATCH --job-name=AUCranking

#SBATCH --output=output.%J.txt

#SBATCH --time=1:30:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety/new_march_21/code
source ~/miniconda3/bin/activate
conda activate my-env
python3 auc_ranking_attack.py 1 'FGSM' '0.005'