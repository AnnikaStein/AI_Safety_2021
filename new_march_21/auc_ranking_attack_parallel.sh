#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=2

#SBATCH --mem-per-cpu=80G

#SBATCH --cpus-per-task=1

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
python3 auc_ranking_attack_parallel.py 49 ${MIN} ${MAX} 'FGSM' '0.005'
# python3 auc_ranking_attack_parallel.py 49 7 11 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 12 17 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 18 23 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 24 29 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 30 35 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 36 41 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 42 47 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 48 53 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 54 59 'noise' '0.005'
# python3 auc_ranking_attack_parallel.py 49 60 67 'noise' '0.005'