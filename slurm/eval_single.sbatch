#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=2:00:00

# Evaluate one trained checkpoint folder on the FJSP benchmark and generated
# evaluation sets. The folder must contain the model .pt files and the
# config.yml written during training.
#
#   sbatch slurm/eval_single.sbatch -r $HOME/cdqac -m checkpoints/<run_folder>

REPO="$HOME/cdqac"
while getopts "r:m:" opt; do
  case ${opt} in
    r ) REPO=$OPTARG ;;
    m ) MODEL=$OPTARG ;;
    * ) echo "Invalid option" 1>&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

module purge
module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0
source activate offline_fjsp_env

cd "$REPO"
srun python -m evaluation.evaluate_fjsp --model_path "$MODEL"
