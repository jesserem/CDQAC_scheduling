#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=6:00:00

# Array training launcher.
#
# Each line of an experiment file in experiments/ holds the CLI arguments for
# one run (see experiments/*.txt). Submit one array task per line, e.g.:
#
#   sbatch --array=1-20 slurm/train_array.sbatch \
#       -r $HOME/cdqac -e diff_dataset_sd1_exp_10_5.txt -s train_cdqac.py
#
# Options:
#   -r  repository root (default: $HOME/cdqac)
#   -e  experiment file name inside experiments/
#   -s  training script (default: train_cdqac.py; use train_iql.py etc. for baselines)

REPO="$HOME/cdqac"
SCRIPT="train_cdqac.py"
while getopts "r:e:s:" opt; do
  case ${opt} in
    r ) REPO=$OPTARG ;;
    e ) EXPERIMENTS=$OPTARG ;;
    s ) SCRIPT=$OPTARG ;;
    * ) echo "Invalid option" 1>&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

module purge
module load 2024
module load Miniconda3/24.7.1-0
module load CUDA/12.6.0
source activate offline_fjsp_env   # create it first: conda create -n offline_fjsp_env python=3.11 && pip install -r requirements.txt

# Stage the dataset on the node-local scratch disk.
cp -r "$REPO/dataset" "$TMPDIR"

EXP_FILE="$REPO/experiments/$EXPERIMENTS"
PYTHON_COMMANDS="--data_path $TMPDIR/dataset $(head -n "$SLURM_ARRAY_TASK_ID" "$EXP_FILE" | tail -1)"
echo "$PYTHON_COMMANDS"

cd "$REPO"
srun python "$SCRIPT" $PYTHON_COMMANDS
