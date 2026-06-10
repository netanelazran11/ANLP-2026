#!/bin/bash
#SBATCH --job-name=anlp_ex2_nazran
#SBATCH --output=ex2_%j.out
#SBATCH --error=ex2_%j.err
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate
python ex2_part2.py
