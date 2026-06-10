#!/bin/bash
#SBATCH --job-name=anlp_ex2_nazran
#SBATCH --output=ex2_%j.out
#SBATCH --error=ex2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

source venv/bin/activate
python ex2_part2.py
