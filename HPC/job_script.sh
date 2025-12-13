#!/bin/bash
#SBATCH --job-name=translation_job 
#SBATCH -A cs4770_fa25
#SBATCH --partition=gpu
#SBATCH --output=output.txt 
#SBATCH --error=error.txt
#SBATCH --time=00:05:00 
#SBATCH --mem=16G 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1

module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

export PATH=$HOME/.local/bin:$PATH

#comment these out after successful run (or failed run but they've been installed successfully)
pip install -U transformers
pip install "unbabel-comet>=2.0.0"
pip install --user google-generativeai

#if needed
#pip install --upgrade pip

#only need to run token exports once- change for whatever token you need to use if necessary
export HF_TOKEN=""
export GOOGLE_API_KEY=""

python translation_model.py
