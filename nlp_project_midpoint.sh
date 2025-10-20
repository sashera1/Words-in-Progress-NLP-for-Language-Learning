#!/bin/bash
#SBATCH --job-name=translation_job # Job name
#SBATCH -A cs4770_fa25
#SBATCH --partition=gpu
#SBATCH --output=output.txt # Output file
#SBATCH --error=error.txt # Error file
#SBATCH --time=00:05:00 # Time limit (hh:mm:ss)
#SBATCH --mem=8G # Memory allocation
#SBATCH --cpus-per-task=4 # Number of CPU cores
#SBATCH --gres=gpu:1

module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

export PATH=$HOME/.local/bin:$PATH

#comment these out after successful run (or failed run but they've been installed successfully)
#pip install -U transformers
#pip install "unbabel-comet>=2.0.0"
#pip install --user google-generativeai

#maybe
#pip install --upgrade pip

#only need to run once- change for whatever token you need to use if necessary
#Later on, we should use a more secure way of doing this
#export HF_TOKEN= REMOVED FOR MIDPOINT SUBMISSIONS

python translation_model.py