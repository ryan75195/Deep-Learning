#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-16:00:00

#export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDA_HOME=/opt/cuda-10.2.89/ 
#export CUDNN_HOME=/opt/cuDNN-7.0/
export CUDNN_HOME=/opt/cuDNN-7.6.5.32/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

id=$(whoami)

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

#print(torch.cuda.is_available())
export CUDA_VISIBLE_DEVICES=0
#print(torch.cuda.is_available())
python expBuildLr.py
