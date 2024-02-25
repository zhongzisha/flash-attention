#!/bin/bash

module load gcc/11.2.0
module load cuda/cuda-11.7

nvidia-smi -a

nvidia-smi

ls /usr/local/*

free -g

df -h

uname -a



which python3
which conda


#python3 dataset/dataset.py

#python3 utils/train_tokenizer.py

#python3 utils/merge_tokenizer.py

#accelerate launch \
#--config_file configs/accelerate_configs/ds_stage1_frce_8gpus.yaml \
#train_lm.py --train_config configs/pretrain_config_frce_8gpus.yaml \
#--model_config configs/model_configs/7B.json

#MAX_JOBS=32 pip3 install flash-attn --no-build-isolation
# MAX_JOBS=32 python3 setup.py install

pip3 install black flake8 flake8-annotations mypy pre-commit isort pytest pytest-cov pytest-timeout remote-pdb parameterized docutils pynvml scikit-learn pygit2 pgzip

exit;

sbatch --partition=gpu --gres=gpu:8 --time=1800:00:00 --cpus-per-task=32 --mem=250g job.sh