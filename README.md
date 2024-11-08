# MIXAD: Memory-Induced Explainable Time Series Anomaly Detection

<div align="center">
    <img src="https://github.com/user-attachments/assets/c0b167a3-b382-45a0-a4b0-3f2ade28bfd4" alt="Image description" width=700"/>
</div>

This repository contains the official implementation for the paper "[MIXAD: Memory-Induced Explainable Time Series Anomaly Detection](https://arxiv.org/abs/2410.22735)" presented at the ICPR 2024.

## Conda Environment Setup
1. Create a new Conda environment:
   ```
   conda create -n {env_name} python=3.8
   ```
2. Install Pytorch:
   ```
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   ```
3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation (SMD, MSDS Datasets)
We use datasets from the [TranAD](https://github.com/imperial-qore/TranAD) repository. Run the `preprocess.py` script in that repository to obtain the final datasets used for experiments.

### Directory Structure:
```
MIXAD
├── data
│   ├── SMD
│   │   ├── machine-1-1_train.npy
│   │   ├── machine-1-1_test.npy
│   │   └── machine-1-1_labels.npy
│   └── MSDS
│       ├── train.npy
│       ├── test.npy
│       └── labels.npy
├── utils
└── main.py
```

## Train & Evaluation

### SMD Dataset
To train and evaluate on the SMD dataset, use:
```
python main.py -dataset "SMD_1_1" -num_nodes 38 -seq_len 30 -horizon 1 -max_diffusion_step 3 -num_rnn_layers 1 -rnn_units 64 -mem_num 5 -mem_dim 64 -lamb_cont 0.01 -lamb_cons 0.1 -lamb_kl 0.0001 -epochs 30 -patience 10 -batch_size 256 -lr 0.001 -n_th_steps 100 -load_dir '' -interpretation 'True' -wandb 'False' -comment '{comment}'
```

### MSDS Dataset
To train and evaluate on the MSDS dataset, use:
```
python main.py -dataset "MSDS" -num_nodes 10 -seq_len 30 -horizon 1 -max_diffusion_step 2 -num_rnn_layers 1 -rnn_units 64 -mem_num 2 -mem_dim 64 -lamb_cont 0.001 -lamb_cons 0.001 -lamb_kl 0.0001 -epochs 30 -patience 10 -batch_size 256 -lr 0.001 -n_th_steps 100 -load_dir '' -interpretation 'True' -wandb 'False' -comment '{comment}'
```

## Evaluation with Pretrained Weights
To use pretrained weights for evaluation, add:
```
-load_dir "./results/{dataset_name}/{directory_containing_pretrained_weight_pth}"
```
Logs will be saved under this directory.

## Results
Logs and evaluation results are stored in:
```
MIXAD/results/{dataset_name}/{%m-%d-%HH%MM%Ss}_{comment}/
```
Generated files:
  - `best.pth`: Best model weights saved during training.
  - `results.txt`: Contains argument settings, data configuration, model configuration, training logs, and evaluation results, including detection and interpretation results.
