# Masked Autoregressive Language Models

Official implementation of ["Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs"](https://arxiv.org/abs/2510.09885), which 1) trains LLMs on new data without the reversal curse and more data-efficiently!

## Overview

This repository contains implementations of several language model training approaches:
- **AR (Autoregressive)**: Standard autoregressive language modeling
- **dLLM (Diffusion LLM)**: Masked diffusion language model
- **Masked AR**: Masked autoregressive training
- **Random Augmented AR**: Autoregressive with random augmentation

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU(s)
- PyTorch 2.8.0+
- 4 GPUs recommended for distributed training

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/xup5/masked_arLLM.git
cd masked_arLLM
```

2. **Create conda environment:**
```bash
conda env create -f environment.yaml
conda activate masked_arllm
```

Alternatively, you can install dependencies manually:
```bash
conda create -n masked_arllm python=3.10
conda activate masked_arllm
pip install torch==2.8.0 transformers==4.56.1 accelerate==1.10.1 flash-attn==2.8.3
pip install wandb datasets evaluate lm-eval rouge-score
```

### Dataset Configuration

The code expects datasets in the `datasets/` directory. You can customize the dataset location by setting the environment variable:
```bash
export MASKED_ARLLM_DATASETS_DIR=/path/to/your/datasets
```

The provided datasets include:
- **Wiki**: Wikipedia-based dataset
- **Biography**: Biography dataset
- **ND**: Name-Description pairs

Each dataset has three variants:
- `no_paraphrase`: Original data without augmentation
- `paraphrase_same_order`: Paraphrased data maintaining order
- `paraphrase_change_order`: Paraphrased data with changed order

## Usage

### Running Experiments

Training scripts are provided in the `scripts/` directory. Run them from the `scripts/` directory:

```bash
cd scripts
```

#### 1. Standard Autoregressive Training
```bash
bash ar.sh
```

#### 2. Diffusion LLM Training
```bash
bash dllm.sh
```

#### 3. Masked Autoregressive Training
```bash
bash masked_ar.sh
```

#### 4. Random Augmented AR Training
```bash
bash random_ar.sh
```

#### 5. Experiments with different temperature (t) values
```bash
bash dllm_t.sh        # dLLM with various t values
bash masked_ar_t.sh   # Masked AR with various t values
```

### Customizing Training

You can modify the training scripts or run individual training commands directly:

```bash
cd scripts
torchrun --nproc_per_node=4 ../src/ar.py \
  --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset_name wiki \
  --paraphrases "same_order" \
  --num_train_epochs 512 \
  --eval_epoch 0 1 2 4 8 16 32 64 128 256 512 \
  --save_checkpoint 1 \
  --save_path ./results/ar/
```

### Key Arguments

- `--model_name`: Hugging Face model identifier (default: "meta-llama/Meta-Llama-3.1-8B-Instruct")
- `--dataset_name`: Dataset to use (`wiki`, `biography`, or `ND`)
- `--paraphrases`: Paraphrase mode (`None`, `same_order`, or `change_order`)
- `--num_train_epochs`: Number of training epochs
- `--eval_epoch`: List of epochs at which to evaluate
- `--save_checkpoint`: Whether to save checkpoints (0 or 1)
- `--save_path`: Directory to save results and checkpoints
- `--t_mode`: Temperature mode for diffusion models (`fixed` or `random`)
- `--t`: Temperature value (for fixed mode)

## Project Structure

```
masked_arLLM/
├── src/
│   ├── ar.py                      # Autoregressive training
│   ├── dllm.py                    # Diffusion LLM training
│   ├── masked_ar.py               # Masked autoregressive training
│   ├── random_augmented_ar.py     # Random augmented AR training
│   ├── dataset_util.py            # Dataset loading utilities
│   ├── eval_util.py               # Evaluation utilities
│   ├── fsdp2_util.py              # FSDP2 utilities
│   └── inference.py               # Inference utilities
├── scripts/
│   ├── ar.sh                      # AR training scripts
│   ├── dllm.sh                    # dLLM training scripts
│   ├── dllm_t.sh                  # dLLM with temperature sweep
│   ├── masked_ar.sh               # Masked AR training scripts
│   ├── masked_ar_t.sh             # Masked AR with temperature sweep
│   └── random_ar.sh               # Random augmented AR scripts
├── datasets/                      # Training and test datasets
├── environment.yaml               # Conda environment specification
├── pyproject.toml                # Python project configuration
└── README.md                     # This file
```

## Distributed Training

The code uses PyTorch FSDP2 (Fully Sharded Data Parallel 2) for efficient distributed training. Training scripts are configured to use 4 GPUs by default via `torchrun --nproc_per_node=4`.

To adjust the number of GPUs:
```bash
torchrun --nproc_per_node=<NUM_GPUS> ../src/ar.py [arguments]
```

## Logging and Monitoring

The code supports Weights & Biases (wandb) for experiment tracking. To use wandb:

1. Install wandb (already included in environment.yaml)
2. Login to wandb: `wandb login`
3. The training scripts will automatically log metrics, losses, and evaluation results

## Results

Training results and checkpoints are saved to `./results/` by default:
- `./results/ar/` - Autoregressive results
- `./results/dllm/` - Diffusion LLM results
- `./results/masked_ar/` - Masked AR results
- `./results/random_ar/` - Random augmented AR results

Each result file contains:
- Training metrics and losses
- Evaluation results (recall scores per task/group)
- Model configuration and hyperparameters

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pan2025closingdataefficiencygapautoregressive,
      title={Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs}, 
      author={Xu Pan and Ely Hahami and Jingxuan Fan and Ziqian Xie and Haim Sompolinsky},
      year={2025},
      eprint={2510.09885},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.09885}, 
}
```
