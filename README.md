# PyramidEE: An Early\-exit\-aware Transformer Architecture

## 1\. Introduction

PyramidEE is an early\-exit\-aware Transformer architecture designed to accelerate both training and inference of large language models and improve model performance\. It introduces pyramid\-structured early\-exit branches and optional skip\-layer mechanisms to reduce computational cost while maintaining competitive model performance\.
The pre‑trained model weights are available for download via the link below:

Link: [https://pan.baidu.com/s/1PjOPLGfXxliNA83zX25quQ?pwd=c3rv](https://pan.baidu.com/s/1PjOPLGfXxliNA83zX25quQ?pwd=c3rv)
Extraction code: c3rv


## 2\. Environment Setup

Install the required dependencies before training\.

**Requirements**

```Plain Text
torch>=2.1.0
transformers>=4.38.0
accelerate>=0.27.0
datasets>=2.17.0
numpy>=1.24
pyyaml
```

**Installation**

```Plain Text
pip install -r requirements.txt
```



## 3. Data Preparation
Download your raw pre‑training dataset manually. The input data should be formatted as JSON lines, where each sample contains a `text` field storing the input string.

Example data sample:
```json
{"text": "your pre‑training text content here ..."}
```

Specify your dataset file path via the `data_path` entry in `config.yaml`. The preprocessing script will automatically split the dataset, cache processed data and generate a hash code. Manually assign this generated hash code to the `hash_code` field in `config.yaml` before training.



## 4\. Model Training

All core hyperparameters and model architectures are configured in `config.yaml`\. No code modification is required for switching different model variants\.

To start pre\-training:

```Plain Text
cd trainer
torchrun --nproc_per_node=8 train_pretrain5.py
```




### Supported Model Variants

You can switch among the following architectures by setting the `model_class_path` field in `config.yaml` for ablation studies: 



- `Base`: Vanilla Transformer baseline without early\-exit and skip\-layer designs\.

- `EE_MLP_linear`: The proposed PyramidEE architecture with linear early\-exit MLP branches\.

- `EE_MLP_same`: Early\-exit MLP branches with the same hidden dimension as the Transformer backbone\.

- `EE_MLP_linear_Skip_Layer`: Enhanced PyramidEE model combining linear early\-exit branches and skip\-layer mechanism\.

- `EE_MLP_same_Skip_Layer`: Same\-dimension early\-exit MLP combined with skip\-layer design\.

- `Skip_Layer`: Skip\-layer\-only baseline without early\-exit branches\.

Before running training commands, please specify your target model variant in the configuration file\.

### Configuring PyramidEE Instantiations
Different instantiations of PyramidEE can be specified via the `mlp_config_path` field in `config.yaml`.
Available MLP options include `mlp_pow`, `mlp_cos`, `mlp_linear`.



## Acknowledgements
This work is built upon the **MiniMind** project. We thank the authors for their open‑source contributions.
Official repository: https://github.com/jingyaogong/minimind

