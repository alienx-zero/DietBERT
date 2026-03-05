DietBERT

Official implementation of DietBERT, a multimodal transformer framework that learns dietary representations from free-text diet logs and structured nutrient records to predict immune, metabolic, cognitive, and mental health outcomes.

This repository contains code for:

multimodal diet representation learning

self-supervised pretraining

downstream prediction tasks

cross-cohort evaluation

Overview

DietBERT integrates two complementary dietary data modalities:

Daily diet logs (free text)

Structured nutrient variables

Both modalities are transformed into unified natural-language representations and encoded using pretrained transformer models. Cross-modal fusion enables the model to learn latent dietary patterns associated with biological and behavioral phenotypes.

The model is evaluated across multiple cohorts including:

NHANES

UCLA-omics

AI4Food

Model Architecture

DietBERT consists of three main components:

Diet log encoder
Encodes free-text daily diet logs using a pretrained DistilBERT backbone.

Nutrient encoder
Converts structured nutrient variables into natural-language descriptions and encodes them with a transformer model.

Cross-modal fusion module
Combines diet-log and nutrient embeddings using a FiLM-attention mechanism to capture interactions between dietary behavior and nutrient composition.

The model is trained using a combination of:

contrastive learning

masked language modeling

supervised regression tasks

Repository Structure
DietBERT/
│
├── data/
│   ├── raw/                # Raw cohort datasets
│   ├── processed/          # Preprocessed data
│
├── models/
│   ├── dietbert.py         # Main model architecture
│   ├── encoders.py         # Diet log and nutrient encoders
│   ├── fusion.py           # Multimodal fusion modules
│
├── training/
│   ├── pretrain.py         # Self-supervised pretraining
│   ├── finetune.py         # Downstream prediction tasks
│
├── datasets/
│   ├── dataset.py          # Dataset loader
│   ├── collator.py         # Hierarchical batch collator
│
├── evaluation/
│   ├── metrics.py          # Evaluation metrics
│   ├── interpretability.py # PLS analysis
│
├── utils/
│   ├── preprocessing.py
│   ├── augmentation.py
│
├── configs/
│   ├── pretrain.yaml
│   ├── finetune.yaml
│
├── figures/
│   ├── architecture.png
│
├── requirements.txt
├── train.sh
├── README.md
Installation

Clone the repository:

git clone https://github.com/yourusername/DietBERT.git
cd DietBERT

Create environment:

conda create -n dietbert python=3.10
conda activate dietbert

Install dependencies:

pip install -r requirements.txt
Data Preparation

Prepare datasets before training.

Example directory structure:

data/
    NHANES/
    UCLA_omics/
    AI4Food/

Run preprocessing:

python utils/preprocessing.py

This step:

formats diet logs

converts nutrient tables into text

prepares training datasets

Pretraining

Self-supervised pretraining learns general dietary representations.

Run:

python training/pretrain.py \
    --config configs/pretrain.yaml

Training objectives include:

contrastive learning between augmented diet logs

masked language modeling on nutrient descriptions

Fine-tuning

Fine-tune the pretrained model for specific prediction tasks.

Example:

python training/finetune.py \
    --task depression \
    --dataset NHANES

Supported prediction tasks include:

depression

anxiety

cognitive performance

inflammatory biomarkers

metabolic measures

Evaluation

Evaluate model performance using:

R²

Pearson correlation

Example:

python evaluation/evaluate.py
Interpretability Analysis

DietBERT embeddings can be interpreted using Partial Least Squares (PLS) regression to identify nutrient axes associated with predicted outcomes.

Run:

python evaluation/interpretability.py
Example Results

DietBERT demonstrates improved predictive performance compared with classical regression baselines across multiple cohorts.

Example performance (NHANES):

Outcome	R²
Depression	0.17
Anxiety	0.19
CRP	0.20
Fasting Glucose	0.32
Citation

If you use this code, please cite:

Zhao Z. et al.

DietBERT: A multimodal transformer framework linking real-world dietary behavior to immune, metabolic, and mental phenotypes.

(Under review)
License

This project is released under the MIT License.

Acknowledgements

This work uses data from:

NHANES

UCLA-omics

AI4Food

We thank collaborators and contributors for their support.
