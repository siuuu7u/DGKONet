# DGKONet

## Prerequisites
The following packages are required to run the scripts:

- Python >= 3.8
- PyTorch >= 1.6
- Torchvision
- Sentence-Transformers
- FAISS-GPU
- pandas
- scikit-learn
- dashscope
- modelscope
- SWIFT (from ModelScope)

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our project.

- For IU X-Ray, you can download the dataset from [here](https://openi.nlm.nih.gov/faq).
- For MIMIC-CXR, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/).
Our experiments on IU X-Ray were done on a machine with 1x3090 GPU.

Our experiments on MIMIC-CXR were done on a machine with 1x3090 GPU.

## Fine-tuning
We fine-tune the vision-language model (Qwen2-VL) using the **SWIFT** framework developed by ModelScope.

- Fine-tuning paradigm: **LoRA** (Low-Rank Adaptation)
- Efficient tuning for large vision-language models
- Freezing the backbone model while only tuning low-rank matrices
- Stable training and fast convergence

All fine-tuning pipelines are built upon the SWIFT framework for reproducibility and efficiency.

## Inference Pipeline
The inference process strictly follows **three sequential stages** using the corresponding scripts:

### Stage 1: Report Generation
Run the report generation script to produce the initial radiology reports.

- IU X-Ray: `01_generate_iu.py`
- MIMIC-CXR: `01_generate_mimic.py`

This stage outputs the raw generated reports based on visual inputs.

### Stage 2: Similar Report Retrieval
Retrieve clinically similar reports from the training set to support report optimization.

- IU X-Ray: `02_retrieve_iu.py`
- MIMIC-CXR: `02_retrieve_mimic.py`

Embedding models and FAISS indexing are used for efficient similarity matching.

### Stage 3: Report Optimization
Optimize the initial reports using retrieved references and diagnostic labels.

- IU X-Ray: `03_optimize_iu.py`
- MIMIC-CXR: `03_optimize_mimic.py`

Large language models (via DashScope API) refine expressions, standardize terminology, and emphasize key medical labels.

## Pseudo Label Generation
You can generate pseudo labels for each dataset by leveraging the automatic labeler [ChexBert](https://github.com/stanfordmlgroup/CheXbert).

These labels serve as MeSH-style diagnostic guidance for the report optimization stage.

## Acknowledgment
Our project references the codes in the following repos. Thanks for their works and sharing.
- [SWIFT](https://github.com/modelscope/ms-swift)
