<div align="center">

<h1><a href="https://ieeexplore.ieee.org/document/10591792">Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis</a></h1>

**[Chenyang Liu](https://chen-yang-liu.github.io/), [Keyan Chen](https://kyanchen.github.io), [Haotian Zhang](https://scholar.google.com/citations?user=c7uR6NUAAAAJ), [Zipeng Qi](https://scholar.google.com/citations?user=KhMtmBsAAAAJ), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), and [Zhenwei Shi*✉](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

<div align="center">
  <img src="resource/Change_Agent.png" width="400"/>
</div>
</div>

> **Fork note:** This is a fork of [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent) that replaces the original [lagent](https://github.com/InternLM/lagent)-based agent with a [Google ADK](https://google.github.io/adk-docs/) + Gemini 2.0 integration. The original README is preserved in [`README_upstream.md`](README_upstream.md).

Official PyTorch implementation of the paper: "**Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis**" in [[IEEE](https://ieeexplore.ieee.org/document/10591792)]  ***(Accepted by IEEE TGRS 2024)***

## Table of Contents
- [Installation](#installation)
- [LEVIR-MCI dataset](#levir-mci-dataset)
- [Training of MCI model](#training-of-the-multi-level-change-interpretation-model)
- [ADK Agent](#adk-agent)
- [Citation](#citation)

## Installation

### Clone the repo

```bash
git clone https://github.com/biplovbhandari/Change-Agent.git
cd Change-Agent
```

### Install uv

Follow the instructions at [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).

### Create environment and install dependencies

```bash
uv sync
source .venv/bin/activate
```

### Download the pretrained model

Download `MCI_model.pth` from [Hugging Face](https://huggingface.co/lcybuaa/Change-Agent/tree/main) and place it in `./src/models_ckpt/`.

## LEVIR-MCI dataset
- Download the LEVIR_MCI dataset: [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main) (**Available Now!**).
- This dataset is an extension of our previously established [LEVIR-CC dataset](https://github.com/Chen-Yang-Liu/RSICC). It contains bi-temporal images as well as diverse change detection masks and descriptive sentences. It provides a crucial data foundation for exploring multi-task learning for change detection and change captioning.
    <br>
    <div align="center">
      <img src="resource/dataset.png" width="800"/>
    </div>
    <br>

### Download and prepare the dataset

```bash
# Install git-lfs if not already installed
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install -y git-lfs
git lfs install

cd src/data

# Clone the dataset
git clone https://huggingface.co/datasets/lcybuaa/LEVIR-MCI
cd LEVIR-MCI
git lfs pull

# Remove the inner repo metadata
rm -rf .git

# Unzip into target folder
unzip LEVIR-MCI-dataset.zip -d ../LEVIR_MCI
cd ../../..
```

### Extract text files

```bash
cd src
python preprocess_data.py --data-path-root="./data/LEVIR_MCI"
```

After that, you can find generated files in `./src/data/LEVIR_MCI/`.

## Training of the multi-level change interpretation model
The overview of the MCI model:
<br>
    <div align="center">
      <img src="resource/MCI_model.png" width="800"/>
    </div>
<br>

### Train
Make sure you performed the data preparation above. Then, start training as follows:
```bash
cd src
python train.py --train_goal 2 --data_folder ./data/LEVIR_MCI/LEVIR-MCI-dataset/images --savepath ./models_ckpt/
```

### Evaluate
```bash
cd src
python test.py --data_folder ./data/LEVIR_MCI/LEVIR-MCI-dataset/images --checkpoint {checkpoint_PATH}
```
We recommend training the model 5 times to get an average score.

### Inference
Run inference to get started as follows:
```bash
cd src
python predict.py --imgA_path {imgA_path} --imgB_path {imgB_path} --mask_save_path ./CDmask.png
```
You can specify a custom checkpoint with `--checkpoint /path/to/model.pth`. Download the pretrained model `MCI_model.pth` from [Hugging Face](https://huggingface.co/lcybuaa/Change-Agent/tree/main) and place it in `./models_ckpt/`.

## ADK Agent

This fork replaces the original lagent-based agent with [Google ADK](https://google.github.io/adk-docs/) + Gemini, providing a conversational interface for change detection and interpretation.

<br>
<div align="center">
      <img src="resource/overview_agent.png" width="800"/>
</div>
<br>

### Setup

1. **Configure environment variables:**

```bash
cp src/adk_app/.env.example src/adk_app/.env
```

Edit `src/adk_app/.env` and set your GCP project:
```env
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
```

2. **Authenticate with Google Cloud:**

```bash
gcloud auth application-default login
```

3. **Download the pretrained model** (if not already done):

Download `MCI_model.pth` from [Hugging Face](https://huggingface.co/lcybuaa/Change-Agent/tree/main) and place it in `./src/models_ckpt/`.

### Run the app

```bash
cd src
streamlit run adk_app/app.py
```

Upload a pair of before/after satellite images (sample images are provided in `resource/sample_images/`) and ask the agent about changes. The agent supports multi-turn conversation:

- *"What changed between these images?"* — runs detection + captioning
- *"How many buildings changed?"* — retrieves cached statistics
- *"Show me just the roads"* — saves a road-only change map

### Configuration reference

All settings are in `src/adk_app/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | — | Your GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | GCP region |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for the agent |
| `GEMINI_MODEL_OPTIONS` | `gemini-2.5-flash,gemini-2.5-pro` | Models shown in sidebar dropdown |
| `MODEL_CHECKPOINT` | `./models_ckpt/MCI_model.pth` | Path to MCI model weights |
| `VOCAB_PATH` | `./data/LEVIR_MCI` | Path to vocabulary/tokens directory |
| `GPU_ID` | `-1` (CPU) | GPU device ID, `-1` for CPU |

## Citation
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{Liu_Change_Agent,
  author={Liu, Chenyang and Chen, Keyan and Zhang, Haotian and Qi, Zipeng and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis},
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Remote sensing;Feature extraction;Semantics;Transformers;Roads;Earth;Task analysis;Interactive Change-Agent;change captioning;change detection;multi-task learning;large language model},
  doi={10.1109/TGRS.2024.3425815}}

```

## Acknowledgement
Thanks to the following repository:

[RSICCformer](https://github.com/Chen-Yang-Liu/RSICC); [Chg2Cap](https://github.com/ShizhenChang/Chg2Cap)

## License
This repo is distributed under [MIT License](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/LICENSE.txt). The code can be used for academic purposes only.
