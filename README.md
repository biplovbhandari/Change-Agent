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
You can modify `--checkpoint` of `Change_Perception.define_args()` in `predict.py`. Then you can use your own model, or download the pretrained model `MCI_model.pth` from [Hugging Face](https://huggingface.co/lcybuaa/Change-Agent/tree/main) and place it in `./models_ckpt/`.

## ADK Agent

<!-- TODO: Add ADK agent setup and usage instructions once src/adk_app/ is added -->

Coming soon — Google ADK + Gemini 2.0 agent with Streamlit chat UI for interactive change interpretation.

<br>
<div align="center">
      <img src="resource/overview_agent.png" width="800"/>
</div>

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
