# Imagic: Text-Based Real Image Editing with Diffusion Models

This repository contains an unofficial implementation of Google's paper **Imagic: Text-Based Real Image Editing with Diffusion Models**. 
The goal of this project is to edits a single real-world image using a target text prompt.



## Installation

1. Clone the repository:

```shell
git clone --recurse-submodules https://github.com/sangminkim-99/Imagic.git
cd Imagic
```

2. Create and activate a new Conda environment:

```shell
conda create -n imagic python=3.9
conda activate imagic
```

3. Install the necessary dependencies:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # change to your own version of torch
pip install -r requirements.txt
```

## Usage

- Show helper messages for all possible commands

```shell
python app.py --help
```

- Train Latent Edge Predictor

Currently supports `--batch-size 1` only.

```shell
python app.py train-lep
```

- Sample image with Latent Edge Predictor

```shell
python app.py sample --sketch-file-path {PATH} --prompt {PROMPT}
```

- Gradio web demo (_debugging_)

```shell
python app.py demo
```


## TODOs

- [ ] Reproduce the bicycle example

- [ ] Upload pretrained LEP


## Acknowledgments

We would like to express our gratitude to the authors of the original paper and the developers of the referenced repositories for their valuable contributions, which served as the foundation for this implementation.


## Disclaimer

This is an unofficial implementation and is not affiliated with Google or the authors of the original paper.