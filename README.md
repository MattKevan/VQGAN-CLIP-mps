# VQGAN+CLIP mps

An implementation of VQGAN+CLIP by Katherine Crowson, using MPS and Metal instead of CUDA.

Performace is roughly ~100 sec/it on a 32gb M1 Pro for a 512x512 image.

## Setup

Create a new virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Clone required repositories:

```
git clone 'https://github.com/MattKevan/VQGAN-CLIP-mps.git'
cd VQGAN-CLIP-mps
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'
```

Install requirements:

```
pip install -r requirements.txt
```

Download the checkpoints:

```
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

## Run

```
python main.py --prompts "a portrait of the insect king | by johannes vermeer"
```
