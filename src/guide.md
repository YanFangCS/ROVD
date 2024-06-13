# Label-efficient variant of iCaRL

This is implementation of our proposed iCaRL-SSL in our paper, also called Label-efficient iCaRL.

We mainly implement this based on code of unbiased teacher officially released by its authors, which can be easily found in github.

# Installation

## Prerequisites

- Linux with Python == 3.9
- PyTorch >= 1.9.0 and torchvision that matches the PyTorch installation.

## Install PyTorch in Conda env

```shell
# create conda env
conda create -n detectron2 python=3.9
# activate the enviorment
conda activate detectron2
# install PyTorch >=1.9.0 with GPU
conda install pytorch torchvision -c pytorch
```

## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset download

We haven't fully prepared to release the full ROVD datasets, it needs some time for us to finish public-ready work.

## Training


```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/ROVD/faster_rcnn_R_50_FPN.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8
```


## Evaluation

```shell
python train_net.py \
      --eval-only \
      --num-gpus 4 \
      --config onfigs/ROVD/faster_rcnn_R_50_FPN.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8 MODEL.WEIGHTS <model weight>.pth
```