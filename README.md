# splatting

This is a reimplementation of [softmax-splatting](https://github.com/sniklaus/softmax-splatting), which can be executed on cpu only devices.

## setup

This impolementation requires [PyTorch](https://pytorch.org/get-started/locally/) to be installed. Once PyTorch is installed, the package can be compiled with
```pip install .``` or ```pip install -e .```

## references
```
[1]  @inproceedings{Niklaus_CVPR_2020,
         author = {Simon Niklaus and Feng Liu},
         title = {Softmax Splatting for Video Frame Interpolation},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2020}
     }
```

## test status
[![Coverage Status](https://coveralls.io/repos/github/hperrot/splatting/badge.svg)](https://coveralls.io/github/hperrot/splatting)
