# splatting

This is a reimplementation of [softmax-splatting](https://github.com/sniklaus/softmax-splatting), which can also be executed on CPU only devices.

## test status

[![Coverage Status](https://coveralls.io/repos/github/hperrot/splatting/badge.svg)](https://coveralls.io/github/hperrot/splatting)

## setup

This implementation requires [PyTorch](https://pytorch.org/get-started/locally/) to be installed.
Once PyTorch is installed, the package can be compiled with
```pip install .``` or ```pip install -e .```.

Alternatively, the extension can be compiled the first time you import it<!-- ([see here](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#building-with-jit-compilation)) -->.
Therefore [ninja](https://ninja-build.org/) needs to be installed, for example with ```pip install ninja```.
Then just import the module in your python script.
The first time you import it, the extension will be compiled.
The second time, it does not need to compile it anymore.

## references

```
[1]  @inproceedings{Niklaus_CVPR_2020,
         author = {Simon Niklaus and Feng Liu},
         title = {Softmax Splatting for Video Frame Interpolation},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2020}
     }
```
