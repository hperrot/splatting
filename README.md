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

## usage

If you installed the extension with pip, you can simply import it:

```
from splatting import Splatting
```

If you choose to JIT compile it, you can add this repository to your project structure like

```
└── your_code
    ├── file1.py
    ├── file2.py
    └── utils
        └── splatting  // this repository
```

you can import like this:

```
from your_code.utils.splatting import Splatting
```

The splatting function can be used like this:

```
import torch
from splatting import Splatting, splatting_function

frame = torch.ones([1, 3, 4, 4])
flow = torch.ones([1, 2, 4, 4])

# use of the slatting function
output = splatting_function("average", frame, flow)

# use of the splatting module
output = Splatting("average")(frame, flow)
```

## references

```
[1]  @inproceedings{Niklaus_CVPR_2020,
         author = {Simon Niklaus and Feng Liu},
         title = {Softmax Splatting for Video Frame Interpolation},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2020}
     }
```
