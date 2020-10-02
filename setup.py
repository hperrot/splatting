from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='splatting_cpp',
    ext_modules=[cpp_extension.CppExtension('splatting_cpp', ['src/splatting.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
