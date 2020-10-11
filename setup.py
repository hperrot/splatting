from setuptools import setup
from torch.utils import cpp_extension
import os
import glob


ext_modules = [
    cpp_extension.CppExtension(
        "splatting.cpu",
        ["cpp/splatting.cpp"],
    ),
]


cublas_include_paths = glob.glob("/usr/local/**/cublas_v2.h", recursive=True)
if len(cublas_include_paths) > 0:
    ext_modules.append(
        cpp_extension.CUDAExtension(
            "splatting.cuda",
            ['cuda/splatting_cuda.cpp', 'cuda/splatting.cu'],
            include_dirs=[os.path.dirname(cublas_include_paths[0])],
        ),
    )


setup(
    name="splatting",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    install_requires=["torch"],
    extras_require={
        "dev": ["pytest", "pytest-cov", "pre-commit"]
    },  # pip install -e '.[dev]'
)
