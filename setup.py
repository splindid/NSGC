# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified for Windows compatibility and CUDA 12.x support without apex

import glob
import os
import platform

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    # Windows 特殊处理
    is_windows = platform.system() == "Windows"
    
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        
        if is_windows:
            # Windows CUDA 编译参数
            extra_compile_args["cxx"] = ["/MD", "/wd4819", "/wd4251", "/wd4244", "/wd4267", "/wd4275", "/wd4018", "/wd4190"]
            extra_compile_args["nvcc"] = [
                "-O3",
                "--expt-relaxed-constexpr",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
            
            # CUDA 12.x 需要的额外参数
            cuda_version = torch.version.cuda
            if cuda_version and int(cuda_version.split('.')[0]) >= 12:
                extra_compile_args["nvcc"].extend([
                    "-allow-unsupported-compiler",
                ])
        else:
            # Linux 编译参数
            extra_compile_args["cxx"] = ["-O3"]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]

    sources = [os.path.join(extensions_dir, s) if not os.path.isabs(s) else s for s in sources]
    
    # 修复源文件路径
    sources_fixed = []
    for s in sources:
        if os.path.exists(s):
            sources_fixed.append(s)
        else:
            # 尝试相对路径
            rel_path = os.path.relpath(s, this_dir)
            if os.path.exists(rel_path):
                sources_fixed.append(rel_path)
    
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources_fixed if sources_fixed else sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="KaihuaTang",
    url="https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch",
    description="Scene Graph Benchmark in PyTorch (Windows Compatible, No Apex)",
    packages=find_packages(exclude=("configs", "tests",)),
    # 如果编译C++扩展失败，可以注释掉下面这行来跳过
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
)