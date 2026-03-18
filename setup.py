#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization_wenqi_tam",
    packages=['diff_gaussian_rasterization_wenqi_tam'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_wenqi_tam._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_rasterizer/adam.cu",
            "rasterize_points.cu",
            "conv.cu",
            "merge_blocks.cu",
            "cpu_adam.cpp",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": ["-Xcompiler", "-fno-gnu-unique", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],
                "cxx": ["-O3", "-fopenmp", "-std=c++17", "-march=native"]
            },
            # -fopenmp links libgomp by default, matching PyTorch's runtime.
            # Do NOT use -lomp (LLVM libomp) — dual OpenMP runtime causes heap corruption.
            extra_link_args=["-fopenmp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
