from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemm_kernels",
    ext_modules=[
        CUDAExtension(
            "gemm_kernels",
            [
                "gemm_kernels.cpp",
                "gemm_kernels_cuda.cu",
            ],
            libraries=["cuda"]
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
