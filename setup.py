from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="chap3_kernels",
    ext_modules=[
        CUDAExtension(
            "chap3_kernels",
            [
                "chap3_kernels.cpp",
                "chap3_kernels_cuda.cu",
            ],
            libraries=["cuda"]
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
