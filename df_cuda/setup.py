from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name = "df_cuda",
    packages=['df_cuda'],
    ext_modules=[
        CUDAExtension(
            name = "df_cuda._C",
            sources = [
                "ext.cpp",
                "df_cuda.cu"
            ]
        )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    }
)
