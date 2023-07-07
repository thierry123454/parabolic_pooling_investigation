from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='morphological_torch',
    ext_modules=[
            CUDAExtension(name="morph_cuda",
                          sources=["src/cuda/morphology.cpp",
                                   # Pooling operations.
                                   "src/cuda/operations/pool_forward.cu",
                                   "src/cuda/operations/pool_backward_f.cu",
                                   "src/cuda/operations/pool_backward_h.cu",
                                   # Unpooling operations.
                                   "src/cuda/operations/unpool_forward.cu",
                                   "src/cuda/operations/unpool_backward_f.cu"],
                          extra_compile_args=['-g'])
        ],
    cmdclass={
        "build_ext": BuildExtension
    },
    package_dir={'': 'src'},
    packages=find_packages('src'),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    version='0.1.0',
    author='Rick Groenendijk',
    author_email='r.w.groenendijk@uva.nl',
    license='LICENSE',
    description='A package that bundles morphological operations for PyTorch.',
    long_description=open('README.md').read(),
    install_requires=[
       "setuptools>=61.0",
       "torch==1.11.0"],
)