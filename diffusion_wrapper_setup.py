
import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class get_pybind_include(object):
    """defer pybind11 import"""

    def __str__(self):
        import pybind11
        return pybind11.get_include()


extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    extra_compile_args = [
        "/std:c++17",
        "/O2",
        "/EHsc",
        "/W3",
        "/D_SILENCE_ALGORITHM_DEPRECATION_WARNING",
    ]
    extra_link_args = []


else:
    extra_compile_args = [
        "-std=c++17", "-O3", "-Wall", "-Wextra",
        "-march=native", "-fopenmp"
    ]
    extra_link_args = ["-fopenmp"]


eigen_path = os.path.join(
    os.environ["CONDA_PREFIX"],
    "Library" if sys.platform == "win32" else "",
    "include", "eigen3"
)

ext_modules = [
    Extension(
        "diffusion",
        ["diffusion_wrapper.cpp", "diffusion.cpp"],
        include_dirs=[get_pybind_include(), eigen_path],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="diffusion",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
