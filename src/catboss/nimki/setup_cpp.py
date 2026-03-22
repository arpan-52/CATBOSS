"""
Build script for NIMKI C++ extension.

Usage:
    cd src/catboss/nimki
    pip install pybind11
    python setup_cpp.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, Extension

try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("pybind11 is required. Install with: pip install pybind11")
    sys.exit(1)


# Get the directory containing this script
here = os.path.dirname(os.path.abspath(__file__))
cpp_dir = os.path.join(here, "cpp")

# Source files
sources = [
    os.path.join(cpp_dir, "nami_core.cpp"),
    os.path.join(cpp_dir, "gabor_fit.cpp"),
    os.path.join(cpp_dir, "uv_calc.cpp"),
    os.path.join(cpp_dir, "data_collection.cpp"),
    os.path.join(cpp_dir, "outlier_detection.cpp"),
]

# Check all sources exist
for src in sources:
    if not os.path.exists(src):
        print(f"Warning: Source file not found: {src}")

# Extension module
ext_modules = [
    Pybind11Extension(
        "_nami_core",
        sources=sources,
        include_dirs=[cpp_dir],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="_nami_core",
    version="1.0.0",
    author="Arpan Pal",
    author_email="arpanpal@ncra.tifr.res.in",
    description="NIMKI C++ core functions",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
