"""
Build configuration for the NIMKI C++ extension.

Project metadata lives in pyproject.toml; this file exists solely to declare
`ext_modules`, which pyproject.toml cannot express for setuptools.

Without this, `pip install .` produced a pure-Python catboss with no compiled
extension, and every NIMKI run failed at import with
"No module named '_nami_core'" - silently, because the callers ignored it.

The extension is declared as `catboss.nimki._nimki_core` so it is installed
*inside* the package and resolves via `from . import _nimki_core`. Building it
as a bare top-level `_nimki_core` only works for an in-place build run from the
package directory, which is what previously masked the problem.
"""

import os

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:  # pragma: no cover
    raise SystemExit(
        "pybind11 is required to build catboss.\n"
        "It is declared in [build-system].requires, so `pip install .` handles "
        "it automatically. For a manual build: pip install pybind11"
    )

HERE = os.path.dirname(os.path.abspath(__file__))

# setuptools requires forward-slash paths RELATIVE to this file. Absolute paths
# survive `build_ext --inplace` but make wheel building fail outright with
# "setup script specifies an absolute path" - which is exactly how a broken
# build can look fine locally and then break `pip install .`.
CPP_DIR = "src/catboss/nimki/cpp"

SOURCES = [
    f"{CPP_DIR}/{name}"
    for name in (
        "nimki_core.cpp",
        "gabor_fit.cpp",
        "uv_calc.cpp",
        "data_collection.cpp",
        "outlier_detection.cpp",
    )
]

missing = [s for s in SOURCES if not os.path.exists(os.path.join(HERE, s))]
if missing:
    raise SystemExit(
        "Cannot build the NIMKI extension, missing C++ sources:\n  "
        + "\n  ".join(missing)
    )

# -march is opt-in. Defaulting to `native` bakes the build machine's
# instruction set into the binary, which is fine locally but crashes with
# SIGILL on any older CPU - a real hazard for container images built on one
# machine and run across a heterogeneous cluster.
#   CATBOSS_MARCH=native   -> fastest, this machine only
#   CATBOSS_MARCH=x86-64-v3 -> portable across most modern x86
march = os.environ.get("CATBOSS_MARCH", "").strip()

compile_args = ["-O3", "-ffast-math", "-fopenmp"]
if march:
    compile_args.append(f"-march={march}")

ext_modules = [
    Pybind11Extension(
        "catboss.nimki._nimki_core",
        sources=sorted(SOURCES),
        include_dirs=[CPP_DIR],
        extra_compile_args=compile_args,
        extra_link_args=["-fopenmp"],
        language="c++",
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
