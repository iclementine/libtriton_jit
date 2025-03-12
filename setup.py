from pathlib import Path

from skbuild import setup

cmake_source_dir = Path(__file__).parent.absolute().resolve()
setup(
    name="flaggems",
    version="0.1.0",
    packages=["flaggems"],
    package_dir={"": "python"},
    cmake_source_dir=".",
    cmake_install_dir="python/flaggems",
)

# When building extension modules `cmake_install_dir` should always be set to the
# location of the package you are building extension modules for.
# Specifying the installation directory in the CMakeLists subtley breaks the relative
# paths in the helloTargets.cmake file to all of the library components.
