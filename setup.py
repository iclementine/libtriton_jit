from skbuild import setup

# When building extension modules `cmake_install_dir` should always be set to the
# location of the package you are building extension modules for.
# TODO: change the package dir to src when merging with flag_gems
setup(
    name="flaggems",
    version="0.1.0",
    packages=["flaggems"],
    package_dir={"": "python"},
    cmake_source_dir=".",
    cmake_install_dir="python/flaggems",
)
