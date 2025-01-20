# libtorch_example
use libtorch to do interesting things

## how to build

1. activate a python virtual environment where torch is installed
2. mkdir build & cd build
3. cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..


