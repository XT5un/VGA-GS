#!/bin/bash

# download the glm and check out the correct version
git submodule update --init --recursive
cd third_party/glm
git checkout 5c46b9c0
cd ../..

# install the different rasterizers
pip install Rasterizers/diff-gaussian-rasterization-depth
pip install Rasterizers/diff-gaussian-rasterization-origin
pip install Rasterizers/diff-gaussian-rasterization-2dgs
pip install Rasterizers/diff-gaussian-rasterization-RaDe

# test the installation
python test.py