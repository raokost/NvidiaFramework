#!/bin/bash
export CUDA_VER=11.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/deepstream/deepstream/lib/cvcore_libs
python3 new1.py file:///opt/nvidia/deepstream/deepstream-6.1/samples/streams/nosound.mp4
