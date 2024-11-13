#!/bin/sh

nvcc -o shfl_sum_float shfl_sum_float.cu  && ./shfl_sum_float
