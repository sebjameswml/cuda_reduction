#!/bin/sh

nvcc -o shfl_sum_int shfl_sum_int.cu  && ./shfl_sum_int
