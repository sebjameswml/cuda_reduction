#!/bin/sh

nvcc -o shfl_sum_float3 shfl_sum_float3.cu  && time ./shfl_sum_float3
