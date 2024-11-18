#!/bin/sh
#-ccbin /usr/bin/cc -m64 --std c++11 -DBUFFER_TYPE_GL_INTEROP -Xcompiler ,\"-fPIC\",\"-msse\",\"-msse2\",\"-msse3\",\"-mfpmath=sse\",\"-O3\",\"-DNDEBUG\",\"-g3\",\"-funroll-loops\" -arch sm_60 --use_fast_math -lineinfo

# Similar to Optix build process:

# Compile step
##nvcc -c -o shfl_sum_arrays_of_float3.o -ccbin /usr/bin/cc -m64 --std c++11 -DBUFFER_TYPE_GL_INTEROP -Xcompiler ,\"-fPIC\",\"-msse\",\"-msse2\",\"-msse3\",\"-mfpmath=sse\",\"-O3\",\"-DNDEBUG\",\"-g3\",\"-funroll-loops\" -arch sm_60 --use_fast_math -lineinfo shfl_sum_arrays_of_float3.cu

# Link step
##/usr/bin/c++ -std=c++11 -fPIC -msse -msse2 -msse3 -mfpmath=sse -O3 -DNDEBUG -g3 -funroll-loops shfl_sum_arrays_of_float3.o -o shfl_sum_arrays_of_float3 -lcuda -lm /usr/lib/x86_64-linux-gnu/libcudart_static.a && ./shfl_sum_arrays_of_float3



# Simple:
nvcc -o shfl_sum_arrays_of_float3 shfl_sum_arrays_of_float3.cu  && ./shfl_sum_arrays_of_float3
