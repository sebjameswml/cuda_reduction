#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cuda.h>

// Parameters - array size (from rowlen) and features that may need to be tweaked for your GPU type.

// We'll imagine an array of rowlen * rowlen fields
static constexpr int rowlen = 150;
// How many threads per block to specify.
static constexpr int threadsperblock = 512;

// T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
__inline__ __device__ int warpReduceSum (int val)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ int blockReduceSum (int val)
{
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val);          // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }   // Write reduced value to shared memory
    __syncthreads();                    // Wait for all partial reductions
    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) { val = warpReduceSum (val); } // Final reduce within first warp
    return val;
}

__global__ void reduceit (int *in, int* out, int N)
{
    int sum = 0;
    // reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum (sum);
    if (threadIdx.x == 0) { out[blockIdx.x] = sum; }
}

__host__ void shufflesum_gpu_work (int *in, int* out, int N)
{
    int threads = threadsperblock;
    int blocks = min((N + threads - 1) / threads, 1024);
    reduceit<<<blocks, threads>>>(in, out, N);
    reduceit<<<1, 1024>>>(out, out, blocks);
}

__host__ void shufflesum_gpu (int* d_weight_ar, int rowlen, std::vector<int>& r_scanf_ar)
{
    // Build input data for the test
    int arraysz = rowlen * rowlen;

    //int blockspergrid = std::ceil (static_cast<float>(arraysz) / static_cast<float>(threadsperblock));
    // To pad the arrays out to exact number of blocks
    int arrayszplus = arraysz;
    if (arraysz % threadsperblock) { arrayszplus = arraysz + threadsperblock - arraysz % threadsperblock; }

    // scanf_ar is a data structure to hold the final sum.
    int* d_scanf_ar = nullptr;
    cudaMalloc (&d_scanf_ar, arrayszplus * sizeof(int));

    shufflesum_gpu_work (d_weight_ar, d_scanf_ar, arraysz);

    int sum;
    cudaMemcpy(&sum, d_scanf_ar, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "sum is " << sum << std::endl;

    // d_scanf_ar should contain the data
    if (r_scanf_ar.size() != arrayszplus) {
        std::cerr << "Can't fill r_scanf_ar with result :L(\n";
    } else {
        cudaMemcpy (r_scanf_ar.data(), d_scanf_ar, arrayszplus * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

int main()
{
    int arraysz = rowlen * rowlen;
    int arrayszplus = arraysz;

    if (arraysz % threadsperblock) {
        arrayszplus = arraysz + threadsperblock - arraysz % threadsperblock;
    } // else it remains same as arraysz
    std::cout << "arraysz in exact number of blocks (arrayszplus) is " << arrayszplus << std::endl;

    std::vector<int> weight_ar (arrayszplus, 0.0f);

    // Now some non-zero, non-unary weights
    weight_ar[0] = 11;
    weight_ar[1] = 13;
    weight_ar[2] = 12;
    weight_ar[3] = 2;
    weight_ar[4] = 3;

    if (rowlen > 3) {
        weight_ar[12] = 17;
    }
    if (rowlen > 4) {
        weight_ar[22] = 19;
    }
    if (rowlen > 5) {
        weight_ar[33] = 23;
    }
    if (rowlen > 17) {
        weight_ar[44] = 23;
    }
    weight_ar[45] = 23;
    weight_ar[55] = 24;
    weight_ar[63] = 25;
    weight_ar[64] = 26;
    weight_ar[65] = 27;
    weight_ar[77] = 28;
    weight_ar[79] = 29;
    weight_ar[80] = 3.0;
    weight_ar[128] = 23;
    weight_ar[129] = 23;
    weight_ar[130] = 25;
    weight_ar[191] = 26;
    weight_ar[192] = 27;
    weight_ar[193] = 28;
    weight_ar[254] = 29;
    weight_ar[255] = 21;
    weight_ar[256] = 22;
    weight_ar[257] = 23;

    if (rowlen > 149) {
        weight_ar[22486] = 254;
    }

    int cpu_sum = 0;
    for (auto w : weight_ar) { cpu_sum += w; }

    // Copy to GPU memory:
    int* d_weight_ar = nullptr;
    cudaMalloc (&d_weight_ar, arrayszplus * sizeof(int));
    cudaMemcpy (d_weight_ar, weight_ar.data(), arrayszplus * sizeof(int), cudaMemcpyHostToDevice);

    // Call the function:
    std::vector<int> r_scanf_ar(arrayszplus);
    shufflesum_gpu (d_weight_ar, rowlen, r_scanf_ar);

#ifdef DEBUG_ALL_VALUES
    // Print result
    unsigned int j = 0;
    while (j < arraysz) {
        std::cout << "weight_ar[" << j << "] = " << weight_ar[j] << " ... scanf_ar[]=" << r_scanf_ar[j] << std::endl;
        ++j;
    }
#endif

    if (r_scanf_ar[0] != cpu_sum) {
        std::cerr << "FAIL\n";
    }
    std::cout << "Array sum is " << r_scanf_ar[arraysz-1] << ". GPU/CPU method difference is "
              << (r_scanf_ar[arraysz-1] - cpu_sum) << "\n";
    return 0;
}
