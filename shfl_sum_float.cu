#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cuda.h>

// Parameters

// How many thread blocks will we sum up?
static constexpr int arrayblocks = 1024;
// How many threads per block to specify.
static constexpr int threadsperblock = 512;

// T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
__inline__ __device__ float warpReduceSum (float val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum (float val)
{
    static __shared__ float shared[32];    // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val);             // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }  // Write reduced value to shared memory
    __syncthreads();                       // Wait for all partial reductions
    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) { val = warpReduceSum (val); } // Final reduce within first warp
    return val;
}

__global__ void reduceit (float *in, float* out, int N)
{
    float sum = 0.0f;
    // reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum (sum);
    if (threadIdx.x == 0) { out[blockIdx.x] = sum; }
}

__host__ void shufflesum_gpu_work (float *in, float* out, int N)
{
    int blocks = min((N + threadsperblock - 1) / threadsperblock, 1024);
    reduceit<<<blocks, threadsperblock>>>(in, out, N);
    reduceit<<<1, 1024>>>(out, out, blocks);
}

__host__ float shufflesum_gpu (float* d_weight_ar, int arraysz)
{
    float* d_scanf_ar = nullptr;
    cudaMalloc (&d_scanf_ar, arraysz * sizeof(float));
    shufflesum_gpu_work (d_weight_ar, d_scanf_ar, arraysz);
    float sum;
    cudaMemcpy(&sum, d_scanf_ar, sizeof(float), cudaMemcpyDeviceToHost);
    return sum;
}

int main()
{
    int arraysz = arrayblocks * threadsperblock;
    std::cout << "Array size is " << arraysz << std::endl;
    std::cout << "arraysz * epsilon is " << arraysz * std::numeric_limits<float>::epsilon() << std::endl;

    std::vector<float> weight_ar (arraysz, 0.0f);

    // Now some non-zero, non-unary weights
    for (int i = 0; i < 1024/*arraysz*/; ++i) {
        if (i % 2 == 0) {
            weight_ar[i] = 1.5f;
        } else if ((i % 3) == 0) {
            weight_ar[i] = 2.4f;
        } else if ((i % 4) == 0) {
            weight_ar[i] = -3.5f;
        } else if ((i % 5) == 0) {
            weight_ar[i] = 1.5f;
        } else if ((i % 6) == 0) {
            weight_ar[i] = -3.5f;
        }
    }

    float cpu_sum = 0.0f;
    for (auto w : weight_ar) { cpu_sum += w; }

    // Copy to GPU memory:
    float* d_weight_ar = nullptr;
    cudaMalloc (&d_weight_ar, arraysz * sizeof(float));
    cudaMemcpy (d_weight_ar, weight_ar.data(), arraysz * sizeof(float), cudaMemcpyHostToDevice);

    // Call the function:
    float gpu_sum = shufflesum_gpu (d_weight_ar, arraysz);

    std::cout << "GPU array sum is " << gpu_sum << ". CPU array sum is " << cpu_sum
              << "\nGPU/CPU method difference is "
              << (std::abs(gpu_sum - cpu_sum) / std::numeric_limits<float>::epsilon())
              << " epsilons = " << std::abs(gpu_sum - cpu_sum) << "\n";

    return 0;
}
