#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

// Parameters

// How many thread blocks will we sum up?
static constexpr int arrayblocks = 10240;
// How many threads per block to specify.
static constexpr int threadsperblock = 512;
// Mask for __shfl_down_sync
static constexpr unsigned int all_in_warp = 0xffffffff;

// In each warp reduce three values per thread
__inline__ __device__ float3 warpReduceSum (float valR, float valG, float valB)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        valR += __shfl_down_sync (all_in_warp, valR, offset);
        valG += __shfl_down_sync (all_in_warp, valG, offset);
        valB += __shfl_down_sync (all_in_warp, valB, offset);
    }
    return make_float3 (valR, valG, valB);
}

__inline__ __device__ float3 blockReduceSum (float3 val)
{
    static __shared__ float3 shared[32];    // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val.x, val.y, val.z); // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }      // Write reduced value to shared memory
    __syncthreads();                           // Wait for all partial reductions
    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float3(0.0f, 0.0f, 0.0f);
    if (wid == 0) { val = warpReduceSum (val.x, val.y, val.z); } // Final reduce within first warp
    return val;
}

// Input is float3 format.
__global__ void reduceit (float3* in, float3* out, int N)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    // reduce multiple elements per thread
    // blockIdx.x * blockDim.x takes us along the array to our own thread block.
    // Adding thread idx gets us to *the first memory location for this thread*; i.
    // The other memory locations that this thread deals wtih are spaced by 1024 * 512 = 524288. N is 5242880, so for arrayblocks = 10240, this reduceit will loop 10 times here and add to the sum for 10 widely spaced locations.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum.x += in[i].x;
        sum.y += in[i].y;
        sum.z += in[i].z;
    }

    // The blockReduceSum function call is called in each thread and sums up the already slightly
    // reduced numbers, with warps of threads co-operating via shuffle calls.
    sum = blockReduceSum (sum); // 1 per thread, sums float3 at a time
    if (threadIdx.x == 0) { out[blockIdx.x] = sum; }
}

__host__ void shufflesum_gpu_work (float3* in, float3* out, int N)
{
    int threads = threadsperblock; // 1024 max
    int gridblocks = min((N + threads - 1) / threads, 1024);
    dim3 gridblcks(gridblocks);  // as many as you like (more or less)
    dim3 thrds(threads); // product of thrds should be <= 1024
    std::cout << "grid will be make of: " << gridblcks.x << " x " << gridblcks.y
              << " blocks  and each block contains "  << thrds.x << " x " << thrds.y << " threads\n";
    std::cout << "N is " << N << std::endl;

    std::cout << "reduceit (thread 0, block 0) will serially sum elements ";
    for (int i = 0 * blockDim.x + 0; i < N; i += thrds.x * gridblcks.x) {
        std::cout << i << ", ";
    }
    std::cout << "\n";

    reduceit<<<gridblcks, thrds>>>(in, out, N);

    reduceit<<<1, 1024>>>(out, out, gridblocks);
}

__host__ float3 shufflesum_gpu (float3* d_weight_ar, int arraysz)
{
    float3* d_scanf_ar = nullptr;
    cudaMalloc (&d_scanf_ar, arraysz * 3 * sizeof(float));
    shufflesum_gpu_work (d_weight_ar, d_scanf_ar, arraysz);
    float3 sum;
    cudaMemcpy(&sum, d_scanf_ar, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    return sum;
}

int main()
{
    int arraysz = arrayblocks * threadsperblock;

    std::vector<float3> weight_ar (arraysz, make_float3(0.0f, 0.0f, 0.0f));

    // Now some non-zero, non-unary weights
    for (int i = 0; i < arraysz; ++i) {
        if (i % 2 == 0) {
            weight_ar[i] = make_float3 (2.6f, 1.03f, 3.4f);
        } else if ((i % 3) == 0) {
            weight_ar[i] = make_float3 (2.73f, 2.03f, 2.4f);
        } else if ((i % 4) == 0) {
            weight_ar[i] = make_float3 (-3.73f, 3.03f, 1.4f);
        } else if ((i % 5) == 0) {
            weight_ar[i] = make_float3 (1.73f, 4.03f, 0.4f);
        } else if ((i % 6) == 0) {
            weight_ar[i] = make_float3 (-3.02f, 1.03f, -1.4f);
        }
    }
#if 0
    float3 cpu_sum = make_float3 (0.0f, 0.0f, 0.0f);
    for (auto w : weight_ar) {
        cpu_sum.x += w.x;
        cpu_sum.y += w.y;
        cpu_sum.z += w.z;
    }
# if 0
    // Copy to GPU memory:
    float3* d_weight_ar = nullptr;
    cudaMalloc (&d_weight_ar, arraysz * 3 * sizeof(float));
    cudaMemcpy (d_weight_ar, weight_ar.data(), arraysz * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Call the function:
    float3 gpu_sum = shufflesum_gpu (d_weight_ar, arraysz);

    std::cout << "GPU array sum is (" << gpu_sum.x << "," << gpu_sum.y << "," << gpu_sum.z << ")."
              << " CPU array sum is " << cpu_sum.x << "," << cpu_sum.y << "," << cpu_sum.z << "\n";
# else
    std::cout << " CPU array sum is " << cpu_sum.x << "," << cpu_sum.y << "," << cpu_sum.z << "\n";
# endif
#else
    // Copy to GPU memory:
    float3* d_weight_ar = nullptr;
    cudaMalloc (&d_weight_ar, arraysz * 3 * sizeof(float));
    cudaMemcpy (d_weight_ar, weight_ar.data(), arraysz * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Call the function:
    float3 gpu_sum = shufflesum_gpu (d_weight_ar, arraysz);
    std::cout << "GPU array sum is (" << gpu_sum.x << "," << gpu_sum.y << "," << gpu_sum.z << ").\n";

#endif
    return 0;
}
