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
__inline__ __device__ float warpReduceSum (float val)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum (float val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val);          // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }   // Write reduced value to shared memory
    __syncthreads();                    // Wait for all partial reductions
    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) { val = warpReduceSum (val); } // Final reduce within first warp
    return val;
}

__global__ void reduceit (float *in, float* out, int N)
{
    float sum = 0.0f;
    // reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum (sum);
    if (threadIdx.x == 0) { out[blockIdx.x] = sum; }
}

__host__ void shufflesum_gpu_work (float *in, float* out, int N)
{
    int threads = threadsperblock;
    int blocks = min((N + threads - 1) / threads, 1024);
    reduceit<<<blocks, threads>>>(in, out, N);
    reduceit<<<1, 1024>>>(out, out, blocks);
}

__host__ void shufflesum_gpu (float* d_weight_ar, int rowlen, std::vector<float>& r_scanf_ar)
{
    // Build input data for the test
    int arraysz = rowlen * rowlen;

    //int blockspergrid = std::ceil (static_cast<float>(arraysz) / static_cast<float>(threadsperblock));
    // To pad the arrays out to exact number of blocks
    int arrayszplus = arraysz;
    if (arraysz % threadsperblock) { arrayszplus = arraysz + threadsperblock - arraysz % threadsperblock; }

    // scanf_ar is a data structure to hold the final sum.
    float* d_scanf_ar = nullptr;
    cudaMalloc (&d_scanf_ar, arrayszplus * sizeof(float));

    shufflesum_gpu_work (d_weight_ar, d_scanf_ar, arraysz);

    float sum;
    cudaMemcpy(&sum, d_scanf_ar, sizeof(float), cudaMemcpyDeviceToHost);
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

    std::vector<float> weight_ar (arrayszplus, 0.0f);

    // Now some non-zero, non-unary weights
    weight_ar[0] = 1.1f;
    weight_ar[1] = 1.3f;
    weight_ar[2] = 1.2f;
    weight_ar[3] = 0.2f;
    weight_ar[4] = 0.3f;

    if (rowlen > 3) {
        weight_ar[12] = 1.7f;
    }
    if (rowlen > 4) {
        weight_ar[22] = 1.9f;
    }
    if (rowlen > 5) {
        weight_ar[33] = 2.3f;
    }
    if (rowlen > 17) {
        weight_ar[44] = 2.3f;
    }
    weight_ar[45] = 2.3f;
    weight_ar[55] = 2.4f;
    weight_ar[63] = 2.5f;
    weight_ar[64] = 2.6f;
    weight_ar[65] = 2.7f;
    weight_ar[77] = 2.8f;
    weight_ar[79] = 2.9f;
    weight_ar[80] = 3.0f;
    weight_ar[128] = 2.3f;
    weight_ar[129] = 2.3f;
    weight_ar[130] = 2.5f;
    weight_ar[191] = 2.6f;
    weight_ar[192] = 2.7f;
    weight_ar[193] = 2.8f;
    weight_ar[254] = 2.9f;
    weight_ar[255] = 2.1f;
    weight_ar[256] = 2.2f;
    weight_ar[257] = 2.3f;

    if (rowlen > 149) {
        weight_ar[22486] = 2.54f;
    }

    float cpu_sum = 0.0f;
    for (auto w : weight_ar) { cpu_sum += w; }

    // Copy to GPU memory:
    float* d_weight_ar = nullptr;
    cudaMalloc (&d_weight_ar, arrayszplus * sizeof(float));
    cudaMemcpy (d_weight_ar, weight_ar.data(), arrayszplus * sizeof(float), cudaMemcpyHostToDevice);

    // To check, copy back into a new thing
    std::vector<float> r_weight_ar (arrayszplus, 0.0f);
    cudaMemcpy (r_weight_ar.data(), d_weight_ar, arrayszplus * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < arrayszplus; ++i) {
        if (r_weight_ar[i] != weight_ar[i]) {
            std::cerr << "Mismatch r_weight_ar[i] " << r_weight_ar[i] << " != " << weight_ar[i] << std::endl;
            exit (-1);
        }
    }

    // Call the function:
    std::vector<float> r_scanf_ar(arrayszplus);
    shufflesum_gpu (d_weight_ar, rowlen, r_scanf_ar);

#ifdef DEBUG_ALL_VALUES
    // Print result
    unsigned int j = 0;
    while (j < arraysz) {
        std::cout << "weight_ar[" << j << "] = " << weight_ar[j] << " ... scanf_ar[]=" << r_scanf_ar[j] << std::endl;
        ++j;
    }
#endif

    if (std::abs(r_scanf_ar[arraysz-1] - cpu_sum) > 2 * cpu_sum * std::numeric_limits<float>::epsilon() ) {
        std::cerr << "FAIL r_scanf_ar[arraysz-1] != cpu_sum: " << r_scanf_ar[arraysz-1] << "!=" << cpu_sum
                  << " delta is " << (std::abs(r_scanf_ar[arraysz-1] - cpu_sum) / std::numeric_limits<float>::epsilon())
                  << " epsilons\n";
    } else {
        std::cout << "Array sum is " << r_scanf_ar[arraysz-1] << ". GPU/CPU method difference is "
                  << (std::abs(r_scanf_ar[arraysz-1] - cpu_sum) / std::numeric_limits<float>::epsilon())
                  << " epsilons = " << std::abs(r_scanf_ar[arraysz-1] - cpu_sum) << "\n";
    }
    return 0;
}
