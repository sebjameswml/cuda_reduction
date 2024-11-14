#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

// Parameters

// How many arrays to sum?
static constexpr int number_of_arrays = 32;
// How many elements per array? May be > or < threadsperblock
static constexpr int elements_per_array = 32;
// Ideal number of threads per block
// static constexpr int threadsperblock = 512;
// Mask for __shfl_down_sync
static constexpr unsigned int all_in_warp = 0xffffffff;

// In each warp reduce three values per thread
__inline__ __device__ float3 warpReduceSum (float valR, float valG, float valB)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        valR += __shfl_down_sync(all_in_warp, valR, offset);
        valG += __shfl_down_sync(all_in_warp, valG, offset);
        valB += __shfl_down_sync(all_in_warp, valB, offset);
    }
    return make_float3 (valR, valG, valB);
}

__inline__ __device__ float3 blockReduceSum (float3 val)
{
    static __shared__ float3 shared[32];    // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val.x, val.y, val.z); // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }  // Write reduced value to shared memory
    __syncthreads();                       // Wait for all partial reductions
    // read from shared memory only if that warp existed

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float3(0.0f, 0.0f, 0.0f);

    if (wid == 0) { val = warpReduceSum (val.x, val.y, val.z); } // Final reduce within first warp
    return val;
}

// Input is float3 format.
__global__ void reduceit_arrays (float3* in, float3* out, int n_arrays, int n_elements)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    // The y axis of our threads/threadblocks indexes which of the n_arrays this sum relates to
    int omm_id = blockIdx.y * blockDim.y + threadIdx.y;
    // This gives a memory offset to get to the right part of the input memory
    int mem_offset = omm_id * n_elements;
    // How many sums will be created for each array? This comes from x axis
    int n_sums = gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements && omm_id < n_arrays;
         i += blockDim.x * gridDim.x) {

        sum.x += in[mem_offset + i].x;
        sum.y += in[mem_offset + i].y;
        sum.z += in[mem_offset + i].z;
    }
    sum = blockReduceSum (sum);

    int sum_id = blockIdx.x/32;

    // Debug:
    // sum.z = (float)sum_id; // omm_id * n_sums;

    // Think about where that sum goes. out is n_arrays * n_sums
    //if (threadIdx.x == 0 /*&& omm_id < n_arrays*/) {
    if (threadIdx.x == 0 && omm_id < n_arrays) {
        out[omm_id * n_sums + sum_id] = sum;
    }
}

// Compute the sum of each of N-arrays arrays that are layed out one-after-another in the input
// in. out should be the same size as in and is used when shuffle-computing the sums and also to
// hold the results.
__host__ void shufflesum_arrays (float3* in, int n_arrays, int n_elements, std::vector<float3>& sums)
{
    // Working out the threads per block is the thing

    // threadsperblock is the ideal size (512). warp size is 32.  So basic threadblock size should
    // be 32 in x and 16 in y, giving 512 threads. If n_elements < 32 or n_arrays < 16, then some
    // kind of padding should happen so that this works, even if it's slow.
    dim3 threadblockdim(32, 16);

    // So figure out how many threadblocks to run.
    dim3 griddim(1, 1);
    griddim.x = n_elements / threadblockdim.x + (n_elements % threadblockdim.x ? 1 : 0);
    griddim.y = n_arrays / threadblockdim.y + (n_arrays % threadblockdim.y ? 1 : 0);

    std::cout << "About to run with gridDim = (" << griddim.x << " x " << griddim.y << ") and blockDim = (32 x 16) thread blocks\n";

    float3* d_output = nullptr;
    // Malloc n_arrays * n_sums (which is griddim.x) elements
    cudaMalloc (&d_output, griddim.x * n_arrays * 3 * sizeof(float));

    reduceit_arrays<<<griddim, threadblockdim>>>(in, d_output, n_arrays, n_elements);

    // I'll probably need a different stage 2
    //dim3 blks2(1, n_arrays, 1);
    //dim3 thrds2 = { 1024, 1, 1 };
    //reduceit_stage2<<<blks2, thrds2>>>(out_final, out, 1, blocks_x);
    // out_final can be only n_arrays in size

    // griddim.x is n_sums
    sums.resize (griddim.x * n_arrays, make_float3(0,0,0));
    std::cout << "resized sums to have size " << griddim.x << " * " << n_arrays << " = " << (griddim.x * n_arrays) << std::endl;

    // Copy d_output into sums
    cudaMemcpy (sums.data(), d_output, sums.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree (d_output);
}

int main()
{
    int arraysz = elements_per_array * number_of_arrays;

    std::vector<std::vector<float3>> many_arrays (number_of_arrays);

    // This is the equivalent of the values in GPU ram that I need to average
    std::vector<float3> many_arrays_in_seq (arraysz);

     for (int j = 0; j < number_of_arrays; ++j) {
        many_arrays[j].resize (elements_per_array, make_float3(0,0,0));
        float e = j % 2 == 0 ? j* 10.0f : -j* 10.0f;
        //std::cout << "For array " << j << " extra data = " << e << std::endl;
        for (int i = 0; i < elements_per_array; ++i) {
            if (i % 2 == 0) {
               many_arrays[j][i] = make_float3 (0.12f+e, -0.23f+e, 0.34f+e);
            } else {
                many_arrays[j][i] = make_float3 (-0.12f+e, 0.23f+e, -0.34f+e);
            }
            many_arrays_in_seq[j*elements_per_array + i] = many_arrays[j][i];
        }
    }

#if 1
    for (int j = 0; j < number_of_arrays; ++j) {
        float3 cpu_sum = make_float3 (0.0f, 0.0f, 0.0f);
        for (int i = 0; i < elements_per_array; ++i) {
            cpu_sum.x += many_arrays[j][i].x;
            cpu_sum.y += many_arrays[j][i].y;
            cpu_sum.z += many_arrays[j][i].z;
        }
        std::cout << "cpu_sum for array " << j << " is (" << cpu_sum.x << "," << cpu_sum.y << "," << cpu_sum.z << ")\n";
    }
#endif

    // Copy to GPU memory:
    float3* d_many_arrays = nullptr;
    cudaMalloc (&d_many_arrays, arraysz * 3 * sizeof(float));
    cudaMemcpy (d_many_arrays, many_arrays_in_seq.data(), arraysz * 3 * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float3> gpu_sums; // container for the sums
    shufflesum_arrays (d_many_arrays, number_of_arrays, elements_per_array, gpu_sums); // resize gpu_sums in here

    int num_sums = gpu_sums.size() / number_of_arrays;
    for (int i = 0; i < number_of_arrays; ++i) {
        std::cout << "\nGPU array i=" << i << " sums are: ";
        for (int j = 0; j < num_sums; ++j) {
            int idx = i * num_sums + j;
            std::cout << "(" << gpu_sums[idx].x << "," << gpu_sums[idx].y << "," << gpu_sums[idx].z << "), ";
        }
        std::cout << std::endl;
    }

    return 0;
}
