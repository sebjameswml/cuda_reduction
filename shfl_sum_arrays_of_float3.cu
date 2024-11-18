#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

// Parameters

// How many arrays to sum?
static constexpr int number_of_arrays = 4;
// How many elements per array? May be > or < threadsperblock
static constexpr int elements_per_array = 2048;
// Ideal number of threads per block
static constexpr int threadsperblock = 512;
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

// Run by the 32 threads of a warp
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
__global__ void reduceit_arrays (const float3* in, float3* out, int n_arrays, int n_elements)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    // The y axis of our threads/threadblocks indexes which of the n_arrays this sum relates to
    int omm_id = blockIdx.y * blockDim.y + threadIdx.y;

    // This gives a memory offset to get to the right part of the input memory
    int thread_offset = omm_id * n_elements;
    // For array index checking
    int data_sz = n_arrays * n_elements;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements && omm_id < n_arrays;
         i += blockDim.x * gridDim.x) { // jumps the whole threadblock as threadblock may not span all data

        int tidx = thread_offset + i;
#if 0 // If your arrays need 'turning 90 degrees'
        // Convert to x/y
        int tx = tidx % n_elements;
        int ty = tidx / n_elements;
        // Convert to real memory index
        tidx = tx * n_arrays + ty;
#endif
        if (tidx < data_sz) {
            sum.x += in[tidx].x;
            sum.y += in[tidx].y;
            sum.z += in[tidx].z;
        } else {
            sum.x = 0.0f;
            sum.y = 0.0f;
            sum.z = 0.0f;
        }
    }

    sum = blockReduceSum (sum);
    __syncthreads();

    // This gets the correct output location in out. gridDim.x is "n_sums"
    if (threadIdx.x == 0 && omm_id < n_arrays) {
        // Number of sums is the number of 1D threadblocks that span n_elements. This is gridDim.x.
        size_t out_idx = omm_id * gridDim.x + blockIdx.x;
        if (out_idx < n_arrays * gridDim.x) {
            out[omm_id * gridDim.x + blockIdx.x] = sum;
        }
    }
}

#include <chrono>

// Compute the sum of each of N-arrays arrays that are layed out one-after-another in the input
// in. out should be the same size as in and is used when shuffle-computing the sums and also to
// hold the results.
__host__ void shufflesum_arrays (const float3* in, int n_arrays, int n_elements,
                                 std::vector<float3>& sums, std::vector<float3>& final_sums)
{
    using namespace std::chrono;
    using sc = std::chrono::steady_clock;

    // Working out the threads per block is the thing

    // threadsperblock is the ideal size (512). warp size is 32.  So basic threadblock
    // size should be 32 in x and 16 in y, giving 512 threads. If n_elements < 32 or
    // n_arrays < 16, then some kind of padding should happen so that this works, even
    // if it's slow.  EXCEPT for the reduction, I can't mix memory values from different
    // arrays in a threadblock, so the threadblock has to be 1D and thus configurable -
    // dynamically sized to match the number of elements in the array.
    int warps_base = n_elements / 32;
    int warps_extra = n_elements % 32;
    int tbx = (warps_base * 32) + (warps_extra ? 32 : 0);
    tbx = std::min (tbx, threadsperblock);
    dim3 stg1_blockdim(tbx, 1);

    // Then figure out how many threadblocks to run.
    dim3 stg1_griddim(1, 1);
    // We will make use of the number of sums later, but we don't store this in stg1_griddim.x
    int n_sums = n_elements / stg1_blockdim.x + (n_elements % stg1_blockdim.x ? 1 : 0);
    stg1_griddim.y = n_arrays / stg1_blockdim.y + (n_arrays % stg1_blockdim.y ? 1 : 0);

    std::cout << "About to run with stg1_griddim = (" << stg1_griddim.x << " x " << stg1_griddim.y
              << ") and stg1_blockdim = (" << stg1_blockdim.x << " x " << stg1_blockdim.y << ") thread blocks\n";

    float3* d_output = nullptr;
    // Malloc n_arrays * n_sums elements
    cudaMalloc (&d_output, n_arrays * n_sums * 3 * sizeof(float));
    auto t0 = sc::now();
    reduceit_arrays<<<stg1_griddim, stg1_blockdim>>>(in, d_output, n_arrays, n_elements);
    cudaDeviceSynchronize();

    float3* d_final = nullptr;
    // Malloc n_arrays elements for the final sums (or could re-use d_output)
    cudaMalloc (&d_final, n_arrays * 3 * sizeof(float));

    auto t1 = sc::now();

    // stg1_griddim.x is n_sums
    sums.resize (stg1_griddim.x * n_arrays, make_float3(0,0,0));
    std::cout << "resized sums to have size " << stg1_griddim.x << " * " << n_arrays << " = " << (stg1_griddim.x * n_arrays) << std::endl;
    // Copy intermediate d_output into sums
    cudaMemcpy (sums.data(), d_output, sums.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    warps_base = stg1_griddim.x / 32;
    warps_extra = stg1_griddim.x % 32;
    tbx = (warps_base * 32) + (warps_extra ? 32 : 0);
    tbx = std::min (tbx, threadsperblock);
    dim3 stg2_blockdim(tbx, 1);
    dim3 stg2_griddim(1, 1);
    stg2_griddim.x = stg1_griddim.x / stg1_blockdim.x + (stg1_griddim.x % stg2_blockdim.x ? 1 : 0);
    stg2_griddim.y = n_arrays / stg2_blockdim.y + (n_arrays % stg2_blockdim.y ? 1 : 0);

    std::cout << "About to run with stg2_griddim = (" << stg2_griddim.x << " x " << stg2_griddim.y
              << ") and stg2_blockdim = (" << stg2_blockdim.x << " x " << stg2_blockdim.y << ") thread blocks\n";
    auto t2 = sc::now();
    reduceit_arrays<<<stg2_griddim, stg2_blockdim>>>(d_output, d_final, n_arrays, stg1_griddim.x);
    // out_final can be only n_arrays in size
    cudaDeviceSynchronize();

    final_sums.resize (n_arrays, make_float3(0,0,0));
    std::cout << "resized final sums to have size " << n_arrays << " = " << n_arrays << std::endl;
    // Copy intermediate d_output into sums
    cudaMemcpy (final_sums.data(), d_final, final_sums.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree (d_output);
    cudaFree (d_final);

    auto t3 = sc::now();

    sc::duration t_d = (t1 - t0) + (t3 - t2);
    std::cout << "CUDA summing took " << duration_cast<milliseconds>(t_d).count() << " ms\n";
}

int main()
{
    using namespace std::chrono;
    using sc = std::chrono::steady_clock;

    int arraysz = elements_per_array * number_of_arrays;

    std::vector<std::vector<float3>> many_arrays (number_of_arrays);

    // This is the equivalent of the values in GPU ram that I need to average
    std::vector<float3> many_arrays_in_seq (arraysz);

    float v1 = 0.0f;
    float v2 = 0.0f;
    float v3 = 0.0f;
    for (int j = 0; j < number_of_arrays; ++j) {
        many_arrays[j].resize (elements_per_array, make_float3(0,0,0));
        float e = j % 2 == 0 ? j * 0.1f + 0.01f : -j * 0.1f - 0.01f;
        for (int i = 0; i < elements_per_array; ++i) {
            if (i % 2 == 0) {
               many_arrays[j][i] = make_float3 (v1+e, -v2+e, v3+e);
            } else {
                many_arrays[j][i] = make_float3 (-v1+e, v2+e, -v3+e);
            }
            many_arrays_in_seq[j * elements_per_array + i] = many_arrays[j][i];
        }
    }

    // Sum/output on the CPU for comparison
    auto t0 = sc::now();
    for (int j = 0; j < number_of_arrays; ++j) {
        float3 cpu_sum = make_float3 (0.0f, 0.0f, 0.0f);
        for (int i = 0; i < elements_per_array; ++i) {
            cpu_sum.x += many_arrays[j][i].x;
            cpu_sum.y += many_arrays[j][i].y;
            cpu_sum.z += many_arrays[j][i].z;
        }
        //std::cout << "cpu_sum for array " << j << " is (" << cpu_sum.x << "," << cpu_sum.y << "," << cpu_sum.z << ")\n";
    }
    auto t1 = sc::now();
    sc::duration t_d = (t1 - t0);
    std::cout << "CPU summing took " << duration_cast<milliseconds>(t_d).count() << " ms\n";

    // Copy to GPU memory:
    float3* d_many_arrays = nullptr;
    cudaMalloc (&d_many_arrays, arraysz * 3 * sizeof(float));
    cudaMemcpy (d_many_arrays, many_arrays_in_seq.data(), arraysz * 3 * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float3> intermediate_sums; // container for intermediates (debug)
    std::vector<float3> gpu_sums; // container for the final sums
    shufflesum_arrays (d_many_arrays, number_of_arrays, elements_per_array, intermediate_sums, gpu_sums);

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
