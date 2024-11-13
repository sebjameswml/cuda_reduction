#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cuda.h>

// Parameters - array size (from rowlen) and features that may need to be tweaked for your GPU type.

// We'll imagine an array of rowlen * rowlen fields
static constexpr int rowlen = 150;
// How many threads per block to specify. Compute capability 6.1 (10series cards) and
// also 8.6/8.9 (30 and 40 series cards) provide 128 threads per SM. Each has 4 warp
// schedulers so a warp is 32 threads.
static constexpr int threadsperblock = 128;
// Shared memory config
static constexpr int num_banks = 16; // 16 and 768 works for GTX1070/Compute capability 6.1 and is ok on later cards too
static constexpr int bank_width_int32 = 768;
// Shared memory size (that we'll use) is 12288 x 32 bit words so 49152 bytes so 48
// KB. There's potentially more shared memory available (96 KB on CC6.1 and 128 KB on
// CC8.x)
static constexpr int sh_mem_size = num_banks * bank_width_int32;

// Device function. Shift an index to avoid bank conflicts
__inline__ __device__ int shifted_idx (int idx)
{
    int idx_idiv_num_banks = idx;
    int idx_mod_num_banks = idx % num_banks;
    int offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks));
    return offs_idx;
}

// Kernel. Apply a prefix sum algorithm to input_ar_, reducing it to the outputs scan_ar_ and
// carry_. The array size (of input_ar_ and scan_ar_) is arraysz
__global__ void reduceit (float* scan_ar_, float* input_ar_, float* carry_, int arraysz)
{
    int _threadsperblock = blockDim.x * blockDim.y * blockDim.z;
    int tb_offset = blockIdx.x * blockDim.x; // threadblock offset
    int d = _threadsperblock >> 1;

    // This runs for every element in input_ar_
    if ((threadIdx.x + tb_offset) < (arraysz - d)) {

        extern __shared__ float temp[]; // Use an argument in the <<< >>> invocation to
                                        // set the size of the shared memory at runtime. See
                                        // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/.

        int ai = threadIdx.x; // within one block
        int bi = ai + d; // ai and bi are well separated across the shared memory
        int ai_s = shifted_idx (ai);
        int bi_s = shifted_idx (bi);

        // Summing scheme.
        temp[ai_s] = input_ar_[ai + tb_offset];
        temp[bi_s] = input_ar_[bi + tb_offset];

        int offset = 1;
        // Upsweep: build sum in place up the tree
        while (d > 0) {
            __syncthreads();
            if (threadIdx.x < d) {
                // Block B
                ai = offset * (2 * threadIdx.x + 1) - 1;
                bi = offset * (2 * threadIdx.x + 2) - 1;
                ai_s = shifted_idx(ai);
                bi_s = shifted_idx(bi);
                temp[bi_s] += temp[ai_s];
            }
            offset <<= 1;
            d >>= 1;

        }
        __syncthreads();

        // Block C: clear the last element - the first step of the downsweep
        if (threadIdx.x == 0) {
            int nm1s = shifted_idx (_threadsperblock - 1);
            // Carry last number in the block
            carry_[blockIdx.x] = temp[nm1s];
            temp[nm1s] = 0;
        }

        // Downsweep: traverse down tree & build scan
        d = 1;
        while (d < _threadsperblock) {
            offset >>= 1;
            __syncthreads();
            if (threadIdx.x < d) {
                // Block D
                ai = offset * (2 * threadIdx.x + 1) - 1;
                bi = offset * (2 * threadIdx.x + 2) - 1;
                ai_s = shifted_idx(ai);
                bi_s = shifted_idx(bi);
                float t = temp[ai_s];
                temp[ai_s] = temp[bi_s];
                temp[bi_s] += t;
            }
            d <<= 1;
        }
        __syncthreads();

        // Block E: write results to device memory
        scan_ar_[ai + tb_offset] = temp[ai_s];
        if (bi < _threadsperblock) { scan_ar_[bi + tb_offset] = temp[bi_s]; }
    }
    __syncthreads();
}

// Kernel. Last job is to add on the carry to each part of scan_ar WHILE AT THE SAME TIME SUMMING WITHIN A BLOCK
__global__ void sum_scans (float* new_carry_ar_, float* scan_ar_, int scan_ar_sz, float* carry_ar_)
{
    int arr_addr = threadIdx.x + (blockIdx.x * blockDim.x); // blockIdx.x * blockDim.x is the threadblock offset
    if (blockIdx.x > 0 && arr_addr < scan_ar_sz) {
        new_carry_ar_[arr_addr] = scan_ar_[arr_addr] + carry_ar_[blockIdx.x];
    } else if (blockIdx.x == 0 && arr_addr < scan_ar_sz) {
        new_carry_ar_[arr_addr] = scan_ar_[arr_addr];
    }
    __syncthreads();
}

// Host function. Sum up d_weight_ar which should be of size rowlen*rowlen. Specify _threadsperblock GPU
// threads per block. Return result in r_scanf_ar.
__host__ void prefixsum_gpu (float* d_weight_ar, int rowlen, int _threadsperblock, std::vector<float>& r_scanf_ar)
{
    // Build input data for the test
    int arraysz = rowlen * rowlen;
    int blockspergrid = std::ceil (static_cast<float>(arraysz) / static_cast<float>(_threadsperblock));
    // To pad the arrays out to exact number of blocks
    int arrayszplus = arraysz;
    if (arraysz % _threadsperblock) { arrayszplus = arraysz + _threadsperblock - arraysz % _threadsperblock; }

    // scan_ar is going to hold the result of scanning the input. Temporary.
    std::vector<float> scan_ar (arrayszplus, 0.0f);
    // Explicitly copy working data to device
    float* d_scan_ar = nullptr;
    cudaMalloc (&d_scan_ar, arrayszplus * sizeof(float));
    cudaMemcpy (d_scan_ar, scan_ar.data(), arrayszplus * sizeof(float), cudaMemcpyHostToDevice);

    // scanf_ar is a data structure to hold the final sum.
    float* d_scanf_ar = nullptr;
    std::vector<float> scanf_ar (arrayszplus, 0.0f);
    cudaMalloc (&d_scanf_ar, arrayszplus * sizeof(float));
    cudaMemcpy (d_scanf_ar, scanf_ar.data(), arrayszplus * sizeof(float), cudaMemcpyHostToDevice);

    // Make up a list of carry vectors and allocate device memory
    std::vector<std::vector<float>> carrylist;
    std::vector<float*> d_carrylist;
    // And containers for the scan
    std::vector<std::vector<float>> scanlist;
    std::vector<float*> d_scanlist;
    int asz = arraysz;
    while (asz > _threadsperblock) {

        int carrysz = std::ceil (static_cast<float>(asz) / static_cast<float>(_threadsperblock));
        // Ensure carrysz is a multiple of _threadsperblock:
        if (carrysz % _threadsperblock) { carrysz = carrysz + _threadsperblock - carrysz % _threadsperblock; }
        carrylist.emplace_back (carrysz, 0.0f);

        // This is: d_carrylist.append (cuda.to_device(carrylist[-1])); // end of carrylist carrylist.back()
        float* d_cl = nullptr;
        cudaMalloc (&d_cl, carrysz * sizeof(float));
        cudaMemcpy (d_cl, carrylist.back().data(), carrysz * sizeof(float), cudaMemcpyHostToDevice);
        d_carrylist.push_back (d_cl);

        asz = std::ceil (static_cast<float>(asz) / static_cast<float>(_threadsperblock));
        int scansz = asz;
        if (scansz % _threadsperblock) { scansz = scansz + _threadsperblock - scansz % _threadsperblock; }

        scanlist.emplace_back (scansz, 0.0f);

        // This is: d_scanlist.append (cuda.to_device(scanlist[-1]));
        float* d_sl = nullptr;
        cudaMalloc (&d_sl, scansz * sizeof(float));
        cudaMemcpy (d_sl, scanlist.back().data(), scansz * sizeof(float), cudaMemcpyHostToDevice);
        d_scanlist.push_back (d_sl);
    }
    // Add a last carrylist, as this will be required as a dummy carry list for the last call to reduceit()
    carrylist.push_back (std::vector<float>(1, 0.0f));
    float* d_cl = nullptr;
    cudaMalloc (&d_cl, 1 * sizeof(float));
    cudaMemcpy (d_cl, carrylist.back().data(), 1 * sizeof(float), cudaMemcpyDeviceToHost);
    d_carrylist.push_back (d_cl);

    //
    // Compute partial scans of the top-level weight_ar and the lower level partial sums
    //
    // The first input is the weight array, compute block-wise prefix-scan sums:
    reduceit<<<blockspergrid, _threadsperblock, sh_mem_size * sizeof(float)>>>(d_scan_ar, d_weight_ar, d_carrylist[0], arrayszplus);
    cudaDeviceSynchronize(); // sync after kernel completes

    asz = std::ceil (static_cast<float>(arrayszplus) / static_cast<float>(_threadsperblock));
    int j = 0;
    int scanblocks = 0;
    while (asz > _threadsperblock) {
        scanblocks = std::ceil (static_cast<float>(asz) / static_cast<float>(_threadsperblock));
        scanblocks = scanblocks + _threadsperblock - scanblocks % _threadsperblock;
        reduceit<<<scanblocks, _threadsperblock, sh_mem_size * sizeof(float)>>>(d_scanlist[j], d_carrylist[j], d_carrylist[j+1], carrylist[j].size());
        cudaDeviceSynchronize();
        asz = scanblocks;
        j++;
    }
    // Plus one more iteration:
    scanblocks = std::ceil (static_cast<float>(asz) / static_cast<float>(_threadsperblock));
    scanblocks = scanblocks + _threadsperblock - scanblocks % _threadsperblock;
    reduceit<<<scanblocks, _threadsperblock, sh_mem_size * sizeof(float)>>>(d_scanlist[j], d_carrylist[j], d_carrylist[j+1], carrylist[j].size());
    cudaDeviceSynchronize();

    // Construct the scans back up the tree by summing the "carry" into the "scans"
    j = static_cast<int>(scanlist.size());
    while (j > 0) {
        int sumblocks = std::ceil(static_cast<float>(scanlist[j-1].size()) / static_cast<float>(_threadsperblock));
        sum_scans<<<sumblocks, _threadsperblock>>>(d_carrylist[j-1], d_scanlist[j-1], scanlist[j-1].size(), d_carrylist[j]);
        cudaDeviceSynchronize();
        // Now d_carrylist[j-1] has had its carrys added from the lower level
        j--;
    }
    // The final sum_scans() call.
    sum_scans<<<blockspergrid, _threadsperblock>>>(d_scanf_ar, d_scan_ar, arrayszplus, d_carrylist[0]);

    // Now just copy back from device to host iunto std::vector<float> r_scanf_ar
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
    prefixsum_gpu (d_weight_ar, rowlen, threadsperblock, r_scanf_ar);

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
