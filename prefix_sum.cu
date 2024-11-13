#include <cmath>
#include <vector>
#include <iostream>
#include <cuda.h>

// C version of my python shifted index fn.
__device__ int shifted_idx (int idx)
{
    int num_banks = 16; // 16 and 768 works for GTX1070/Compute capability 6.1
    int bank_width_int32 = 768;
    // max_idx = (bank_width_int32 * num_banks)
    int idx_idiv_num_banks = idx; // num_banks
    int idx_mod_num_banks = idx % num_banks;
    int offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks));

    return offs_idx;
}

// This should work - it comes from my PyCuda implementation in VisualAttention and was fairly easy to convert to C
__global__ void reduceit (float* scan_ar_, float* input_ar_, float* carry_, int n, int arraysz)
{
    // Access thread/block/grid info like this:
    // int i = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.x * gridDim.x);

    int thid = threadIdx.x;
    int tb_offset = blockIdx.x * blockDim.x; // threadblock offset
    //int d = n / 2; // Integer division
    int d = n >> 1;

    // This runs for every element in input_ar_
    if ((thid + tb_offset) < (arraysz - d)) {

        extern __shared__ float temp[]; // You have to use an argument in the <<< >>>
                                        // invocation to set this at runtime. See
                                        // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/. Also,
                                        // it could be statically set at compile time
                                        // (but that won't work here)

        int ai = thid; // within one block
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
            if (thid < d) {
                // Block B
                ai = offset * (2 * thid + 1) - 1;
                bi = offset * (2 * thid + 2) - 1;
                ai_s = shifted_idx(ai);
                bi_s = shifted_idx(bi);
                temp[bi_s] += temp[ai_s];
            }
            offset *= 2;
            d >>= 1;

        }
        __syncthreads();
#if 0
        // DEBUG
        scan_ar_[ai] = (float)bi_s;

    }
#else

        // Block C: clear the last element - the first step of the downsweep
        if (thid == 0) {
            int nm1s = shifted_idx (n - 1);
            // Carry last number in the block
            carry_[blockIdx.x] = temp[nm1s];
            temp[nm1s] = 0;
        }

        // Downsweep: traverse down tree & build scan
        d = 1;
        while (d < n) {
            offset >>= 1;
            __syncthreads();
            if (thid < d) {
                // Block D
                ai = offset * (2 * thid + 1) - 1;
                bi = offset * (2 * thid + 2) - 1;
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
        if (bi < n) { scan_ar_[bi + tb_offset] = temp[bi_s]; }
    }
#endif
    __syncthreads();
}

// Last job is to add on the carry to each part of scan_ar WHILE AT THE SAME TIME SUMMING WITHIN A BLOCK
__global__ void sum_scans (float* new_carry_ar_, float* scan_ar_, int scan_ar_sz, float* carry_ar_)
{
    int thid = threadIdx.x;
    int tb_offset = blockIdx.x * blockDim.x; // threadblock offset
    int arr_addr = thid + tb_offset;
    if (blockIdx.x > 0 && arr_addr < scan_ar_sz) {
        new_carry_ar_[arr_addr] = scan_ar_[arr_addr] + carry_ar_[blockIdx.x];
    } else if (blockIdx.x == 0 && arr_addr < scan_ar_sz) {
        new_carry_ar_[arr_addr] = scan_ar_[arr_addr];
    }
    __syncthreads();
}

// Sum up d_weight_ar which should be of size rowlen*rowlen. Use threadsperblock GPU
// threads per block. Return result in r_scanf_ar.
__host__ void prefixsum_gpu (float* d_weight_ar, int rowlen, int threadsperblock, std::vector<float>& r_scanf_ar)
{
    std::cout << __func__ << " called" << std::endl;
    // Build input data for the test
    int arraysz = rowlen * rowlen;
    int blockspergrid = std::ceil (static_cast<float>(arraysz) / static_cast<float>(threadsperblock));
    // To pad the arrays out to exact number of blocks
    int arrayszplus = arraysz;
    if (arraysz % threadsperblock) { arrayszplus = arraysz + threadsperblock - arraysz % threadsperblock; }

    // scan_ar is going to hold the result of scanning the input. Temporary.
    std::vector<float> scan_ar (arrayszplus, 0.0f);
    // Explicitly copy working data to device
    float* d_scan_ar = nullptr;
    cudaMalloc (&d_scan_ar, arrayszplus * sizeof(float));
    cudaMemcpy (d_scan_ar, scan_ar.data(), arrayszplus * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << __func__ << " alloced d_scan_ar" << std::endl;
#if 1
    // scanf_ar is a data structure to hold the final, corrected scan (was uint, but no longer). Prolly can be d_scan_ar
    float* d_scanf_ar = nullptr;
    std::vector<float> scanf_ar (arrayszplus, 0.0f);
    cudaMalloc (&d_scanf_ar, arrayszplus * sizeof(float));
    cudaMemcpy (d_scanf_ar, scanf_ar.data(), arrayszplus * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << __func__ << " alloced d_scanf_ar" << std::endl;
#endif
    // Make up a list of carry vectors and allocate device memory
    std::vector<std::vector<float>> carrylist;
    std::vector<float*> d_carrylist;
    // And containers for the scan
    std::vector<std::vector<float>> scanlist;
    std::vector<float*> d_scanlist;
    int asz = arraysz;
    while (asz > threadsperblock) {

        int carrysz = std::ceil (static_cast<float>(asz) / static_cast<float>(threadsperblock));
        // Ensure carrysz is a multiple of threadsperblock:
        if (carrysz % threadsperblock) { carrysz = carrysz + threadsperblock - carrysz % threadsperblock; }
        carrylist.emplace_back (carrysz, 0.0f);

        // This is: d_carrylist.append (cuda.to_device(carrylist[-1])); // end of carrylist carrylist.back()
        float* d_cl = nullptr;
        cudaMalloc (&d_cl, carrysz * sizeof(float));
        std::cout << __func__ << " alloced a d_carrylist entry with carrysz " << carrysz << std::endl;
        cudaMemcpy (d_cl, carrylist.back().data(), carrysz * sizeof(float), cudaMemcpyHostToDevice);
        d_carrylist.push_back (d_cl);

        asz = std::ceil (asz / threadsperblock);
        std::cout << "asz = " << asz << std::endl;
        int scansz = asz;
        if (scansz % threadsperblock) { scansz = scansz + threadsperblock - scansz % threadsperblock; }

        scanlist.emplace_back (scansz, 0.0f);

        // This is: d_scanlist.append (cuda.to_device(scanlist[-1]));
        float* d_sl = nullptr;
        cudaMalloc (&d_sl, scansz * sizeof(float));
        std::cout << __func__ << " alloced a d_scanlist entry with scansz " << scansz << std::endl;
        cudaMemcpy (d_sl, scanlist.back().data(), scansz * sizeof(float), cudaMemcpyHostToDevice);
        d_scanlist.push_back (d_sl);
    }
    std::cout << "d_carrylist has " << d_carrylist.size() << " entries and d_scanlist has " << d_scanlist.size() << std::endl;
    // Add a last carrylist, as this will be required as a dummy carry list for the last call to reduceit()
    carrylist.push_back (std::vector<float>(1, 0.0f));
    float* d_cl = nullptr;
    cudaMalloc (&d_cl, 1 * sizeof(float));
    cudaMemcpy (d_cl, carrylist.back().data(), 1 * sizeof(float), cudaMemcpyDeviceToHost);
    d_carrylist.push_back (d_cl);

    cudaDeviceSynchronize();
    //
    // Compute partial scans of the top-level weight_ar and the lower level
    // partial sums
    //
    // The first input is the weight array, compute block-wise prefix-scan sums:
    std::cout << "reduceit<<<" << blockspergrid << ", " << threadsperblock << ", "
              << (threadsperblock * sizeof(float)) << ">>>\n";
    reduceit<<<blockspergrid, threadsperblock, 12288 * sizeof(float)>>>(d_scan_ar, d_weight_ar, d_carrylist[0], threadsperblock, arrayszplus);
    cudaDeviceSynchronize();
    std::cout << "reduceit synchronized" << std::endl; // If fails to show, reduceit failed.

    // Debug what reduceit did
    std::vector<float> debug_ar (arrayszplus, 0.0f);
    cudaMemcpy (debug_ar.data(), d_scan_ar, arrayszplus * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float dsmax = 0.0f;
    for (int i = 0; i < arrayszplus; ++i) {
        if (i < 16) {
            std::cout << "d_scan_ar:  " << debug_ar[i] << std::endl;
        }
        dsmax = debug_ar[i] > dsmax ? debug_ar[i] : dsmax;
    }
    std::cout << "dsmax is " << dsmax << std::endl;

    asz = std::ceil (arrayszplus / threadsperblock);
    int j = 0;
    int scanblocks = 0;
    while (asz > threadsperblock) {
        scanblocks = std::ceil (asz / threadsperblock);
        scanblocks = scanblocks + threadsperblock - scanblocks % threadsperblock;
        std::cout << "scanblocks reduceit<<<" << scanblocks << ", " << threadsperblock << ", "
                  << (threadsperblock * sizeof(float))<< ">>>\n";
        reduceit<<<scanblocks, threadsperblock, 12288 * sizeof(float)>>>(d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, carrylist[j].size());
        asz = scanblocks;
        j++;
    }
    // Plus one more iteration:
    scanblocks = std::ceil (asz / threadsperblock);
    scanblocks = scanblocks + threadsperblock - scanblocks % threadsperblock;
    reduceit<<<scanblocks, threadsperblock, 12288 * sizeof(float)>>>(d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, carrylist[j].size());
    std::cout << "final scanblocks reduceit<<<" << scanblocks << ", " << threadsperblock << ", "
              << (threadsperblock * sizeof(float))<< ">>>\n";

    // Construct the scans back up the tree by summing the "carry" into the "scans"
    j = static_cast<int>(scanlist.size());
    while (j > 0) {
        int sumblocks = std::ceil( scanlist[j-1].size() / threadsperblock );
        sum_scans<<<sumblocks, threadsperblock>>>(d_carrylist[j-1], d_scanlist[j-1], scanlist[j-1].size(), d_carrylist[j]);
        // Now d_carrylist[j-1] has had its carrys added from the lower level
        j--;
    }
    // The final sum_scans() call.
    sum_scans<<<blockspergrid, threadsperblock>>>(d_scanf_ar, d_scan_ar, arrayszplus, d_carrylist[0]);

    // Now just copy back from device to host iunto std::vector<float> r_scanf_ar
    if (r_scanf_ar.size() != arrayszplus) {
        std::cerr << "Can't fill r_scanf_ar with result :L(\n";
    } else {
        std::cout << "Copy to r_scanf_ar...\n";
        //cudaMemcpy (r_scanf_ar.data(), d_scanf_ar, arrayszplus * sizeof(float), cudaMemcpyDeviceToHost);
    }

    std::cout << "End of prefixsum_gpu\n";
}

int main()
{
    int rowlen = 150;
    int threadsperblock = 128;
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
    weight_ar[55] = 2.3f;
    weight_ar[63] = 2.3f;
    weight_ar[64] = 2.3f;
    weight_ar[65] = 2.3f;
    weight_ar[77] = 2.3f;
    weight_ar[79] = 2.3f;
    weight_ar[80] = 2.3f;
    weight_ar[128] = 2.3f;
    weight_ar[129] = 2.3f;
    weight_ar[130] = 2.3f;
    weight_ar[191] = 2.3f;
    weight_ar[192] = 2.3f;
    weight_ar[193] = 2.3f;
    weight_ar[254] = 2.3f;
    weight_ar[255] = 2.3f;
    weight_ar[256] = 2.3f;
    weight_ar[257] = 2.3f;

    if (rowlen > 149) {
        weight_ar[22486] = 2.54f;
    }

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

#if 0
    // Print result
    unsigned int j = 0;
    while (j < arraysz) {
        std::cout << "weight_ar[" << j << "] = " << weight_ar[j] << " ... scanf_ar[]=" << r_scanf_ar[j] << std::endl;
        ++j;
    }
#endif
    return 0;
}
