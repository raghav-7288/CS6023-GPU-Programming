#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

/**
 * @brief CUDA kernel for 2D convolution.
 *
 * @param gpu_input Input matrix on the device.
 * @param gpu_output Output matrix on the device.
 * @param gpu_filter Convolution filter on the device.
 * @param k Size of the filter.
 * @param m Number of rows in the input matrix.
 * @param n Number of columns in the input matrix.
 */
__global__ void dkernel(long int *gpu_input, long int *gpu_output, long int *gpu_filter, int k, int m, int n)
{
    extern __shared__ long int filter[];

    unsigned long int id = threadIdx.y * blockDim.x + threadIdx.x;

    // Copy filter data to shared memory
    while (id < k * k)
    {
        filter[id] = gpu_filter[id];
        id += blockDim.x * blockDim.y;
    }
    __syncthreads();

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n)
    {
        int radius = k / 2;
        long int res = 0;

        // Convolution computation
        for (int a = 0; a < k; a++)
        {
            for (int b = 0; b < k; b++)
            {
                int row = i + a - radius;
                int col = j + b - radius;
                if (row >= 0 && col >= 0 && row < m && col < n)
                {
                    res += gpu_input[row * n + col] * filter[a * k + b];
                }
            }
        }

        // Store the result in the output matrix
        gpu_output[i * n + j] = res;
    }
}

int main(int argc, char **argv)
{
    int m, n, k;
    cin >> m >> n >> k;

    long int *h_mat = new long int[m * n];
    long int *h_filter = new long int[k * k];
    long int *h_ans = new long int[m * n];

    for (long int i = 0; i < m * n; i++)
    {
        cin >> h_mat[i];
    }

    for (long int i = 0; i < k * k; i++)
    {
        cin >> h_filter[i];
    }

    /**
     * Input Matrix and Filter Initialization
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     **/

    /***************** CUDA Initialization ********************/
    long int *gpu_mat;
    long int *gpu_filter;
    long int *gpu_output;

    cudaMalloc(&gpu_mat, m * n * sizeof(long int));
    cudaMemcpy(gpu_mat, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_output, m * n * sizeof(long int));

    cudaMalloc(&gpu_filter, k * k * sizeof(long int));
    cudaMemcpy(gpu_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now(); // Timing start just before the kernel launch

    // Configure CUDA grid and block dimensions
    int blockSizeX = 16; // Choose an appropriate block size (e.g., 16x16)
    int blockSizeY = 16;
    dim3 blockSize(blockSizeX, blockSizeY);

    int gridSizeX = (n + blockSizeX - 1) / blockSizeX; // Grid size along x-axis
    int gridSizeY = (m + blockSizeY - 1) / blockSizeY; // Grid size along y-axis
    dim3 gridSize(gridSizeX, gridSizeY);

    // Shared memory size for filter
    int sharedMemSize = k * k * sizeof(long int);

    // Launch CUDA kernel
    dkernel<<<gridSize, blockSize, sharedMemSize>>>(gpu_mat, gpu_output, gpu_filter, k, m, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now(); // Timing end just after the kernel launch

    // Copy the result from the device to the host
    cudaMemcpy(h_ans, gpu_output, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_mat);
    cudaFree(gpu_filter);
    cudaFree(gpu_output);

    // File Output Preparation
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     */

    // Write the output matrix to cuda.out
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < m; i++)
        {
            for (long int j = 0; j < n; j++)
            {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    // Write the timing information to cuda_timing.out
    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
