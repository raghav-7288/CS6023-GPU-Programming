#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__device__ unsigned num_rounds = 1; // keep track of number of rounds

// __device__ int in_sight(long long int x_firing, long long int x_middle, long long int x_target, long long int y_firing, long long int y_middle, long long int y_target)
// {
//   long long int slope1 = (x_firing - x_middle) * (y_target - y_firing);
//   long long int slope2 = (y_firing - y_middle) * (x_target - x_firing);
//   if (slope1 == slope2)
//   { // checking for colinear
//     // checking orientation
//     if (x_firing != x_target)
//     {
//       if ((x_firing > x_target && x_firing < x_middle) || (x_firing < x_target && x_firing > x_middle))
//       {
//         return 0;
//       }
//       return 1;
//     }
//     else
//     {
//       if ((y_firing > y_target && y_firing < y_middle) || (y_firing < y_target && y_firing > y_middle))
//       {
//         return 0;
//       }
//       return 1;
//     }
//   }
//   return 0;
// }

__global__ void init_arrays(int* hp_gpu, int *is_destroyed_gpu, unsigned int* score_gpu, int hp){
  hp_gpu[threadIdx.x] = hp;
  is_destroyed_gpu[threadIdx.x] = 0;
  score_gpu[threadIdx.x] = 0;
}

__global__ void calculate_num_alive_tanks(int num_tanks, int *is_destroyed_gpu, unsigned *num_alive_tanks, int *hp_gpu)
{
  // calculating no. of distroyed tanks after a round
  if (hp_gpu[threadIdx.x] < 1)
  {
    if (!is_destroyed_gpu[threadIdx.x])
    {
      is_destroyed_gpu[threadIdx.x] = 1;
      atomicSub(num_alive_tanks, 1);
    }
  }
  if (threadIdx.x == 0)
  {
    bool check = num_rounds % num_tanks == num_tanks - 1;
    if (check)
    {
      num_rounds += 2;
    }
    else
    {
      num_rounds++;
    }
  }
}

__global__ void perform_round(int num_tanks, int *pos_x_gpu, int *pos_y_gpu, int *is_destroyed_gpu, unsigned *score_gpu, int *hp_gpu)
{

  __shared__ int lock, target_tank;
  __shared__ long long int x_target_tank, y_target_tank;
  __shared__ volatile int closest_distance, closest_tank;

  if (threadIdx.x == 0) // to initialise volatile variables
  {
    lock = 0;
    target_tank = (blockIdx.x + num_rounds) % num_tanks;
    closest_tank = -1;
    x_target_tank = pos_x_gpu[target_tank];
    y_target_tank = pos_y_gpu[target_tank];
    closest_distance = INT_MAX;
  }

  __syncthreads();

  long long int x_firing_tank = pos_x_gpu[blockIdx.x];
  long long int y_firing_tank = pos_y_gpu[blockIdx.x];
  bool self_hit = (threadIdx.x == blockIdx.x);                                       // if the tank is hitting itself
  bool is_alive = (!is_destroyed_gpu[blockIdx.x] && !is_destroyed_gpu[threadIdx.x]); // if the firing tank and target tanks are undestroyed

  if (!self_hit && is_alive)
  {

    long long int x_middle_tank = pos_x_gpu[threadIdx.x];
    long long int y_middle_tank = pos_y_gpu[threadIdx.x];

    long long int slope1 = (x_firing_tank - x_middle_tank) * (y_target_tank - y_firing_tank);
    long long int slope2 = (y_firing_tank - y_middle_tank) * (x_target_tank - x_firing_tank);
    bool is_colinear = (slope1 == slope2), is_in_sight;

    if (x_firing_tank != x_target_tank)
    {
      if ((x_firing_tank > x_target_tank && x_firing_tank < x_middle_tank) || (x_firing_tank < x_target_tank && x_firing_tank > x_middle_tank))
      {
        is_in_sight = false;
      }
      else
        is_in_sight = true;
    }
    else
    {
      if ((y_firing_tank > y_target_tank && y_firing_tank < y_middle_tank) || (y_firing_tank < y_target_tank && y_firing_tank > y_middle_tank))
      {
        is_in_sight = false;
      }
      else
        is_in_sight = true;
    }

    if (is_colinear && is_in_sight)
    { // checking proper orientation i.e. if in line of fire
      // making firing tank as reference
      int x_middle = pos_x_gpu[threadIdx.x] - x_firing_tank;
      int y_middle = pos_y_gpu[threadIdx.x] - y_firing_tank;

      // implementing lock
      for (int i = 0; i < 32; i++)
      {
        // Check if the current thread's index within the warp (group of 32 threads) matches the provided index 'i'
        if (i == threadIdx.x % 32)
        {
          // Attempt to acquire a lock using atomic compare-and-swap (CAS)
          while (atomicCAS(&lock, 0, 1) != 0)
          {
            // If CAS fails (another thread already holds the lock), keep trying
          }
          int distance;
          // calculating distance in terms of target tank's x y coordinates
          if (x_middle == 0)
          {
            distance = (y_middle > 0) ? y_middle : -y_middle;
          }
          else
          {
            distance = (x_middle > 0) ? x_middle : -x_middle;
          }
          // Critical section (protected by the lock)
          if (closest_distance > distance)
          {
            // Update closest distance and tank if the current distance is smaller
            closest_distance = distance;
            closest_tank = threadIdx.x;
          }
          // Release the lock after updating within the critical section
          atomicExch(&lock, 0);
        }
      }
    }
  }
  __syncthreads();
  // updating score and hp of the firing and the closest undistroyed tank in line of fire respectively
  if (closest_tank != -1 && threadIdx.x == 0)
  {
    atomicAdd(&score_gpu[blockIdx.x], 1);
    atomicSub(&hp_gpu[closest_tank], 1);
  }
}

//***********************************************

int main(int argc, char **argv)
{
  // Variable declarations
  int M, N, T, H, *xcoord, *ycoord, *score;

  FILE *inputfilepointer;

  // File Opening for read
  char *inputfilename = argv[1];
  inputfilepointer = fopen(inputfilename, "r");

  if (inputfilepointer == NULL)
  {
    printf("input.txt file failed to open.");
    return 0;
  }

  fscanf(inputfilepointer, "%d", &M);
  fscanf(inputfilepointer, "%d", &N);
  fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
  fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

  // Allocate memory on CPU
  xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
  ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
  score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

  // Get the Input of Tank coordinates
  for (int i = 0; i < T; i++)
  {
    fscanf(inputfilepointer, "%d", &xcoord[i]);
    fscanf(inputfilepointer, "%d", &ycoord[i]);
  }

  auto start = chrono::high_resolution_clock::now();

  //***************************************************************************************************
  // Your Code begins here (Do not change anything in main() above this comment)
  //**************************************************************************************************

  // pinned memory to keep count of num of distroyed tanks so far
  int *hp_gpu, *hp_cpu = (int *)malloc(T * sizeof(int));
  unsigned *num_alive_tanks;
  cudaHostAlloc(&num_alive_tanks, sizeof(unsigned), 0);
  *num_alive_tanks = T;

  int *pos_x_gpu, *pos_y_gpu, *is_destroyed_gpu;
  unsigned *score_gpu;

  cudaMalloc(&hp_gpu, T * sizeof(int));
  cudaMalloc(&pos_x_gpu, T * sizeof(int));
  cudaMalloc(&pos_y_gpu, T * sizeof(int));
  cudaMalloc(&is_destroyed_gpu, T * sizeof(int));
  cudaMalloc(&score_gpu, T * sizeof(unsigned));

  cudaMemcpy(hp_gpu, hp_cpu, T * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pos_x_gpu, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pos_y_gpu, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemset(&is_destroyed_gpu, 0, T * sizeof(int));
  // cudaMemset(&score_gpu, 0, T * sizeof(int));

  init_arrays<<<1,T>>>(hp_gpu, is_destroyed_gpu, score_gpu, H);

  // calling kernels until the alive tank remain 0 or 1
  while (*num_alive_tanks > 1)
  {
    perform_round<<<T, T>>>(T, pos_x_gpu, pos_y_gpu, is_destroyed_gpu, score_gpu, hp_gpu);
    calculate_num_alive_tanks<<<1, T>>>(T, is_destroyed_gpu, num_alive_tanks, hp_gpu);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(score, score_gpu, sizeof(int) * T, cudaMemcpyDeviceToHost);

  cudaFree(pos_x_gpu);
  cudaFree(pos_y_gpu);
  cudaFreeHost(num_alive_tanks);
  cudaFree(is_destroyed_gpu);
  cudaFree(hp_gpu);
  cudaFree(score_gpu);

  //**************************************************************************************************
  // Your Code ends here (Do not change anything in main() below this comment)
  //**************************************************************************************************

  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double, std::micro> timeTaken = end - start;

  printf("Execution time : %f\n", timeTaken.count());

  // Output
  char *outputfilename = argv[2];
  char *exectimefilename = argv[3];
  FILE *outputfilepointer;
  outputfilepointer = fopen(outputfilename, "w");

  for (int i = 0; i < T; i++)
  {
    fprintf(outputfilepointer, "%d\n", score[i]);
  }
  fclose(inputfilepointer);
  fclose(outputfilepointer);

  outputfilepointer = fopen(exectimefilename, "w");
  fprintf(outputfilepointer, "%f", timeTaken.count());
  fclose(outputfilepointer);

  free(xcoord);
  free(ycoord);
  free(score);
  cudaDeviceSynchronize();
  return 0;
}