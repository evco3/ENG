//Evan Cohen 20214967


#include "cuda_runtime.h"
#include <stdio.h>


int totalCores(int minor, int major) {
      switch (major) {
      case 2:
            return minor == 1 ? 48 : 32;
            break;
      case 3:
            return 192;
            break;
      case 5:
            return 128;
            break;
      case 6:
            if ((minor == 1) || (minor == 2)) return 128;
            else if (minor == 0) return 64;
            break;
      case 7:
            if ((minor == 0) || (minor == 5)) return 64;
            break;
      }
      return -1;
}

int main() {
      int totalDevices;
      cudaGetDeviceCount(&totalDevices);
      printf("Number of CUDA devices on GPU servers: &d\n", totalDevices);

      for (int i = 0; i < totalDevices; i++) {
            cudaDeviceProp device;
            cudaGetDeviceProperties(&device, i);

            printf("\nDevice %d: %s\n", i, device.name);
            printf("Clock Rate: %d KHz\n", device.clockRate);
            printf("Number of Streaming Multiprocessors: %d\n", device.multiProcessorCount);
            printf("Number of Cores: %d\n", device.multiProcessorCount);
            printf("Warp Size: %d\n", device.warpSize);
            printf("Global Memory: %lu b\n", device.totalGlobalMem);
            printf("Shared Memory pre Block: %lu b\n", device.sharedMemPerBlock);
            printf("Number of Registers pre Block: %d\n", device.regsPerBlock);
            printf("Maximum Number of Threads per Block: %d\n", device.maxThreadsPerBlock);
            printf("Maximum Size of Each Dimension of a Block: (%d, %d, %d)\n", device.maxThreadsDim[0], device.maxThreadsDim[1], device.maxThreadsDim[2]);
            printf("Maximum Size of Each Dimension of a Grid: (%d, %d, %d)\n", device.maxGridSize[0], device.maxGridSize[1], device.maxGridSize[2]);
      }
}