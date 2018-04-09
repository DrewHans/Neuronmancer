/*******************************************************************************************
 * Filename: functions_misc.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains helper functions - simple array operations, dealing with
 *              critical errors, printing insults, increasing/decreasing thread size, etc.
 *******************************************************************************************
 */

/*
 * getDeviceProperties - detects and stores the number of SMs and warpsize in arguments passed in
 * @params: multiProcessorCount - a pointer an int value (stores multiProcessorCount of the device)
 * @params: warpSize - a pointer an int value (stores the warpSize of the device)
 */
void getDeviceProperties(unsigned int* multiProcessorCount, unsigned int* warpSize) {
    cudaDeviceProp devProp; //initialize cudaDeviceProp struct
    cudaGetDeviceProperties(&devProp, 0); //getDeviceProperties of device 0 and stuff them into address of devProp

    /*
     //basic device information
     printf("Name:                                   %s\n", devProp.name);
     printf("Major revision number:                  %d\n", devProp.major);
     printf("Minor revision number:                  %d\n", devProp.minor);

     //grid, block, thread info
     printf("Clock rate:                             %d kHz\n", devProp.clockRate);
     printf("Number of multiprocessors:              %d multiprocessors\n", devProp.multiProcessorCount);
     printf("Warp size:                              %d threads\n", devProp.warpSize);
     printf("Maximum threads per block:              %d threads\n", devProp.maxThreadsPerBlock);
     for (int i = 0; i < 3; ++i)
     printf("Maximum dimension %d of block:          %d\n", i, devProp.maxThreadsDim[i]);
     for (int i = 0; i < 3; ++i)
     printf("Maximum dimension %d of grid:           %d\n", i, devProp.maxGridSize[i]);

     //memory info
     printf("Total registers per multiprocessor:     %d 32-bits each\n", devProp.regsPerMultiprocessor);
     printf("Total registers per block:              %d 32-bits each\n", devProp.regsPerBlock);
     printf("Total shared memory per block:          %lu bytes\n", devProp.sharedMemPerBlock);
     printf("Total global memory:                    %lu bytes\n", devProp.totalGlobalMem);

     printf("Maximum memory pitch:                   %lu\n", devProp.memPitch);
     printf("Total constant memory:                  %lu bytes\n", devProp.totalConstMem);

     //other info
     printf("Texture alignment:                      %lu\n", devProp.textureAlignment);
     printf("Concurrent copy and execution:          %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
     printf("Kernel execution timeout:               %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
     */

    *multiProcessorCount = devProp.multiProcessorCount;
    *warpSize = devProp.warpSize;
} //end getDeviceProperties function

/*
 * getOptimalThreadSize - finds the "optimal" number of threads 
 * @params: blocks - the int number of blocks we plan to launch on GPU
 * @params: threads - the int number of threads we have set to launch on GPU
 * @params: minimumThreadsNeeded - the int minimum number of threads needed
 * @params: gpuWarpsize - the int warpsize of the GPU
 * @returns: the int number of "optimal" threads to launch
 */
int getOptimalThreadSize(const unsigned int blocks, unsigned int threads, const unsigned int minimumThreadsNeeded, const unsigned int gpuWarpsize) {
    // double or devide the number of threads until we have a number close to the number of neurons in right-layer
    if ((blocks*threads) < minimumThreadsNeeded) {
        while((blocks*threads) < minimumThreadsNeeded) {
            threads = threads * 2;
        }
    } else if ((threads > gpuWarpsize) && ((blocks*(threads/2)) > minimumThreadsNeeded)) {
        while((threads > gpuWarpsize) && ((blocks*(threads/2)) > minimumThreadsNeeded)) {
            threads = threads / 2;
        }
    }
    return threads;
}//end getOptimalThreadSize function

/*
 * initArrayToRandomFloats - initializes all array elements to a random floating-point value
 * @params: a - a pointer to an array of float values
 * @params: n - the size of array a
 */
void initArrayToRandomFloats(float** a, const unsigned int n) {
    // seed pseudo-random number generator with current time
    srand ( time ( NULL));

    // generate random floats in range [0.0, 1.0)
    for (int i = 0; i < n; i++) {
        (*a)[i] = ((float) rand()) / ((float) RAND_MAX);
    }
} //end initArrayToRandomFloats function

/*
 * initArrayToZeros - initializes all array elements to zero
 * @params: a - a pointer to an array of float values
 * @params: n - the size of array a
 */
void initArrayToZeros(float** a, const unsigned int n) {
    // set all neuron values to zero
    for (int i = 0; i < n; i++) {
        (*a)[i] = 0.0;
    }
} //end initArrayToZeros function

/*
 * initDeviceArrayToZeros - initializes all device-array elements to zero
 * @params: devA - device copy of a float array
 * @params: n - the size of devA
 */
void initDeviceArrayToZeros(float* devA, const unsigned int n) {
    // use getDeviceProperties helper function to get GPU device information
    unsigned int numberOfSMs = 0; // the number of SMs on the device (1 SM can process 1 block at a time)
    unsigned int warpsize = 0; // the number of threads that an SM can manage at one time
    getDeviceProperties(&numberOfSMs, &warpsize); 

    // set blocks and threads to a size that will fully utilize the GPU (overkill, I know, but we're going for performance here)
    unsigned int blocks = numberOfSMs; // should be equal to the number of SMs on the GPU device after getDeviceProperties
    unsigned int threads = warpsize; // should be equal to the warpsize on the GPU device after getDeviceProperties
    
    // double or devide the number of threads until we have a number close to the size of the array
    threads = getOptimalThreadSize(blocks, threads, n, warpsize);

    // calculate the deltas for each neuron no in the output-layer
    cudaKernel_initArrayToZeros<<<blocks, threads>>>(devA, n);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        onCudaKernelLaunchFailure("cudaKernel_initArrayToZeros", cudaStatus)
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        onCudaDeviceSynchronizeError("cudaKernel_initArrayToZeros", cudaStatus);
    }
} //end initArrayToZeros function

/*
 * cudaKernel_initArrayToZeros - initializes all device-array elemtns to zero
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devA - device copy of a float array
 * @params: n - the size of devA
 */
__global__ static void cudaKernel_initArrayToZeros(float* devA, const unsigned int n) {
    volatile unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        devA[id] = 0.0;
    }
}//end cudaKernel_initArrayToZeros function

/*
 * printarray_float - prints out all array elements to terminal
 * @params: name - a pointer to a char string
 * @params: a - a pointer to an array of float values
 * @params: n - the size of array a
 */
void printarray_float(const char* name, const float* a, const unsigned int n) {
    for (int i = 0; i < n; i++) {
        printf("%s[%d]=%lf\n", name, i, a[i]);
    }
    printf("\n");
} //end printarray_float function

/*
 * printarray_int - prints out all array elements to terminal
 * @params: name - a pointer to a char string
 * @params: a - a pointer to an array of int values
 * @params: n - the size of array a
 */
void printarray_int(const char* name, const unsigned int* a, const unsigned int n) {
    for (int i = 0; i < n; i++) {
        printf("%s[%d]=%d\n", name, i, a[i]);
    }
    printf("\n");
} //end printarray_int function

/*
 * printFarewellMSG - prints out one final insult before we crash
 */
void printFarewellMSG() {
    printf("Sorry, I did everything I could but it looks like I'm crashing...\n...\n...your computer sucks, good-bye.\n");
} //end printFarewellMSG function

/*
 * onCudaKernelLaunchFailure - crashes the program when called (SOS, we're going down)
 */
void onCudaKernelLaunchFailure(const char* kernel, const cudaError_t cudaStatus) {
    fprintf(stderr, "ERROR: %s launch failed: %s\n", kernel, cudaGetErrorString(cudaStatus));
    printFarewellMSG();
    exit(1);
}//end onCudaKernelLaunchFailure function

/*
 * onCudaDeviceSynchronizeError - crashes the program when called (SOS, we're going down)
 */
void onCudaDeviceSynchronizeError(const char* kernel, const cudaError_t cudaStatus) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching %s!\n", cudaStatus, kernel);
    printFarewellMSG();
    exit(1);
}//end onCudaDeviceSynchronizeError function

/*
 * onCudaMallocError - crashes the program when called (SOS, we're going down)
 * @params: size - the size of the device memory that we couldn't allocate
 */
void onCudaMallocError(const unsigned int size) {
    fprintf(stderr, "ERROR: Failed to cudaMalloc %d of memory!\n", size);
    printFarewellMSG();
    exit(1);
} //end onCudaMallocError function

/*
 * onCudaMemcpyError - crashes the program when called (SOS, we're going down)
 * @params: size - the name of the host variable that we couldn't copy
 */
void onCudaMemcpyError(const char* hostVariable) {
    fprintf(stderr, "ERROR: Failed to cudaMemcpy %s to device!\n", hostVariable);
    printFarewellMSG();
    exit(1);
} //end onCudaMemcpyError function

/*
 * onFailToSetGPUDevice - crashes the program when called (SOS, we're going down)
 */
void onFailToSetGPUDevice() {
    fprintf(stderr, "ERROR: Failed to find a CUDA enabled GPU device!\n");
    printFarewellMSG();
    exit(1);
} //end onFailToSetGPUDevice function

/*
 * onFileOpenError - crashes the program when called (SOS, we're going down)
 * @params: path - file that failed to open
 */
void onFileOpenError(const char* path) {
    fprintf(stderr, "ERROR: Failed to open %s!\n", path);
    printFarewellMSG();
    exit(1);
} //end onFileOpenError function

/*
 * onFileReadError - crashes the program when called (SOS, we're going down)
 * @params: path - file that failed to read
 */
void onFileReadError(const char* path) {
    fprintf(stderr, "ERROR: Failed to read from file %s!\n", path);
    printFarewellMSG();
    exit(1);
} //end onFileReadError function

/*
 * onInvalidInput - prints out insults when the user screws up (silly humans)
 * @params: myPatience - the current state of my patience, represented as an int
 */
void onInvalidInput(const int myPatience) {
    if (myPatience == 2) {
        printf("Looks like you entered an illegal value... you're testing my patience, try again!\n\n");
    } else if (myPatience == 1) {
        printf("That's the second time you've entered an illegal value... do you think this is funny? Try again!\n\n");
    } else if (myPatience == 0) {
        printf("Sigh... you just can't do anything right, can you?\n\n");
    } else {
        printf("Look dude, I've got all day. If you wanna keep wasting your time then that's fine by me. You know what you're supposed to do.\n\n");
    }
} //end onInvalidInput function

/*
 * onMallocError - crashes the program when called (SOS, we're going down)
 * @params: size - the size of the memory that we couldn't allocate
 */
void onMallocError(const unsigned int size) {
    fprintf(stderr, "ERROR: Failed to malloc %d of memory!\n", size);
    printFarewellMSG();
    exit(1);
} //end onMallocError function

