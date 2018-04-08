/*******************************************************************************************
 * Filename: functions_misc.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains helper functions - simple array operations, dealing with
 *              critical errors, printing insults, increasing/decreasing thread size, etc.
 *******************************************************************************************
 */

/*
 * getDeviceProperties
 * @params: multiProcessorCount - a pointer an int value (stores multiProcessorCount of the device)
 * @params: warpSize - a pointer an int value (stores the warpSize of the device)
 */
void getDeviceProperties(int* multiProcessorCount, int* warpSize) {
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
int getOptimalThreadSize(int blocks, int threads, int minimumThreadsNeeded, int gpuWarpsize) {
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
 * initArrayToRandomDoubles
 * @params: a - a pointer to an array of double values
 * @params: n - the size of array a
 */
void initArrayToRandomDoubles(double** a, int n) {
    // generate random doubles in range [0, 1)
    for (int i = 0; i < n; i++) {
        srand (time(NULL)); // seed pseudo-random number generator with current time
(        *a)[i] = ((double) rand()) / ((double) RAND_MAX);
    }
} //end initArrayToRandomDoubles function

/*
 * initArrayToZeros
 * @params: a - a pointer to an array of double values
 * @params: n - the size of array a
 */
void initArrayToZeros(double** a, int n) {
    // set all neuron values to zero
    for (int i = 0; i < n; i++) {
        (*a)[i] = 0;
    }
} //end initArrayToZeros function

/*
 * printarray_double - prints out array double values to terminal
 * @params: name - a pointer to a char string
 * @params: a - a pointer to an array of double values
 * @params: n - the size of array a
 */
void printarray_double(const char* name, double* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%s[%d]=%lf\n", name, i, a[i]);
    }
    printf("\n");
} //end printarray function

/*
 * printarray_int - prints out array int values to terminal
 * @params: name - a pointer to a char string
 * @params: a - a pointer to an array of int values
 * @params: n - the size of array a
 */
void printarray_int(const char* name, int* a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%s[%d]=%d\n", name, i, a[i]);
    }
    printf("\n");
} //end printarray function

/*
 * printFarewellMSG - prints out one final insult before we crash
 */
void printFarewellMSG() {
    printf("Sorry, I did everything I could but it looks like I'm crashing...\n...\n...your computer sucks, good-bye.\n");
} //end printFarewellMSG function

void onCudaKernelLaunchFailure(char* kernel, cudaError_t cudaStatus) {
    printf("%s launch failed: %s\n", kernel, cudaGetErrorString(cudaStatus));
    printFarewellMSG();
    exit(1);
}

void onCudaDeviceSynchronizeError(char* kernel, cudaError_t cudaStatus) {
    printf("cudaDeviceSynchronize returned error code %d after launching %s!\n", cudaStatus, kernel);
    printFarewellMSG();
    exit(1);
}

/*
 * onCudaMallocError - SOS, we're going down
 * @params: size - the size of the device memory that we couldn't allocate
 */
void onCudaMallocError(int size) {
    printf("ERROR: Failed to cudaMalloc %d of memory!\n", size);
    printFarewellMSG();
    exit(1);
} //end onCudaMallocError function

/*
 * onCudaMemcpyError - SOS, we're going down
 * @params: size - the name of the host variable that we couldn't copy
 */
void onCudaMemcpyError(const char* hostVariable) {
    printf("ERROR: Failed to cudaMemcpy %s to device!\n", hostVariable);
    printFarewellMSG();
    exit(1);
} //end onCudaMemcpyError function

/*
 * onFailToSetGPUDevice - SOS, we're going down
 */
void onFailToSetGPUDevice() {
    printf("ERROR: Failed find GPU device!\n");
    printFarewellMSG();
    exit(1);
} //end onFailToSetGPUDevice function

/*
 * onFileOpenError - SOS, we're going down
 * @params: path - file that failed to open
 */
void onFileOpenError(const char* path) {
    printf("ERROR: Failed to open %s!\n", path);
    printFarewellMSG();
    exit(1);
} //end onFileOpenError function

/*
 * onFileReadError - SOS, we're going down
 * @params: path - file that failed to read
 */
void onFileReadError(const char* path) {
    printf("ERROR: Failed to read value from file %s!\n", path);
    printFarewellMSG();
    exit(1);
} //end onFileReadError function

/*
 * onInvalidInput - prints out insults when the user screws up (silly humans)
 * @params: myPatience - the current state of my patience, represented as an int
 */
void onInvalidInput(int myPatience) {
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
 * onMallocError - SOS, we're going down
 * @params: size - the size of the memory that we couldn't allocate
 */
void onMallocError(int size) {
    printf("ERROR: Failed to malloc %d of memory!\n", size);
    printFarewellMSG();
    exit(1);
} //end onMallocError function
