/*******************************************************************************
 * Filename: functions_cuda.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to train networks on
 *              cuda-enabled GPU devices.
 *******************************************************************************
 */

#include "main.h"

// loads the MNIST training set images and labels into constant memory
void cuda_loadMNISTTrainingSetToDevice()
{
    const size_t trainImagesSize = sizeof(MNIST_Image) * MNIST_TRAINING_SET_SIZE;
    const size_t trainLabelsSize = sizeof(MNIST_Label) * MNIST_TRAINING_SET_SIZE;

    // malloc host memory for MNIST training set data
    MNIST_Image *trainImages = (MNIST_Image *)malloc(trainImagesSize);
    if (trainImages == NULL)
    {
        printf("Abort! Could not malloc memory to store trainImages!\n");
        exit(1);
    }

    MNIST_Label *trainLabels = (MNIST_Label *)malloc(trainLabelsSize);
    if (trainLabels == NULL)
    {
        printf("Abort! Could not malloc memory to store trainLabels!\n");
        exit(1);
    }

    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGES_PATH);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABELS_PATH);

    // copy file data into host variables
    for (int sample = 0; sample < MNIST_TRAINING_SET_SIZE; sample++)
    {
        // read the next sample image and label
        trainImages[sample] = getImage(imageFile);
        trainLabels[sample] = getLabel(labelFile);
    }

    // declare cudaStatus variable to check for success of cuda operations
    cudaError_t cudaStatus;

    // copy host memory values into device global memory variables
    cudaStatus = cudaMemcpyToSymbol(dev_trainImages, trainImages, trainImagesSize, 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpyToSymbol trainImages!\n");
        exit(1);
    }

    cudaStatus = cudaMemcpyToSymbol(dev_trainLabels, trainLabels, trainLabelsSize, 0, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpyToSymbol trainLabels!\n");
        exit(1);
    }

    free(trainImages);
    free(trainLabels);
}

// returns the Sigmoid activation of x
__device__ float cuda_sigmoid(const float x)
{
    return (1.0 / (1.0 + exp((x * -1.0))));
}

// returns the Sigmoid derivative of x
__device__ float cuda_sigmoidPrime(const float x)
{
    return cuda_sigmoid(x) * (1.0 - cuda_sigmoid(x));
}

// performs a single training cycle:
// one feedforward, one backpropagation, one weight & bias update
void cuda_trainNetwork(InputLayer *dev_il, HiddenLayer *dev_hl,
                       OutputLayer *dev_ol, ExpectedOutput *dev_expected,
                       int sample,
                       unsigned int iBlocks, unsigned int iThreads,
                       unsigned int hBlocks, unsigned int hThreads,
                       unsigned int oBlocks, unsigned int oThreads)
{

    // (A) Feedforward Step
    // (A1) FeedInputLayer
    cudakernel_feedInputLayer<<<iBlocks, iThreads>>>(dev_il, sample);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_feedInputLayer failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_feedInputLayer failed to synchronize!\n");
        exit(1);
    }

    // (A2) FeedHiddenLayer
    cudakernel_feedHiddenLayer<<<hBlocks, hThreads>>>(dev_hl, dev_il);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_feedHiddenLayer failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_feedHiddenLayer failed to synchronize!\n");
        exit(1);
    }

    // (A3) FeedOutputLayer
    cudakernel_feedOutputLayer<<<oBlocks, oThreads>>>(dev_ol, dev_hl);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_feedOutputLayer failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_feedOutputLayer failed to synchronize!\n");
        exit(1);
    }

    // (B) Backpropagation Step
    cudakernel_getExpectedOutput<<<oBlocks, oThreads>>>(dev_expected, sample);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_getExpectedOutput failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_getExpectedOutput failed to synchronize!\n");
        exit(1);
    }

    cudakernel_calculateOutputLayerDeltas<<<oBlocks, oThreads>>>(dev_ol, dev_expected);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_calculateOutputLayerDeltas failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_calculateOutputLayerDeltas failed to synchronize!\n");
        exit(1);
    }

    cudakernel_calculateHiddenLayerDeltas<<<hBlocks, hThreads>>>(dev_hl, dev_ol);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_calculateHiddenLayerDeltas failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_calculateHiddenLayerDeltas failed to synchronize!\n");
        exit(1);
    }

    // (C) Update Weights & Biases
    // (C1) update HiddenLayer's weights and biases
    cudakernel_updateHiddenLayerWeightsAndBiases<<<hBlocks, hThreads>>>(dev_hl, dev_il);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_updateHiddenLayerWeightsAndBiases failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_updateHiddenLayerWeightsAndBiases failed to synchronize!\n");
        exit(1);
    }

    // (C2) update OutputLayer's weights and biases
    cudakernel_updateOutputLayerWeightsAndBiases<<<oBlocks, oThreads>>>(dev_ol, dev_hl);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_updateOutputLayerWeightsAndBiases failed to launch!\n");
        exit(1);
    }

    cudaStatus = cudaDeviceSynchronize(); // waits for the kernel to finish
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! cudakernel_updateOutputLayerWeightsAndBiases failed to synchronize!\n");
        exit(1);
    }
}

// trains a neural network using a CUDA enabled GPU device
void cuda_train(InputLayer *il, HiddenLayer *hl, OutputLayer *ol)
{
    // declare cudaStatus variable to check for success of cuda operations
    cudaError_t cudaStatus;

    // run on GPU 0, this will need to be changed on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not detect a CUDA-enabled GPU!\n");
        exit(1);
    }

    cuda_loadMNISTTrainingSetToDevice();

    // declare helper structs on the device
    InputLayer *dev_il;
    HiddenLayer *dev_hl;
    OutputLayer *dev_ol;
    ExpectedOutput *dev_expected;

    // allocate device memory
    cudaStatus = cudaMalloc((void **)&dev_il, sizeof(*dev_il));
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMalloc device memory to store dev_il!\n");
        exit(1);
    }

    cudaStatus = cudaMalloc((void **)&dev_hl, sizeof(*dev_hl));
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMalloc device memory to store dev_hl!\n");
        exit(1);
    }

    cudaStatus = cudaMalloc((void **)&dev_ol, sizeof(*dev_ol));
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMalloc device memory to store dev_ol!\n");
        exit(1);
    }

    cudaStatus = cudaMalloc((void **)&dev_expected, sizeof(*dev_expected));
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMalloc device memory to store dev_expected!\n");
        exit(1);
    }

    // copy host variable values into device copies
    cudaStatus = cudaMemcpy(dev_il, il, sizeof(*il), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy il to dev_il!\n");
        exit(1);
    }
    cudaStatus = cudaMemcpy(dev_il->input, il->input, sizeof(uint8_t) * INPUT_LAYER_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy il->input to dev_il->input!\n");
        exit(1);
    }

    cudaStatus = cudaMemcpy(dev_hl, hl, sizeof(*hl), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy hl to dev_hl!\n");
        exit(1);
    }
    cudaStatus = cudaMemcpy(dev_hl->hNeuron, hl->hNeuron, sizeof(HLNeuron) * HIDDEN_LAYER_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy hl->hNeuron to dev_hl->hNeuron!\n");
        exit(1);
    }

    cudaStatus = cudaMemcpy(dev_ol, ol, sizeof(*ol), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy ol to dev_ol!\n");
        exit(1);
    }
    cudaStatus = cudaMemcpy(dev_ol->oNeuron, ol->oNeuron, sizeof(OLNeuron) * OUTPUT_LAYER_SIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy ol->oNeuron to dev_ol->oNeuron!\n");
        exit(1);
    }

    // declare variables for holding the "optimal" number of blocks / threads for a given layer
    unsigned int iBlocks, iThreads; // for input-layer cudakernels
    unsigned int hBlocks, hThreads; // for hidden-layer cudakernels
    unsigned int oBlocks, oThreads; // for output-layer cudakernels

    // calculate the "optimal" number of blocks / threads for each layer
    getOptimalBlocksAndThreads(&iBlocks, &iThreads, INPUT_LAYER_SIZE);
    getOptimalBlocksAndThreads(&hBlocks, &hThreads, HIDDEN_LAYER_SIZE);
    getOptimalBlocksAndThreads(&oBlocks, &oThreads, OUTPUT_LAYER_SIZE);

    // begin training

    printf("\n--- beginning training on GPU device ---\n");

    // for each epoch
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        printf(" --- starting epoch %d of %d ---\n", epoch + 1, EPOCHS);

        // for each MNIST sample in the training set
        for (int sample = 0; sample < MNIST_TRAINING_SET_SIZE; sample++)
        {

            cuda_trainNetwork(dev_il, dev_hl, dev_ol,
                              dev_expected, sample,
                              iBlocks, iThreads,
                              hBlocks, hThreads,
                              oBlocks, oThreads);

            if ((sample + 1) % 10000 == 0)
            {
                printf("    => sample %d of %d complete\n", sample + 1, MNIST_TRAINING_SET_SIZE);
            }
        }

        printf(" --- epoch %d of %d complete ---\n", epoch + 1, EPOCHS);
    }

    printf("\n--- training on GPU device complete ---\n\n");

    // copy device variable values into host variables
    cudaStatus = cudaMemcpy(il, dev_il, sizeof(*dev_il), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy dev_il to il!\n");
        exit(1);
    }

    cudaStatus = cudaMemcpy(hl, dev_hl, sizeof(*dev_hl), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy dev_hl to hl!\n");
        exit(1);
    }

    cudaStatus = cudaMemcpy(ol, dev_ol, sizeof(*dev_ol), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        printf("Abort! Could not cudaMemcpy dev_ol to ol!\n");
        exit(1);
    }

    cudaFree(dev_il);
    cudaFree(dev_hl);
    cudaFree(dev_ol);
    cudaFree(dev_expected);
}

// calculates the delta values for output-layer neurons
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_calculateHiddenLayerDeltas(
    HiddenLayer *__restrict__ dev_hl,
    OutputLayer *__restrict__ dev_ol)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < HIDDEN_LAYER_SIZE)
    {
        dev_hl->hNeuron[id].delta = 0.0; // clear out previous delta value

        // for each oNeuron in OutputLayer
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
            // propagate ol->oNeuron[i]'s delta backwards
            dev_hl->hNeuron[id].delta += dev_ol->oNeuron[i].weight[id] * dev_ol->oNeuron[i].delta;
        }

        // calculate hl->hNeuron[i]'s delta
        dev_hl->hNeuron[id].delta = cuda_sigmoidPrime(dev_hl->hNeuron[id].weightedSum) * dev_hl->hNeuron[id].delta;
    }
}

// calculates the delta values for output-layer neurons
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_calculateOutputLayerDeltas(
    OutputLayer *__restrict__ dev_ol,
    ExpectedOutput *__restrict__ dev_expected)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < OUTPUT_LAYER_SIZE)
    {
        dev_ol->oNeuron[id].delta = (cuda_sigmoidPrime(dev_ol->oNeuron[id].weightedSum) * (dev_expected->value[id] - dev_ol->oNeuron[id].output));
    }
}

// feeds MNIST pixel values into the input-layer
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_feedInputLayer(
    InputLayer *__restrict__ dev_il,
    int sample)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < INPUT_LAYER_SIZE)
    {
        // if dev_image->pixel[i] !0 then set dev_il->input[i] to 1, else set to 0
        dev_il->input[id] = (dev_trainImages[sample].pixel[id] ? 1 : 0);
    }
}

// feeds input-layer values into hidden-layer
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_feedHiddenLayer(
    HiddenLayer *__restrict__ dev_hl,
    InputLayer *__restrict__ dev_il)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < HIDDEN_LAYER_SIZE)
    {
        dev_hl->hNeuron[id].weightedSum = 0.0; // clear out previous weightedSum

        // for each input[i] to HLNeuron, add il->input[i] * hNeuron[id].weight[i] to HLNeuron's weighted sum
        for (int i = 0; i < INPUT_LAYER_SIZE; i++)
        {
            dev_hl->hNeuron[id].weightedSum += dev_il->input[i] * dev_hl->hNeuron[id].weight[i];
        }

        // apply sigmoid activation to hNeuron's weighted sum plus bias
        dev_hl->hNeuron[id].output = cuda_sigmoid(dev_hl->hNeuron[id].weightedSum + dev_hl->hNeuron[id].bias);
    }
}

// feeds hidden-layer values into output-layer
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_feedOutputLayer(
    OutputLayer *__restrict__ dev_ol,
    HiddenLayer *__restrict__ dev_hl)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < OUTPUT_LAYER_SIZE)
    {
        dev_ol->oNeuron[id].weightedSum = 0.0; // clear out previous weightedSum

        // for each input[i] to HLNeuron, add il->input[i] * hNeuron[id].weight[i] to HLNeuron's weighted sum
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
        {
            dev_ol->oNeuron[id].weightedSum += dev_hl->hNeuron[i].output * dev_ol->oNeuron[id].weight[i];
        }

        // apply sigmoid activation to the hln's weighted sum plus bias
        dev_ol->oNeuron[id].output = cuda_sigmoid(dev_ol->oNeuron[id].weightedSum + dev_ol->oNeuron[id].bias);
    }
}

// fills dev_expected with the expected output values
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_getExpectedOutput(
    ExpectedOutput *__restrict__ dev_expected,
    int sample)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < OUTPUT_LAYER_SIZE)
    {
        // if id == dev_trainLabels[sample] set dev_expected->value[i] to 1, else set to 0
        dev_expected->value[id] = (id == dev_trainLabels[sample] ? 1 : 0);
    }
}

// updates the HiddenLayer hNeuron's weights and biases
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_updateHiddenLayerWeightsAndBiases(
    HiddenLayer *__restrict__ dev_hl,
    InputLayer *__restrict__ dev_il)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < HIDDEN_LAYER_SIZE)
    {
        // update each weight between InputLayer and hNeuron[id]
        for (int i = 0; i < INPUT_LAYER_SIZE; i++)
        {
            dev_hl->hNeuron[id].weight[i] += LEARNING_RATE * dev_il->input[i] * dev_hl->hNeuron[id].delta;
        }

        // update hNeuron[id]'s bias
        dev_hl->hNeuron[id].bias += LEARNING_RATE * dev_hl->hNeuron[id].delta;
    }
}

// updates the OutputLayer oNeuron's weights and biases
// __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
// __restrict__ decoration tells NVCC this pointer will only be used to refer to the underlying data
__global__ void cudakernel_updateOutputLayerWeightsAndBiases(
    OutputLayer *__restrict__ dev_ol,
    HiddenLayer *__restrict__ dev_hl)
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id

    // check that thread id is within our desired range (extra threads may have been launched for GPU optimization)
    if (id < OUTPUT_LAYER_SIZE)
    {
        // update each weight between HiddenLayer and oNeuron[id]
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
        {
            dev_ol->oNeuron[id].weight[i] += LEARNING_RATE * dev_hl->hNeuron[i].output * dev_ol->oNeuron[id].delta;
        }

        // update oNeuron[id]'s bias
        dev_ol->oNeuron[id].bias += LEARNING_RATE * dev_ol->oNeuron[id].delta;
    }
}

// detects and stores the number of SMs and warpsize in arguments passed in
void getDeviceProperties(
    unsigned int *multiProcessorCount,
    unsigned int *warpSize,
    unsigned int *maxThreadsPerBlock)
{
    cudaDeviceProp devProp;               //initialize cudaDeviceProp struct
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
    *maxThreadsPerBlock = devProp.maxThreadsPerBlock;
}

//finds the "optimal" number of blocks and threads needed to launch a kernel
void getOptimalBlocksAndThreads(
    unsigned int *blocks,
    unsigned int *threads,
    const unsigned int minimumThreadsNeeded)
{
    // get relevant information about GPU device
    unsigned int gpuMultiProcessorCount = 0;
    unsigned int gpuWarpSize = 0;
    unsigned int gpuMaxThreadsPerBlock = 0;
    getDeviceProperties(&gpuMultiProcessorCount, &gpuWarpSize, &gpuMaxThreadsPerBlock);

    // first 5 if statements should be sufficient for small kernel launches
    if (gpuWarpSize > minimumThreadsNeeded)
    {
        // set the minimum blocks and threads that we must have for any cudaKernel
        (*blocks) = 1;
        (*threads) = gpuWarpSize;
    }
    else if (gpuWarpSize * 2 > minimumThreadsNeeded && gpuWarpSize * 2 < gpuMaxThreadsPerBlock)
    {
        (*blocks) = 1;
        (*threads) = gpuWarpSize * 2;
    }
    else if (gpuWarpSize * 4 > minimumThreadsNeeded && gpuWarpSize * 4 < gpuMaxThreadsPerBlock)
    {
        (*blocks) = 1;
        (*threads) = gpuWarpSize * 4;
    }
    else if (gpuWarpSize * 8 > minimumThreadsNeeded && gpuWarpSize * 8 < gpuMaxThreadsPerBlock)
    {
        (*blocks) = 1;
        (*threads) = gpuWarpSize * 8;
    }
    else if (gpuWarpSize * 16 > minimumThreadsNeeded && gpuWarpSize * 16 < gpuMaxThreadsPerBlock)
    {
        // start adding blocks, probably easier than doubling the threads at this point
        (*blocks) = 1;
        (*threads) = gpuWarpSize * 16;
    }
    else if (gpuWarpSize * 16 > minimumThreadsNeeded && gpuWarpSize * 8 < gpuMaxThreadsPerBlock)
    {
        // start adding blocks, probably easier than doubling the threads at this point
        (*blocks) = 2;
        (*threads) = gpuWarpSize * 8;
    }
    else if (gpuWarpSize * 32 > minimumThreadsNeeded && gpuWarpSize * 8 < gpuMaxThreadsPerBlock)
    {
        // add another two blocks, still probably easier than doubling the threads
        (*blocks) = 4;
        (*threads) = gpuWarpSize * 8;
    }
    else if (gpuWarpSize * 64 > minimumThreadsNeeded && gpuWarpSize * 16 < gpuMaxThreadsPerBlock)
    {
        // okay, guess we're doubling threads then...
        (*blocks) = 4;
        (*threads) = gpuWarpSize * 16;
    }
    else if (gpuWarpSize * 128 > minimumThreadsNeeded && gpuWarpSize * 16 < gpuMaxThreadsPerBlock)
    {
        // okay, guess we're adding blocks and doubling threads...
        (*blocks) = 8;
        (*threads) = gpuWarpSize * 16;
    }
    else if (gpuWarpSize * 256 > minimumThreadsNeeded && gpuWarpSize * 16 < gpuMaxThreadsPerBlock)
    {
        // okay, guess we're adding blocks and doubling threads...
        (*blocks) = 16;
        (*threads) = gpuWarpSize * 16;
    }
    else if (gpuWarpSize * 512 > minimumThreadsNeeded && gpuWarpSize * 16 < gpuMaxThreadsPerBlock)
    {
        // this is getting ridiculous...
        (*blocks) = 32;
        (*threads) = gpuWarpSize * 16;
    }
    else if (gpuWarpSize * 1024 > minimumThreadsNeeded && gpuWarpSize * 16 < gpuMaxThreadsPerBlock)
    {
        // I hope this block of code never has to be executed... that poor GPU...
        (*blocks) = 64;
        (*threads) = gpuWarpSize * 16;
    }
    else
    {
        // heuristic time! now we brute force numbers (not even close to optimal, buy hey, we've got work to do)
        (*blocks) = minimumThreadsNeeded / (gpuWarpSize * 8);
        (*threads) = gpuMaxThreadsPerBlock / 2;
        while (((*blocks) * (*threads)) < minimumThreadsNeeded)
        {
            (*blocks) = (*blocks) + 1; // add another block
            if (((*blocks) * ((*threads) / 2)) > minimumThreadsNeeded)
            {
                // if adding another block allows us to cut our threads in half then do it.
                (*threads) = (*threads) / 2;
            }
        }
    }
}
