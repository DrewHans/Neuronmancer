/********************************************************************************
 * Filename: functions_cuda.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: functions_cuda.cu's header file - contains function prototypes
 *              and definitions for global __device__ variables.
 ********************************************************************************
 */

#ifndef FUNCTIONS_CUDA_H
#define FUNCTIONS_CUDA_H

/////////////////////////////////////////////////////////////////////////////////
// define device copies of training images and labels in device memory //////////

// Consider adding __constant__ after __device__ to store array in constant memory on the GPU device.
// If you get "ptxas error : File uses too much global constant data" then your device doesn't
// have enough constant memory to store the arrays

__device__ MNIST_Image dev_trainImages[MNIST_TRAINING_SET_SIZE];
__device__ MNIST_Label dev_trainLabels[MNIST_TRAINING_SET_SIZE];

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for functions_cuda.cu /////////////////////////////

void cuda_loadMNISTTrainingSetToDevice();
__device__ float cuda_sigmoid(const float x);
__device__ float cuda_sigmoidPrime(const float x);
void cuda_trainNetwork(InputLayer* dev_il, HiddenLayer* dev_hl, OutputLayer* dev_ol, ExpectedOutput* dev_expected, int sample, 
                        unsigned int iBlocks, unsigned int iThreads, 
                        unsigned int hBlocks, unsigned int hThreads, 
                        unsigned int oBlocks, unsigned int oThreads);
void cuda_train(InputLayer* il, HiddenLayer* hl, OutputLayer* ol);
__global__ void cudakernel_calculateHiddenLayerDeltas(HiddenLayer* __restrict__ dev_hl, OutputLayer* __restrict__ dev_ol);
__global__ void cudakernel_calculateOutputLayerDeltas(OutputLayer* __restrict__ dev_ol, ExpectedOutput* __restrict__ dev_expected);
__global__ void cudakernel_feedInputLayer(InputLayer* __restrict__ dev_il, int sample);
__global__ void cudakernel_feedHiddenLayer(HiddenLayer* __restrict__ dev_hl, InputLayer* __restrict__ dev_il);
__global__ void cudakernel_feedOutputLayer(OutputLayer* __restrict__ dev_ol, HiddenLayer* __restrict__ dev_hl);
__global__ void cudakernel_getExpectedOutput(ExpectedOutput* __restrict__ dev_expected, int sample);
__global__ void cudakernel_updateHiddenLayerWeightsAndBiases(HiddenLayer* __restrict__ dev_hl, InputLayer* __restrict__ dev_il);
__global__ void cudakernel_updateOutputLayerWeightsAndBiases(OutputLayer* __restrict__ dev_ol, HiddenLayer* __restrict__ dev_hl);
void getDeviceProperties(unsigned int* multiProcessorCount, unsigned int* warpSize, unsigned int* maxThreadsPerBlock);
void getOptimalBlocksAndThreads(unsigned int* blocks, unsigned int* threads, const unsigned int minimumThreadsNeeded);

/////////////////////////////////////////////////////////////////////////////////

#endif /* FUNCTIONS_CUDA_H */
