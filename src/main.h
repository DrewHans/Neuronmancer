/********************************************************************************
 * Filename: main.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: Header file for main.cu
 ********************************************************************************
 */

#ifndef MAIN_H
#define MAIN_H

/////////////////////////////////////////////////////////////////////////////////
// include the standard C headers ///////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

/////////////////////////////////////////////////////////////////////////////////
// include the CUDA headers /////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/////////////////////////////////////////////////////////////////////////////////
// define macros for input buffers, debugging mode, etc. ////////////////////////
#define MAXINPUT 32
#define DEBUG

/////////////////////////////////////////////////////////////////////////////////
// define macros for model_read.cu and model_save.cu ////////////////////////////
#define VALUEDELIM ","
#define MODELDIRECTORY "../nmModel"
#define MODELVALUESLOCATION "../nmModel/modelvalues.csv"
#define WEIGHTSFILELOCATION "../nmModel/weights.csv"
#define BIASESFILELOCATION "../nmModel/biases.csv"
#define EPOCHSFILELOCATION "../nmModel/epochs.txt"
#define LEARNINGRATEFILELOCATION "../nmModel/learningrate.txt"

/////////////////////////////////////////////////////////////////////////////////
// define macros for working with the MNIST data set ////////////////////////////
#define MNISTCLASSIFICATIONS 10
#define MNISTSAMPLEDATASIZE 784
#define MNISTTESTSETSIZE 10000
#define MNISTTRAININGSETSIZE 60000
#define MNISTTESTFILELOCATION "../mnist/mnist_test.csv"
#define MNISTTRAINFILELOCATION "../mnist/mnist_train.csv"

/////////////////////////////////////////////////////////////////////////////////
// define struct for using the stat command /////////////////////////////////////
struct stat st = { 0 };

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for activations.cu ////////////////////////////////
__host__ __device__ float sigmoidFunction(const float d);
__host__ __device__ float sigmoidDerivative(const float d);
__host__ __device__ float tanhFunction(const float d);
__host__ __device__ float tanhDerivative(const float d);
__host__ __device__ float reluFunction(const float d);
__host__ __device__ float reluDerivative(const float d);
__global__ void cudaKernel_ActivateLayerUsingSigmoid(float* devNeurons, const unsigned int indexOfFirstNeuronInLayer, const unsigned int numberOfNeuronsInLayer);
__global__ void cudaKernel_ActivateLayerUsingTanh(float* devNeurons, const unsigned int indexOfFirstNeuronInLayer, const unsigned int numberOfNeuronsInLayer);
__global__ void cudaKernel_ActivateLayerUsingRelu(float* devNeurons, const unsigned int indexOfFirstNeuronInLayer, const unsigned int numberOfNeuronsInLayer);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for backpropagation.cu ////////////////////////////
void backpropagationUsingHost(float** neuronDeltas, const float* expected, const float* neurons, const float* weights, const float* biases, 
                              const unsigned int numberOfLayers, const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                              const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer);

void backpropagationUsingDevice(float* devNeuronDeltas, float* devExpected, float* devNeurons, float* devWeights, float* devBiases, 
                                const unsigned int numberOfLayers, const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                                const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer);
__global__ static void cudaKernel_CalculateLeftLayerDeltas(float* devNeuronDeltas, __restrict__ const float* devExpected, 
                                                           __restrict__ const float* devNeurons, __restrict__ const float* devWeights, 
                                                           const unsigned int numberOfNeuronsInLeft, const unsigned int numberOfNeuronsInRight,
                                                           const unsigned int indexOfFirstLeftNeuron, const unsigned int indexOfFirstRightNeuron, 
                                                           const unsigned int indexOfFirstWeight);
__global__ static void cudaKernel_CalculateOutputLayerDeltas(float* devNeuronDeltas, __restrict__ const float* devExpected, 
                                                             __restrict__ const float* devNeurons, 
                                                             const unsigned int numberOfNeuronsInOutput, 
                                                             const unsigned int indexOfFirstOutputNeuron);
__global__ static void cudaKernel_updateBiases(__restrict__ const float* devNeuronDeltas, float* devBiases, 
                                               const unsigned int numberOfNeuronsTotal, const float learningRate);
__global__ static void cudaKernel_updateWeightsBetweenLayers(__restrict__ const float* devNeuronDeltas, __restrict__ const float* devNeurons, float* devWeights, 
                                                             const unsigned int numberOfNeuronsInLeft, const unsigned int numberOfNeuronsInRight, 
                                                             const unsigned int numberOfWeightsBetweenLayers, const unsigned int indexOfFirstLeftNeuron, 
                                                             const unsigned int indexOfFirstWeight, const float learningRate);
__host__ __device__ float quadraticCostDerivative(const float expectedValue, const float calculatedValue);
void updateBiasesUsingDevice(float* devNeuronDeltas, float* devBiases, const unsigned int numberOfNeuronsTotal, const float learningRate);
void updateBiasesUsingHost(const float* neuronDeltas, float** biases, const unsigned int numberOfNeuronsTotal, const float learningRate);
void updateWeightsUsingDevice(float* devNeuronDeltas, float* devNeurons, float* devWeights, const unsigned int numberOfLayers, 
                              const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer, 
                              const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer, 
                              const float learningRate);
void updateWeightsUsingHost(const float* neuronDeltas, const float* neurons, float** weights, const unsigned int numberOfLayers, 
                            const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer, 
                            const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer, 
                            const float learningRate);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for feedforward.cu ////////////////////////////////
void feedforwardUsingHost(float** neurons, const float* weights, const float* biases, 
                          const unsigned int numberOfLayers, const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                          const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer);
void feedforwardUsingDevice(float* devNeurons, float* devWeights, float* devBiases, 
                            const unsigned int numberOfLayers, const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                            const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer);
__global__ void static cudaKernel_CalculateWeightedSumPlusBias(float* devNeurons, __restrict__ const float* devWeights, __restrict__ const float* devBiases, 
                                                               const unsigned int numberOfNeuronsInLeft, const unsigned int numberOfNeuronsInRight, 
                                                               const unsigned int indexOfFirstLeftNeuron, const unsigned int indexOfFirstRightNeuron, 
                                                               const unsigned int indexOfFirstWeight);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for functions_misc.cu /////////////////////////////
void getDeviceProperties(unsigned int* multiProcessorCount, unsigned int* warpSize);
int getOptimalThreadSize(const unsigned int blocks, unsigned int threads, const unsigned int minimumThreadsNeeded, const unsigned int gpuWarpsize);
void initArrayToRandomFloats(float** a, const unsigned int n);
void initDeviceArrayToZeros(float* devA, const unsigned int n);
__global__ static void cudaKernel_initArrayToZeros(float* devA, const unsigned int n);
void printarray_float(const char* name, const float* a, const unsigned int n);
void printarray_int(const char* name, const unsigned int* a, const unsigned int n);
void printConfusionMatrix(const int* cm, const unsigned int n);
void printFarewellMSG();
void onCudaKernelLaunchFailure(const char* kernel, const cudaError_t cudaStatus);
void onCudaDeviceSynchronizeError(const char* kernel, const cudaError_t cudaStatus);
void onCudaMallocError(const unsigned int size);
void onCudaMemcpyError(const char* hostVariable);
void onFailToSetGPUDevice();
void onFileOpenError(const char* path);
void onFileReadError(const char* path);
void onInvalidInput(const int myPatience);
void onMallocError(const unsigned int size);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for functions_mnist.cu ////////////////////////////
int getCalculatedMNISTClassificationUsingHost(const float* neurons, const unsigned int indexOfFirstOutputNeuron);
void getCalculatedMNISTClassificationUsingDevice(int* devClassification, const float* devNeurons, const unsigned int indexOfFirstOutputNeuron);
void loadMnistSampleUsingHost(const char* mnistLabels, const unsigned char* mnistData, 
                              const unsigned int indexOfSampleLabel, const unsigned int indexOfSampleFirstData, 
                              float** expected, float** neurons);
void loadMnistSampleUsingDevice(const char* devMnistLabels, const unsigned char* devMnistData, 
                                  const unsigned int indexOfSampleLabel, const unsigned int indexOfSampleFirstData, 
                                  float* devExpected, float* devNeurons);
void readMnistTestSamplesFromDisk(unsigned char** testData, char** testLabels, unsigned int* numberOfSamples);
void readMnistTrainingSamplesFromDisk(unsigned char** trainingData, char** trainingLabels, unsigned int* numberOfSamples);
__global__ void cudaKernel_GetCalculatedMNISTClassification(int* devClassification, 
                                                            __restrict__ const float* devNeurons, 
                                                            const unsigned int indexOfFirstOutputNeuron);
__global__ void cudaKernel_loadMnistSampleLabelIntoExpected(__restrict__ const char* devMnistLabels, 
                                                            const unsigned int indexOfNextSampleLabel, 
                                                            float* devExpected);
__global__ void cudaKernel_loadMnistSampleDataIntoInputLayer(__restrict__ const unsigned char* devMnistData, 
                                                             const unsigned int indexOfNextSampleFirstData, 
                                                             float* devNeurons);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for model_read.cu /////////////////////////////////
float** readBiasesFromDisk(unsigned int numberOfBiasesTotal);
unsigned int readEpochsFromDisk();
float readLearningRateFromDisk();
float** readWeightsFromDisk(unsigned int numberOfWeightsTotal);
void readModelValuesFromDisk(unsigned int* p_numberOfLayers, unsigned int* p_numberOfNeuronsTotal, unsigned int* p_numberOfWeightsTotal, 
                             unsigned int** p_numberOfNeuronsInLayer, unsigned int** p_numberOfWeightsInFrontOfLayer, 
                             unsigned int** p_indexOfFirstNeuronInLayer, unsigned int** p_indexOfFirstWeightInFrontOfLayer);
void readModel(float* p_learningRate, unsigned int* p_epochs, 
               unsigned int* p_numberOfLayers, unsigned int* p_numberOfNeuronsTotal, unsigned int* p_numberOfWeightsTotal, 
               unsigned int** p_numberOfNeuronsInLayer, unsigned int** p_numberOfWeightsInFrontOfLayer,
               unsigned int** p_indexOfFirstNeuronInLayer, unsigned int** p_indexOfFirstWeightInFrontOfLayer, 
               float** p_weights, float** p_biases);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for model_save.cu /////////////////////////////////
void saveBiasesToDisk(const float* biases, const unsigned int numberOfBiasesTotal);
void saveEpochsToDisk(const unsigned int epochs);
void saveLearningRateToDisk(const float learningRate);
void saveWeightsToDisk(const float* weights, const unsigned int numberOfWeightsTotal);
void saveModelValuesToDisk(const unsigned int numberOfLayers, const unsigned int numberOfNeuronsTotal, const unsigned int numberOfWeightsTotal, 
                           const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                           const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer);
void saveModel(const unsigned int numberOfLayers, const unsigned int numberOfNeuronsTotal, const unsigned int numberOfWeightsTotal, 
               const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
               const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer, 
               const float* weights, const float* biases, const float learningRate, const unsigned int epochs);

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for ui_create.cu //////////////////////////////////
void ui_create();

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for ui_evaluate.cu ////////////////////////////////
void ui_evaluate();

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for ui_train.cu ///////////////////////////////////
void ui_train();

/////////////////////////////////////////////////////////////////////////////////
// include the Neuronmancer src code files (after all function prototypes have been defined)
#include "./functions_misc.cu"
#include "./functions_mnist.cu"
#include "./model_read.cu"
#include "./model_save.cu"
#include "./activations.cu"
#include "./backpropagation.cu"
#include "./feedforward.cu"
#include "./ui_create.cu"
#include "./ui_evaluate.cu"
#include "./ui_train.cu"

#endif /* MAIN_H */
