/********************************************************************************
 * Filename: main.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: Header file for main.cu
 ********************************************************************************
 */

#ifndef MAIN_H
#define MAIN_H

// include the standard C headers ///////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

// include the CUDA headers /////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// define macros for input buffers, debugging mode, etc. ////////////////////////
#define MAXINPUT 32
#define DEBUG

// define macros for model_read.cu and model_save.cu ////////////////////////////
#define VALUEDELIM ","
#define MODELDIRECTORY "../nmModel"
#define MODELVALUESLOCATION "../nmModel/modelvalues.csv"
#define WEIGHTSFILELOCATION "../nmModel/weights.csv"
#define BIASESFILELOCATION "../nmModel/biases.csv"
#define EPOCHSFILELOCATION "../nmModel/epochs.txt"
#define LEARNINGRATEFILELOCATION "../nmModel/learningrate.txt"

// define macros for working with the MNIST data set ////////////////////////////
#define MNISTCLASSIFICATIONS 10
#define MNISTSAMPLEDATASIZE 784
#define MNISTTESTSETSIZE 10000
#define MNISTTRAININGSETSIZE 60000
#define MNISTTESTFILELOCATION "../mnist/mnist_test.csv"
#define MNISTTRAINFILELOCATION "../mnist/mnist_train.csv"

// define struct for using the stat command /////////////////////////////////////
struct stat st = { 0 };

// define function prototypes for activations.cu ////////////////////////////////
__host__ __device__ float sigmoidFunction(float d);
__host__ __device__ float sigmoidDerivative(float d);
__host__ __device__ float tanhFunction(float d);
__host__ __device__ float tanhDerivative(float d);
__host__ __device__ float reluFunction(float d);
__host__ __device__ float reluDerivative(float d);
__global__ void cudaKernel_ActivateLayerUsingSigmoid(float* devNeurons, unsigned int indexOfFirstNeuronInLayer, unsigned int numberOfNeuronsInLayer);
__global__ void cudaKernel_ActivateLayerUsingTanh(float* devNeurons, unsigned int indexOfFirstNeuronInLayer, unsigned int numberOfNeuronsInLayer);
__global__ void cudaKernel_ActivateLayerUsingRelu(float* devNeurons, unsigned int indexOfFirstNeuronInLayer, unsigned int numberOfNeuronsInLayer);

// define function prototypes for backpropagation.cu ////////////////////////////
void backpropagationUsingHost(float** neuronDeltas, float* expected, float* neurons, float* weights, float* biases, 
                              unsigned int numberOfLayers, unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer,
                              unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer);
void backpropagationUsingDevice(float* devNeuronDeltas, float* devExpected, float* devNeurons, float* devWeights, float* devBiases, 
                                unsigned int numberOfLayers, unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer,
                                unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer);
__global__ static void cudaKernel_CalculateLeftLayerDeltas(float* devNeuronDeltas, float* devExpected, float* devNeurons, float* devWeights, 
                                                           unsigned int numberOfNeuronsInLeft, unsigned int numberOfNeuronsInRight,
                                                           unsigned int indexOfFirstLeftNeuron, unsigned int indexOfFirstRightNeuron, 
                                                           unsigned int indexOfFirstWeight);
__global__ static void cudaKernel_CalculateOutputLayerDeltas(float* devNeuronDeltas, float* devExpected, float* devNeurons, 
                                                             unsigned int numberOfNeuronsInOutput, unsigned int indexOfFirstOutputNeuron);
__global__ static void cudaKernel_updateBiases(float* devNeuronDeltas, float* devNeurons, float* devBiases, 
                                               unsigned int numberOfNeuronsTotal, float learningRate);
__global__ static void cudaKernel_updateWeightsBetweenLayers(float* devNeuronDeltas, float* devNeurons, float* devWeights, 
                                                             unsigned int numberOfNeuronsInLeft, unsigned int numberOfNeuronsInRight, 
                                                             unsigned int numberOfWeightsBetweenLayers, unsigned int indexOfFirstLeftNeuron, 
                                                             unsigned int indexOfFirstWeight, float learningRate);
__host__ __device__ float quadraticCostDerivative(float expectedValue, float calculatedValue);
void updateBiasesUsingDevice(float* devNeuronDeltas, float* devNeurons, float* devBiases, unsigned int numberOfNeuronsTotal, float learningRate);
void updateBiasesUsingHost(float* neuronDeltas, float* neurons, float** biases, unsigned int numberOfNeuronsTotal, float learningRate);
void updateWeightsUsingDevice(float* devNeuronDeltas, float* devNeurons, float* devWeights, unsigned int numberOfLayers, 
                              unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer, 
                              unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer, float learningRate);
void updateWeightsUsingHost(float* neuronDeltas, float* neurons, float** weights, 
                            unsigned int numberOfLayers, unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer, 
                            unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer, float learningRate);

// define function prototypes for feedforward.cu ////////////////////////////////
void feedforwardUsingHost(float** neurons, float* weights, float* biases, 
                          unsigned int numberOfLayers, unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer,
                          unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer);
void feedforwardUsingDevice(float* devNeurons, float* devWeights, float* devBiases, 
                            unsigned int numberOfLayers, unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer,
                            unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer);
__global__ void static cudaKernel_CalculateWeightedSumPlusBias(float* devNeurons, float* devWeights, float* devBiases, 
                                                               unsigned int numberOfNeuronsInLeft, unsigned int numberOfNeuronsInRight, 
                                                               unsigned int indexOfFirstLeftNeuron, unsigned int indexOfFirstRightNeuron, 
                                                               unsigned int indexOfFirstWeight);

// define function prototypes for functions_misc.cu /////////////////////////////
void getDeviceProperties(unsigned int* multiProcessorCount, unsigned int* warpSize);
int getOptimalThreadSize(unsigned int blocks, unsigned int threads, unsigned int minimumThreadsNeeded, unsigned int gpuWarpsize);
void initArrayToRandomFloats(float** a, unsigned int n);
void initArrayToZeros(float** a, unsigned int n);
void printarray_float(const char* name, float* a, unsigned int n);
void printarray_int(const char* name, unsigned int* a, unsigned int n);
void printFarewellMSG();
void onCudaKernelLaunchFailure(char* kernel, cudaError_t cudaStatus);
void onCudaDeviceSynchronizeError(char* kernel, cudaError_t cudaStatus);
void onCudaMallocError(unsigned int size);
void onCudaMemcpyError(const char* hostVariable);
void onFailToSetGPUDevice();
void onFileOpenError(const char* path);
void onFileReadError(const char* path);
void onInvalidInput(int myPatience);
void onMallocError(unsigned int size);

// define function prototypes for functions_mnist.cu ////////////////////////////
int getCalculatedMNISTClassification(float* neurons, unsigned int indexOfFirstOutputNeuron);
void loadMnistTestSamples(unsigned char** testData, char** testLabels, unsigned int* numberOfSamples);
void loadMnistTrainingSamples(unsigned char** trainingData, char** trainingLabels, unsigned int* numberOfSamples);
void loadNextMnistSampleUsingHost(const char* mnistLabels, const unsigned char* mnistData, 
                                  unsigned int indexOfNextSampleLabel, unsigned int indexOfNextSampleFirstData, 
                                  float** expected, float** neurons);
void loadNextMnistSampleUsingDevice(const char* devMnistLabels, const unsigned char* devMnistData, 
                                  unsigned int indexOfNextSampleLabel, unsigned int indexOfNextSampleFirstData, 
                                  float* devExpected, float* devNeurons);
__global__ void cudaKernel_loadNextMnistSampleLabelIntoExpected(const char* devMnistLabels, int indexOfNextSampleLabel, float* devExpected);
__global__ void cudaKernel_loadNextMnistSampleDataIntoInputLayer(const unsigned char* devMnistData, int indexOfNextSampleFirstData, float* devNeurons);

// define function prototypes for model_read.cu /////////////////////////////////
float** readBiasesFromDisk(unsigned int numberOfBiasesTotal);
int readEpochsFromDisk();
float readLearningRateFromDisk();
float** readWeightsFromDisk(unsigned int numberOfWeightsTotal);
void readModelValuesFromDisk(unsigned int* p_numberOfLayers, unsigned int* p_numberOfNeuronsTotal, unsigned int* p_numberOfWeightsTotal, 
                             unsigned int** p_numberOfNeuronsInLayer, unsigned int** p_numberOfWeightsInFrontOfLayer, 
                             unsigned int** p_indexOfFirstNeuronInLayer, unsigned int** p_indexOfFirstWeightInFrontOfLayer);

// define function prototypes for model_save.cu /////////////////////////////////
void saveBiasesToDisk(float* biases, unsigned int numberOfBiasesTotal);
void saveEpochsToDisk(unsigned int epochs);
void saveLearningRateToDisk(float learningRate);
void saveWeightsToDisk(float* weights, unsigned int numberOfWeightsTotal);
void saveModelValuesToDisk(unsigned int numberOfLayers, unsigned int numberOfNeuronsTotal, unsigned int numberOfWeightsTotal, 
                           unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer,
                           unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer);
void saveModel(unsigned int numberOfLayers, unsigned int numberOfNeuronsTotal, unsigned int numberOfWeightsTotal, 
               unsigned int* numberOfNeuronsInLayer, unsigned int* numberOfWeightsInFrontOfLayer,
               unsigned int* indexOfFirstNeuronInLayer, unsigned int* indexOfFirstWeightInFrontOfLayer, 
               float* weights, float* biases, float learningRate, unsigned int epochs);

// define function prototypes for ui_create.cu //////////////////////////////////
void ui_create();

// define function prototypes for ui_evaluate.cu ////////////////////////////////
void ui_evaluate();

// define function prototypes for ui_train.cu ///////////////////////////////////
void ui_train();

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

