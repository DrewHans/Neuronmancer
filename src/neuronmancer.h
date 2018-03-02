/*******************************************************************************************
 * Filename: neuronmancer.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: Header file for main.c
 *******************************************************************************************
 */

#ifndef NEURONMANCER_H
    #define NEURONMANCER_H

    // define function prototypes for activationfunctions.cu
    __host__ __device__ double sigmoidFunction(double d);

    __host__ __device__ double sigmoidDerivative(double d);

    __global__ void sigmoidKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);

    __host__ __device__ double tanhFunction(double d);

    __host__ __device__ double tanhDerivative(double d);

    __global__ void tanhKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);

    __host__ __device__ double reluFunction(double d);

    __host__ __device__ double reluDerivative(double d);

    __global__ void reluKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer);
    
    // define function prototypes for backpropagationfunctions.cu
    __global__ void backpropagateErrorsKernel();

    __global__ void weightUpdateKernel();

    void backpropagateWithDevice();

    void backpropagateWithHost(double* expectedOutput, double* neurons, double* weights, double* neuronErrors, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, double learningRate);
    
    // define function prototypes for combinationfunctions.cu
    void combinationFunction(double* neurons, double* weights, int neuronIndex, int prevLayerIndexStart, int weightIndexStart, int prevLayerSize);

    __global__ void combinationFunctionKernel(double* devNeurons, double* devWeights, int neuronIndexStart, int prevLayerNeuronIndexStart, int weightIndexStart, int numberOfNeuronsInLayer, int numberOfNeuronsInPrevLayer);
    
    // define function prototypes for costfunctions.cu
    double costFunction(double* expectedValue, double* calculatedValue);

    __global__ void costFunctionKernel(double* devExpectedOutput, double* devNeurons, double* devNeuronErrors, int neuronIndexStart, int numberOfNeuronsInLayer);
    
    // define function prototypes for feedforwardfunctions.cu
    void feedforwardWithDevice(double* devNeurons, double* devWeights, int numberOfLayers, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);
    
    void feedforwardWithHost(double* neurons, double* weights, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer);
    
    // define function prototypes for helperfunctions.cu
    void printarray(const char* name, double* array, int n);
    
    // define function prototypes for initnetwork.cu
    void initNeurons(double* neurons, int n);
    void initWeights(double* weights, int n);

    // define function prototypes for loadinput.cu
    void loadInput(double* neurons, int n);

#endif
