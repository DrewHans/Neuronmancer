/*******************************************************************************************
 * Filename: activationfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device activation functions.
 *******************************************************************************************
 */

#include <math.h>

/*
 * sigmoidFunction
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a double value
 * @returns: a double value in range [0, 1]
 */
__host__ __device__ double sigmoidFunction(double d) {
    return 1.0 / (1.0 + exp(-1.0 * d));
}//end sigmoid activation function

/*
 * sigmoidDerivative
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a double value
 * @returns: a double value
 */
__host__ __device__ double sigmoidDerivative(double d) {
    return sigmoidFunction(d) * (1.0 - sigmoidFunction(d));
}//end sigmoid derivative function

/*
 * sigmoidKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values in GPU device memory
 * @params: neuronIndexStart - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void sigmoidKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        devNeurons[neuronIndexStart + id] = sigmoidFunction(devNeurons[neuronIndexStart + id]);
    }
}//end sigmoid activation kernel

/*
 * tanhFunction
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a double value
 * @returns: a double value
 */
__host__ __device__ double tanhFunction(double d) {
    return (2.0 / (1.0 + exp(-2.0 * d))) - 1.0;
}//end tanh activation function

/*
 * tanhDerivative
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a double value
 * @returns: a double value
 */
__host__ __device__ double tanhDerivative(double d) {
    return 1.0 - pow(tanhFunction(d), 2.0);
}//end tanh derivative function

/*
 * tanhKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values in GPU device memory
 * @params: neuronIndexStart - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void tanhKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        devNeurons[neuronIndexStart + id] = tanhFunction(devNeurons[neuronIndexStart + id]);
    }
}//end tanh activation kernel

/*
 * reluFunction
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a double value
 * @returns: a double value
 */
__host__ __device__ double reluFunction(double d) {
    if (d < 0) {
        return 0;
    } else {
        return d;
    }
}//end relu activation function

/*
 * reluDerivative
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: d - a double value
 * @returns: a double value
 */
__host__ __device__ double reluDerivative(double d) {
    if (d < 0) {
        return 0;
    } else {
        return 1;
    }
}//end relu derivative function

/*
 * reluKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values in GPU device memory
 * @params: neuronIndexStart - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void reluKernel(double* devNeurons, int neuronIndexStart, int numberOfNeuronsInLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        devNeurons[neuronIndexStart + id] = reluFunction(devNeurons[neuronIndexStart + id]);
    }
}//end relu activation kernel

