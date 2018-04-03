/*******************************************************************************************
 * Filename: feedforwardfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device cost functions.
 *******************************************************************************************
 */

/*
 * costDerivativeKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devExpectedOutput - a pointer to an array of double values in GPU device memory
 * @params: devNeurons - a pointer to an array of double values in GPU device memory
 * @params: devNeuronErrors - a pointer to an array of double values in GPU device memory
 * @params: neuronIndexStart - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void costDerivativeKernel(double* devExpectedOutput, double* devNeurons, double* devNeuronErrors, int neuronIndexStart, int numberOfNeuronsInLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        devNeuronErrors[neuronIndexStart + id] = costDerivative(devExpectedOutput[id], devNeurons[neuronIndexStart + id]);
    }
} //end costDerivativeKernel

/*
 * costDerivative method
 * __host__ decoration tells NVCC this function should run on CPU, and be callable from the CPU host
 * __device__ decoration tells NVCC this function should run on GPU, and be callable from the GPU device
 * @params: expectedValue - a pointer to a double value
 * @params: calculatedValue - a pointer to a double value
 * @returns: the difference between outputExpected and calculated values
 */
__host__ __device__ double costDerivative(double expectedValue, double calculatedValue) {
    return expectedValue - calculatedValue;
} //end costDerivative method
