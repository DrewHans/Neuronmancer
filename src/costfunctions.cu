/*******************************************************************************************
 * Filename: feedforwardfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device cost functions.
 *******************************************************************************************
 */

/*
 * costFunctionKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devExpectedOutput - a pointer to an array of double values in GPU device memory
 * @params: devNeurons - a pointer to an array of double values in GPU device memory
 * @params: devNeuronErrors - a pointer to an array of double values in GPU device memory
 * @params: neuronIndexStart - the index of the first neuron in the layer
 * @params: numberOfNeuronsInLayer - the total number of neurons in the layer
 */
__global__ void costFunctionKernel(double* devExpectedOutput, double* devNeurons, double* devNeuronErrors, int neuronIndexStart, int numberOfNeuronsInLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        int difference = devExpectedOutput[id] - devNeurons[neuronIndexStart + id];
        devNeuronErrors[neuronIndexStart + id] = difference * sigmoidDerivative(devNeurons[neuronIndexStart + id]);
    }
} //end cost function kernel

/*
 * costFunction method - compares a calculated output value to the outputExpected value and returns the error amount
 * @params: expectedValue - a pointer to a double value
 * @params: calculatedValue - a pointer to a double value
 * @returns: the difference between outputExpected and calculated values
 */
double costFunction(double* expectedValue, double* calculatedValue) {
    return expectedValue - calculatedValue;
} //end costFunction method
