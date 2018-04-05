/*******************************************************************************************
 * Filename: updatebiasesfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device functions for updating biases.
 *******************************************************************************************
 */

/*
 * biasUpdateKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values on device (the neuron values)
 * @params: devBiases - a pointer to an array of double values on device (the bias values)
 * @params: devNeuronErrors - a pointer to an array of double values on device (the deltas for each neuron)
 * @params: numberOfNeuronsTotal - the number of total neurons in the network
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
__global__ void biasUpdateKernel(double* devNeurons, double* devBiases, double* devNeuronErrors, int numberOfNeuronsTotal, double learningRate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsTotal) {
        devBiases[id] = devBiases[id] + (learningRate * devNeuronErrors[id] * devNeurons[id]);
    }
} //end bias update kernel function

/*
 * updateBiases
 * @params: neurons - a pointer to an array of double values (the neuron values)
 * @params: biases - a pointer to an array of double values (the bias values)
 * @params: neuronErrors - a pointer to an array of double values (the deltas for each neuron)
 * @params: numberOfNeuronsTotal - the number of total neurons in the network
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void updateBiases(double* neurons, double* biases, double* neuronErrors, int numberOfNeuronsTotal, double learningRate) {
    for (int i = 0; i < numberOfNeuronsTotal; i++) {
        biases[i] = biases[i] + (learningRate * neuronErrors[i] * neurons[i]);
    }
} //end update biases function
