/*******************************************************************************************
 * Filename: feedforwardfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device feedforward functions.
 *******************************************************************************************
 */

/*
 * feedforwardWithDevice - propagates the inputs forward to compute the outputs
 */
void feedforwardWithDevice(int numBlocks, int threadsPerBlock, double* devNeurons, double* devWeights, double* devBiases, int numberOfLayers,
        int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    // go layer to layer and for each neuron in the current layer: spawn a thread, perform combination, sync threads, spawn a thread
    for (int l = 1; l < numberOfLayers; l++) {
        combinationFunctionKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devWeights, devBiases, firstNeuronIndexPerLayer[l],
                firstNeuronIndexPerLayer[l - 1], firstWeightIndexPerLayer[l], numberOfNeuronsPerLayer[l], numberOfNeuronsPerLayer[l - 1]);
        cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel
        sigmoidKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, firstNeuronIndexPerLayer[l], numberOfNeuronsPerLayer[l]);
        cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel
    }
} //end feedforwardWithDevice method

/*
 * feedforwardWithHost - propagates the inputs forward to compute the outputs
 */
void feedforwardWithHost(double* neurons, double* weights, double* biases, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer,
        int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    // go layer to layer
    for (int i = 1; i < numberOfLayers; i++) {
        // go neuron to neuron in layer i
        for (int j = 0; j < neuronsPerLayer[i]; j++) {
            combinationFunction(neurons, weights, biases, firstNeuronIndexPerLayer[i] + j, firstNeuronIndexPerLayer[i - 1], firstWeightIndexPerLayer[i] + j,
                    neuronsPerLayer[i - 1]);
            sigmoidFunction(neurons[firstNeuronIndexPerLayer[i] + j]);
            printf("neurons[%d]=%f\n", (firstNeuronIndexPerLayer[i] + j), neurons[firstNeuronIndexPerLayer[i] + j]);
        }
    }
} //end feedforwardWithHost method
