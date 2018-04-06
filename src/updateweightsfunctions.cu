/*******************************************************************************************
 * Filename: updateweightsfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device functions for updating weights.
 *******************************************************************************************
 */

/*
 * weightUpdateKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values on device (the neuron values)
 * @params: devWeights - a pointer to an array of double values on device (the weight values)
 * @params: devNeuronErrors - a pointer to an array of double values on device (the deltas for each neuron)
 * @params: numberOfNeuronsInLeftLayer - the number of neurons in the layer left of the weights
 * @params: numberOfNeuronsInRightLayer - the number of neurons in the layer right of the weights
 * @params: numberOfWeightsBetweenLayers - the number of weights between the layers
 * @params: indexOfFirstNeuronInLeft - the index of left layer's first neuron
 * @params: indexOfFirstWeight - the index of the first weight between layers
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
__global__ void weightUpdateKernel(double* devNeurons, double* devWeights, double* devNeuronErrors, int numberOfNeuronsInLeftLayer,
        int numberOfNeuronsInRightLayer, int numberOfWeightsBetweenLayers, int indexOfFirstNeuronInLeft, int indexOfFirstWeight, double learningRate) {
    if (blockIdx.x < numberOfNeuronsInLeftLayer && threadIdx.x < numberOfWeightsBetweenLayers) {
        int weightIndex = indexOfFirstWeight + numberOfNeuronsInRightLayer * blockIdx.x + threadIdx.x;
        int neuronIndex = indexOfFirstNeuronInLeft + blockIdx.x;
        devWeights[weightIndex] = devWeights[weightIndex] - (learningRate * devNeuronErrors[neuronIndex] * devNeurons[neuronIndex]);
    }
} //end weight update kernel function

/*
 * updateWeights
 * @params: neurons - a pointer to an array of double values (the neuron values)
 * @params: weights - a pointer to an array of double values (the weight values)
 * @params: neuronErrors - a pointer to an array of double values (the deltas for each neuron)
 * @params: numberOfLayers - the total number of layers in our artificial neural network
 * @params: neuronsPerLayer - a pointer to an array of int values (the number of neurons in each layer)
 * @params: firstNeuronIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first neuron)
 * @params: firstWeightIndexPerLayer - a pointer to an array of int values (the indexes of each layer's first weight)
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 */
void updateWeights(double* neurons, double** weights, double* neuronErrors, int numberOfLayers, int* neuronsPerLayer, int* firstNeuronIndexPerLayer,
        int* firstWeightIndexPerLayer, double learningRate) {
    // for each layer l after input layer, update the weights in the layer
    for (int l = 1; l < numberOfLayers; l++) {
        // for each neuron in layer l
        for (int n = 0; n < neuronsPerLayer[l]; n++) {
            for (int w = 0; w < neuronsPerLayer[l - 1]; w++) {
                int weightIndex = firstWeightIndexPerLayer[l] + neuronsPerLayer[l - 1] * n + w;
                int neuronIndex = firstNeuronIndexPerLayer[l] + n;
                (*weights)[weightIndex] = (*weights)[weightIndex] - (learningRate * neuronErrors[neuronIndex] * neurons[neuronIndex]);
            }
        }
    }
} //end update weights function
