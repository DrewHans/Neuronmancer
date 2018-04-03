/*******************************************************************************************
 * Filename: backpropagationfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device backpropagation functions.
 *******************************************************************************************
 */

/*
 * backpropagateErrorsKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 */
__global__ void backpropagateErrorsKernel(double* devNeurons, double* devWeights, double* devBiases, double* devNeuronErrors, int numberOfNeuronsInLeftLayer,
        int numberOfWeightsBetweenLayers, int indexOfFirstNeuronInLeft, int indexOfFirstNeuronInRight, int indexOfFirstWeight) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLeftLayer) {
        int neuronId = indexOfFirstNeuronInLeft + id;
        double neuronError = 0.0;
        for (int w = 0; w < numberOfWeightsBetweenLayers; w += numberOfNeuronsInLeftLayer) {
            neuronError = neuronError + (devWeights[indexOfFirstWeight + w] * devNeuronErrors[indexOfFirstNeuronInRight + w]);
        }
        devNeuronErrors[neuronId] = neuronError * sigmoidDerivative(devNeurons[neuronId]);
    }
} //end backpropagate errors kernel

/*
 * backpropagateWithDevice method
 */
void backpropagateWithDevice(int numBlocks, int threadsPerBlock, double* devExpectedOutput, double* devNeurons, double* devWeights, double* devBiases,
        double* devNeuronErrors, int numberOfLayers, int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    // for each node in the output layer, calculate the output error (spawn 1 thread for each neuron in the output layer)
    int outputLayerIndex = numberOfLayers - 1;
    costDerivativeKernel<<<numBlocks, threadsPerBlock>>>(devExpectedOutput, devNeurons, devNeuronErrors, firstNeuronIndexPerLayer[outputLayerIndex],
            neuronsPerLayer[outputLayerIndex]);
    cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel

    // for each layer l between output and input, visit in reverse order, backpropagate error values and update weights
    for (int l = outputLayerIndex - 1; l > 0; l--) {
        // for each node in layer l, backpropagate the error from layer l+1 (spawn 1 thread for each neuron in layer l)
        backpropagateErrorsKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devWeights, devBiases, devNeuronErrors, neuronsPerLayer[l],
                weightsPerLayer[l + 1], firstNeuronIndexPerLayer[l], firstNeuronIndexPerLayer[l + 1], firstWeightIndexPerLayer[l + 1]);
        cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel
    }
} //end backpropagateWithDevice method

/*
 * backpropagateWithHost method
 */
void backpropagateWithHost(double* expectedOutput, double* neurons, double* weights, double* biases, double* neuronErrors, int numberOfLayers,
        int* neuronsPerLayer, int* weightsPerLayer, int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    // for each node in the output layer, calculate the output error
    int outputLayerIndex = numberOfLayers - 1;
    int neuronId = 0;
    double neuronError = 0.0;
    for (int i = 0; i < neuronsPerLayer[outputLayerIndex]; i++) {
        neuronError = 0.0; // zero out the neuron error for next neuron
        neuronId = firstNeuronIndexPerLayer[outputLayerIndex] + i; //
        neuronError = costDerivative(expectedOutput[i], neurons[neuronId]); // store the cost/error/loss of output layer neuron
        neuronErrors[neuronId] = neuronError * sigmoidDerivative(neurons[neuronId]); // calculate the output layer delta and shove in neuronErrors
    }

    // for each layer l between output and input, visit in reverse order and backpropagate error values
    for (int l = outputLayerIndex - 1; l > 0; l--) {
        // for each neuron in layer l
        for (int n = 0; n < neuronsPerLayer[l]; n++) {
            neuronError = 0.0; // zero out the neuron error for next neuron
            // for each connection between layer l and l+1
            for (int w = 0; w < weightsPerLayer[l + 1]; w = w + neuronsPerLayer[l]) {
                // layer l neuron n delta
                neuronError = neuronError + (weights[firstWeightIndexPerLayer[l + 1] + w] * neuronErrors[firstNeuronIndexPerLayer[l + 1] + w]);
            }
            neuronErrors[firstNeuronIndexPerLayer[l] + n] = neuronError * sigmoidDerivative(neurons[firstNeuronIndexPerLayer[l] + n]);
        }
    }
} //end backpropagateWithHost method
