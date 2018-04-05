/*******************************************************************************************
 * Filename: combinationfunctions.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains combination functions used for feeding forward input.
 *******************************************************************************************
 */

/*
 * combinationFunctionKernel
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - a pointer to an array of double values (the neuron values) in device memory
 * @params: devWeights - a pointer to an array of double values (the weight values) in device memory
 * @params: devBiases - a pointer to an array of double values (the bias values) in device memory
 * @params: neuronIndexStart - the index of the first neuron in the current layer
 * @params: prevLayerNeuronIndexStart - the index of the first neuron in the previous layer
 * @params: weightIndexStart - the index of the weight associated with the first neuron in the previous layer
 * @params: numberOfNeuronsInLayer - the number of neurons in the current layer
 * @params: numberOfNeuronsInPrevLayer - the number of neurons in the previous layer
 */
__global__ void combinationFunctionKernel(double* devNeurons, double* devWeights, double* devBiases, int neuronIndexStart, int prevLayerNeuronIndexStart,
        int weightIndexStart, int numberOfNeuronsInLayer, int numberOfNeuronsInPrevLayer) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < numberOfNeuronsInLayer) {
        for (int n = 0; n < numberOfNeuronsInPrevLayer; n++) {
            devNeurons[neuronIndexStart + id] = devNeurons[neuronIndexStart + id]
                    + (devNeurons[prevLayerNeuronIndexStart + n] * devWeights[weightIndexStart + n]) + devBiases[neuronIndexStart + id];
        }
    }
} //end combination function kernel function

/*
 * combinationFunction method - combines input from previous layer into the neuron at index neuronIndex
 * @params: neurons - pointer to an array of double values
 * @params: weights - pointer to an array of double values
 * @params: biases - pointer to an array of double values
 * @params: neuronIndex - the index of the value we are updating in the neurons array
 * @params: prevLayerIndexStart - the index of the first neuron in the previous layer
 * @params: weightIndexStart - the index of the weight associated with the first neuron in the previous layer
 * @params: prevLayerSize - the number of neurons in the previous layer
 */
void combinationFunction(double* neurons, double* weights, double* biases, int neuronIndex, int prevLayerIndexStart, int weightIndexStart, int prevLayerSize) {
    neurons[neuronIndex] = 0.0; // clear out any garbage left over from previous feedforward
    // go neuron to neuron in the previous layer
    for (int i = 0; i < prevLayerSize; i++) {
        neurons[neuronIndex] = neurons[neuronIndex] + (neurons[prevLayerIndexStart + i] * weights[weightIndexStart + i]) + biases[neuronIndex];
    }
}    //end combinationFunction function
