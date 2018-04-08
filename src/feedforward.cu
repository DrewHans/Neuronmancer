/*******************************************************************************************
 * Filename: feedforward.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the host and device feedforward functions and a cudaKernel
 *              for feeding a left-layer's neurons' output into it's right-layer's neurons.
 *******************************************************************************************
 */

/*
 * feedforwardUsingHost - propagates the input-layer's values through the network using CPU
 * @params: neurons - a double pointer-pointer to the chunk of memory containing the neuron values
 * @params: weights - a double pointer to the chunk of memory containing the weight values
 * @params: biases - a double pointer to the chunk of memory containing the biases values
 * @params: numberOfLayers - the int number of layers in the network (index 0 => input, index numberOfLayers-1 => output)
 * @params: numberOfNeuronsInLayer - an int pointer to a chunk of memory containing the number of neurons in each layer
 * @params: numberOfWeightsInFrontOfLayer - an int pointer to a chunk of memory containing the number of weights in front of each layer
 * @params: indexOfFirstNeuronInLayer - an int pointer to a chunk of memory containing the indexes of the first neurons in each layer
 * @params: indexOfFirstWeightInFrontOfLayer - an int pointer to a chunk of memory containing the indexes of the first weight in front of each layer
 */
void feedforwardUsingHost(double** neurons, double* weights, double* biases, 
                          int numberOfLayers, int* numberOfNeuronsInLayer, int* numberOfWeightsInFrontOfLayer,
                          int* indexOfFirstNeuronInLayer, int* indexOfFirstWeightInFrontOfLayer) {
    // for each layer i in network (starting at the first non-input-layer): 
    // propagate the left-layer output's to the right-layer, activate the right-layer, then repeat until output-layer is activated
    for (int l = 1; l < numberOfLayers; l++) {
        int numberOfNeuronsInLeft = numberOfNeuronsInLayer[l-1]; // left-layer size
        int numberOfNeuronsInRight = numberOfNeuronsInLayer[l]; // right-layer size

        int indexOfFirstLeftNeuron = indexOfFirstNeuronInLayer[l-1]; // start at this neuron for left-layer
        int indexOfFirstRightNeuron = indexOfFirstNeuronInLayer[l]; // start at this neuron for right-layer
        int indexOfFirstWeight = indexOfFirstWeightInFrontOfLayer[l]; // start at this weight 

        // for each neuron nr in right-layer: 
        // combine left-layer's outputs with appropriate weights, then apply activation to the neuron
        for (int nr = 0; nr < numberOfNeuronsInRight; nr++) {
            double weightedSum = 0.0; // temporarily store the weighted sum for the neuron

            // for each neuron nl in left-layer:
            // combine the neuron nl's output with the appropriate weight and store in weightedSum
            for (int nl = 0; nl < numberOfNeuronsInLeft; nl++) {
                weightedSum = weightedSum + (((*neurons)[indexOfFirstLeftNeuron + nl]) * (weights[indexOfFirstWeight + numberOfNeuronsInLeft * nr + nl]));
            }

            // store the weighted sum plus the bias in neuron nr (over-writes any previous value in neuron nr)
            (*neurons)[indexOfFirstRightNeuron + nr] = weightedSum + biases[indexOfFirstRightNeuron + nr];

            // perfrom activation on neuron nr and then store the output back in neuron nr
            (*neurons)[indexOfFirstRightNeuron + nr] = sigmoidFunction((*neurons)[indexOfFirstRightNeuron + nr]);
        }
    }
} //end feedforwardUsingHost function

/*
 * feedforwardUsingDevice - propagates the input-layer's values through the network using GPU device
 * @params: devNeurons - device copy of double* neurons
 * @params: devWeights - device copy of double* weights
 * @params: devBiases - device copy of double* biases
 * @params: numberOfLayers - the int number of layers in the network (index 0 => input, index numberOfLayers-1 => output)
 * @params: numberOfNeuronsInLayer - an int pointer to a chunk of memory containing the number of neurons in each layer
 * @params: numberOfWeightsInFrontOfLayer - an int pointer to a chunk of memory containing the number of weights in front of each layer
 * @params: indexOfFirstNeuronInLayer - an int pointer to a chunk of memory containing the indexes of the first neurons in each layer
 * @params: indexOfFirstWeightInFrontOfLayer - an int pointer to a chunk of memory containing the indexes of the first weight in front of each layer
 */
void feedforwardUsingDevice(double* devNeurons, double* devWeights, double* devBiases, 
                           int numberOfLayers, int* numberOfNeuronsInLayer, int* numberOfWeightsInFrontOfLayer,
                          int* indexOfFirstNeuronInLayer, int* indexOfFirstWeightInFrontOfLayer) {
    // use getDeviceProperties helper function to get GPU device information
    int numberOfSMs = 0; // the number of SMs on the device (1 SM can process 1 block at a time)
    int warpsize = 0; // the number of threads that an SM can manage at one time
    getDeviceProperties(&numberOfSMs, &warpsize); 

    // set blocks and threads to a size that will fully utilize the GPU (overkill, I know, but we're going for performance here)
    int blocks = numberOfSMs; // should be equal to the number of SMs on the GPU device after getDeviceProperties
    int threads = warpsize; // should be equal to the warpsize on the GPU device after getDeviceProperties
    

    // for each layer l in network (starting at the first non-input-layer): 
    // propagate the left-layer output's to the right-layer, activate the right-layer, then repeat until output-layer is activated
    for (int l = 1; l < numberOfLayers; l++) {
        int numberOfNeuronsInLeft = numberOfNeuronsInLayer[l-1]; // left-layer size
        int numberOfNeuronsInRight = numberOfNeuronsInLayer[l]; // right-layer size

        int indexOfFirstLeftNeuron = indexOfFirstNeuronInLayer[l-1]; // start at this neuron for left-layer
        int indexOfFirstRightNeuron = indexOfFirstNeuronInLayer[l]; // start at this neuron for right-layer
        int indexOfFirstWeight = indexOfFirstWeightInFrontOfLayer[l]; // start at this weight 

        // for each neuron nr in right-layer: 
        // combine left-layer's outputs with appropriate weights, then apply activation to the neuron
        for (int nr = 0; nr < numberOfNeuronsInRight; nr++) {
            // double or devide the number of threads until we have a number close to the number of neurons in right-layer
            threads = getOptimalThreadSize(blocks, threads, numberOfNeuronsInRight, warpsize);

            // using a cudaKernel: calculate the weighted sum plus bias for each neuron nr in right-layer and store result back in neuron nr
            cudaKernel_CalculateWeightedSumPlusBias<<<blocks, threads>>>(devNeurons, devWeights, devBiases, 
                                                                         numberOfNeuronsInLeft, numberOfNeuronsInRight, 
                                                                         indexOfFirstLeftNeuron, indexOfFirstRightNeuron, indexOfFirstWeight);

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                onCudaKernelLaunchFailure("cudaKernel_CalculateWeightedSumPlusBias", cudaStatus)
            }
    
            // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                onCudaDeviceSynchronizeError("cudaKernel_CalculateWeightedSumPlusBias", cudaStatus);
            }

            // using a cudaKernel: apply sigmoid activation on each neuron nr in right-layer and store result back in neuron nr
            cudaKernel_ActivateLayerUsingSigmoid<<<blocks, threads>>>(devNeurons, indexOfFirstRightNeuron, numberOfNeuronsInRight);

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                onCudaKernelLaunchFailure("cudaKernel_ActivateLayerUsingSigmoid", cudaStatus)
            }
    
            // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                onCudaDeviceSynchronizeError("cudaKernel_ActivateLayerUsingSigmoid", cudaStatus);
            }
        }
    }
} //end feedforwardUsingDevice function

/*
 * cudaKernel_CalculateWeightedSumPlusBias - calculates and stores the weighted sum plus bias for every neuron in a layer
 * __global__ decoration tells NVCC this function should run on GPU, and be callable from the CPU host
 * @params: devNeurons - device copy of double* neurons
 * @params: devWeights - device copy of double* weights
 * @params: devBiases - device copy of double* biases
 * @params: numberOfNeuronsInLeft - the int number of neurons in left-layer
 * @params: numberOfNeuronsInRight - the int number of neurons in right-layer
 * @params: indexOfFirstLeftNeuron - the int index of the first neuron in left-layer
 * @params: indexOfFirstRightNeuron - the int index of the first neuron in right-layer
 * @params: indexOfFirstWeight - the int index of the first weight between the two layers
 */
__global__ void static cudaKernel_CalculateWeightedSumPlusBias(double* devNeurons, double* devWeights, double* devBiases, 
                                                int numberOfNeuronsInLeft, int numberOfNeuronsInRight, 
                                                int indexOfFirstLeftNeuron, int indexOfFirstRightNeuron, int indexOfFirstWeight) {
    volatile unsigned int nr = threadIdx.x + blockIdx.x * blockDim.x; // calculate the thread id (used as offset from indexOfFirstRight)

    // check that this thread is within our desired range (extra threads may have been launched for GPU optimization)
    if (nr < numberOfNeuronsInRight) {
        double weightedSum = 0.0; // temporarily store the weighted sum for the neuron

        // for each neuron nl in left-layer:
        // combine neuron nl's output with the appropriate weight and store in weightedSum
        for (int nl = 0; nl < numberOfNeuronsInLeft; nl++) {
            weightedSum = weightedSum + ((devNeurons[indexOfFirstLeftNeuron + nl]) * (devWeights[indexOfFirstWeight + numberOfNeuronsInLeft * nr + nl]));
        }

        // store the weighted sum plus the bias in neuron nr (over-writes any previous value in neuron nr)
        devNeurons[indexOfFirstRightNeuron + nr] = weightedSum + devBiases[indexOfFirstRightNeuron + nr];

        // return after storing weighted sum plus bias, don't forget to activate this layer using cudaKernel_ActivateLayerUsingSigmoid after returning
    }
} //end cudaKernel_CalculateWeightedSumPlusBias function

