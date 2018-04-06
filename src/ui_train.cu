/*******************************************************************************************
 * Filename: ui_train.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for training a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_train method - user interface for training a model */
void ui_train() {
    // declare variables needed to store the model information
    char inputBuffer[MAXINPUT]; // store the user's input (gets recycled a lot)
    int tempInt; // store temp int input from user (used for determining whether to run on host or GPU device)
    int numberOfLayers; // store the total number of layers in the network
    int numberOfNeuronsTotal; // store the total number of neurons in our neural network
    int numberOfWeightsTotal; // store the total number of weights in our neural network
    int* numberOfNeuronsPerLayer; // store the total number of neurons in each layer in our neural network in a 1d array of size numberOfLayers
    int* numberOfWeightsPerLayer; // store the total number of weights between each layer in our neural network in a 1d array of size numberOfLayers-1
    int* firstNeuronIndexPerLayer; // store the indexes of each layer's first neuron value
    int* firstWeightIndexPerLayer; // store the indexes of each layer's first weight value
    double* neurons; // store the neuron values of our neural network in a 1d array of size neuronSize (1d arrays are easy to work with in CUDA)
    double* weights; // store the weight values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    double* biases; // store the biases values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    double* outputExpected; // store the outputExpected output values for backpropagation
    double* neuronErrors; // store the error "cost" of each neuron during backpropagation
    int epochs; // store the number of epochs for training
    double learningRate; // store the rate that our network will learn

    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    // initialize pointers with malloc (will be resized in readmodel.cu later)
    numberOfNeuronsPerLayer = (int *) malloc(sizeof(int));
    if (numberOfNeuronsPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    numberOfWeightsPerLayer = (int *) malloc(sizeof(int));
    if (numberOfWeightsPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    firstNeuronIndexPerLayer = (int *) malloc(sizeof(int));
    if (firstNeuronIndexPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    firstWeightIndexPerLayer = (int *) malloc(sizeof(int));
    if (firstWeightIndexPerLayer == NULL) {
        onMallocError(sizeof(int));
    }

    weights = (double *) malloc(sizeof(double));
    if (weights == NULL) {
        onMallocError(sizeof(double));
    }

    biases = (double *) malloc(sizeof(double));
    if (biases == NULL) {
        onMallocError(sizeof(double));
    }

    printf("Searching ./nmModel for files...\n");

    readModel(&numberOfLayers, &numberOfNeuronsTotal, &numberOfWeightsTotal, &numberOfNeuronsPerLayer, &numberOfWeightsPerLayer, &firstNeuronIndexPerLayer,
            &firstWeightIndexPerLayer, &weights, &biases, &learningRate, &epochs);

    printf("...files found!\n");

#ifdef DEBUG
    // print out information read from disk
    printf("epochs                 = %d\n", epochs);
    printf("learningRate           = %lf\n", learningRate);
    printf("numberOfLayers         = %d\n", numberOfLayers);
    printf("numberOfNeuronsTotal   = %d\n", numberOfNeuronsTotal);// remember, numberOfNeuronsTotal equals numberOfBiasesTotal
    printf("numberOfWeightsTotal   = %d\n", numberOfWeightsTotal);

    for(int i = 0; i < numberOfLayers; i++) {
        printf("numberOfNeuronsPerLayer[%d]  = %d\n", i, numberOfNeuronsPerLayer[i]);
        printf("numberOfWeightsPerLayer[%d]  = %d\n", i, numberOfWeightsPerLayer[i]);
        printf("firstNeuronIndexPerLayer[%d] = %d\n", i, firstNeuronIndexPerLayer[i]);
        printf("firstWeightIndexPerLayer[%d] = %d\n", i, firstWeightIndexPerLayer[i]);
    }

    //printarray("biases", biases, numberOfNeuronsTotal);
    //printarray("weights", weights, numberOfWeightsTotal);
#endif

    // malloc memory for uninitialized arrays
    neurons = (double *) malloc(numberOfNeuronsTotal * sizeof(double));
    if (neurons == NULL) {
        onMallocError(numberOfNeuronsTotal * sizeof(double));
    }

    outputExpected = (double *) malloc(numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(double));
    if (outputExpected == NULL) {
        onMallocError(numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(double));
    }

    neuronErrors = (double *) malloc(numberOfNeuronsTotal * sizeof(double));
    if (neuronErrors == NULL) {
        onMallocError(numberOfNeuronsTotal * sizeof(double));
    }

    printf("Loading MNIST training samples into memory (this might take awhile)...\n");

    // Load MNIST training data and labels into memory
    unsigned char* trainingData;
    char* trainingLabels;
    int numberOfTrainingSamples = 0;

    trainingData = (unsigned char *) malloc(sizeof(char));
    if (trainingData == NULL) {
        onMallocError(sizeof(char));
    }

    trainingLabels = (char *) malloc(sizeof(char));
    if (trainingLabels == NULL) {
        onMallocError(sizeof(char));
    }

    loadMnistTrainingSamples(&trainingData, &trainingLabels, &numberOfTrainingSamples);

    printf("...MNIST training samples loaded!\n");

    // get user input for running on CPU or GPU
    tempInt = 5; // assign 5 to enter loop

    while (1) {
        printf("Do you want to train on the host machine or GPU device?\nEnter 1 for host or 2 for device:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%d", &tempInt); // format and dump the user's input

        if ((tempInt == 1) || (tempInt == 2)) {
            break;
        } else {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    if (tempInt == 1) {
        printf("Today we keep tradition, looks like we're training on the host machine!\n");

        printf("Press enter to begin training:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        printf("\n");

        printf("Beginning training on host now...");

        // do(LOADINPUT, FEEDFORWARD, COMPAREOUTPUT, BACKPROPAGATEERRS) for all samples in batch, WEIGHTUPDATE & BIASUPDATE, then repeat until i == epochs
        for (int i = 0; i < epochs; i++) {
            // for each sample: loadTrainingData, feedforward, compareOutput, backpropagate error signal
            for (int s = 0; s < numberOfTrainingSamples; s++) {
                // load pixel data from an MNIST sample into the input layer
                loadNextMnistSampleData(&neurons, trainingData, s);
                loadNextMnistSampleLabel(&outputExpected, trainingLabels, s);

                // feedforward the data in the input layer
                feedforwardWithHost(neurons, weights, biases, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
                        firstWeightIndexPerLayer);

                // calculate and backpropagate error signals
                backpropagateWithHost(outputExpected, neurons, weights, biases, neuronErrors, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer,
                        firstNeuronIndexPerLayer, firstWeightIndexPerLayer);

                // use error signal to update weights and biases
                updateWeights(neurons, &weights, neuronErrors, numberOfLayers, numberOfNeuronsPerLayer, firstNeuronIndexPerLayer, firstWeightIndexPerLayer,
                        learningRate);
                updateBiases(neurons, &biases, neuronErrors, numberOfNeuronsTotal, learningRate);
                if (s % 1000 == 0) {
                    printf("...epoch %d of %d: sample %d seen...\n", i + 1, epochs, s);
                }
            }
        }
        printf("...training complete!\n");

    } else if (tempInt == 2) {
        printf("Today we break with tradition, looks like we're training on the GPU device!\n");
        // declare our device variables
        double* devNeurons;
        double* devWeights;
        double* devBiases;
        double* devOutputExpected;
        double* devNeuronErrors;
        unsigned char* devTrainingData;
        char* devTrainingLabels;

        // declare our cudaStatus variable
        cudaError_t cudaStatus;

        // run on GPU 0, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            onFailToSetGPUDevice();
        }

        printf("Allocating GPU device memory...\n");

        // allocate device memory for device variables and copy host values to device copies
        cudaStatus = cudaMalloc((void **) &devNeurons, (numberOfNeuronsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsTotal * sizeof(double));
        }

        cudaStatus = cudaMalloc((void **) &devWeights, (numberOfWeightsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfWeightsTotal * sizeof(double));
        }

        cudaStatus = cudaMalloc((void **) &devBiases, (numberOfNeuronsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsTotal * sizeof(double));
        }

        cudaStatus = cudaMalloc((void **) &devNeuronErrors, (numberOfNeuronsTotal * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsTotal * sizeof(double));
        }

        cudaStatus = cudaMalloc((void **) &devOutputExpected, (numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(double))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(double));
        }

        cudaStatus = cudaMalloc((void **) &devTrainingData, (MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char));
        }

        cudaStatus = cudaMalloc((void **) &devTrainingLabels, (MNISTTRAININGSETSIZE * sizeof(char))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(MNISTTRAININGSETSIZE * sizeof(char));
        }

        printf("...allocation successful!\n");

        printf("Copying over Host values to GPU device...");

        cudaStatus = cudaMemcpy(devWeights, weights, (numberOfWeightsTotal * sizeof(double)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("weights");
        }

        cudaStatus = cudaMemcpy(devBiases, biases, (numberOfNeuronsTotal * sizeof(double)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("biases");
        }

        cudaStatus = cudaMemcpy(devTrainingData, trainingData, (MNISTSAMPLEDATASIZE * MNISTTRAININGSETSIZE * sizeof(char)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("trainingData");
        }

        cudaStatus = cudaMemcpy(devTrainingLabels, trainingLabels, (MNISTTRAININGSETSIZE * sizeof(char)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("trainingLabels");
        }

        printf("...copy successful!\n");

        // use getDeviceProperties helper function to determine the numBlocks and threadsPerBlock before launching CUDA Kernels
        int numBlocks = 5; // set 5 as default, should be equal to the number of SMs on the GPU device
        int threadsPerBlock = 32; // set 32 as default, should be equal to the warpsize on the GPU device
        getDeviceProperties(&numBlocks, &threadsPerBlock);

        // stretch threadsPerBlock when more threads are needed to call a kernel on all neurons
        int scalar = 1;
        int largestLayerSize = 0;
        for (int i = 0; i < numberOfLayers; i++) {
            if (largestLayerSize < numberOfNeuronsPerLayer[i]) {
                largestLayerSize = numberOfNeuronsPerLayer[i];
            }
        }
        if (numBlocks * threadsPerBlock < largestLayerSize) {
            scalar = ((largestLayerSize / numBlocks) / threadsPerBlock) + 1;
        }
        threadsPerBlock = threadsPerBlock * scalar;

        printf("Beginning training on GPU device now...");

        // do(LOADINPUT, FEEDFORWARD, COMPAREOUTPUT, BACKPROPAGATEERRS) for all samples in batch, WEIGHTUPDATE & BIASUPDATE, then repeat until i == epochs
        for (int i = 0; i < epochs; i++) {
            // for each sample: loadTrainingData, feedforward, compareOutput, backpropagate error signal
            for (int s = 0; s < numberOfTrainingSamples; s++) {
                // load pixel data from an MNIST sample into the input layer
                loadNextMnistSampleDataKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devTrainingData, s);
                loadNextMnistSampleLabelKernel<<<numBlocks, threadsPerBlock>>>(devOutputExpected, devTrainingLabels, s);

                cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel

                // feedforward the data in the input layer
                feedforwardWithDevice(numBlocks, threadsPerBlock, devNeurons, devWeights, devBiases, numberOfLayers, numberOfNeuronsPerLayer,
                        numberOfWeightsPerLayer, firstNeuronIndexPerLayer, firstWeightIndexPerLayer);

                // calculate and backpropagate error signals
                backpropagateWithDevice(numBlocks, threadsPerBlock, devOutputExpected, devNeurons, devWeights, devBiases, devNeuronErrors, numberOfLayers,
                        numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer, firstWeightIndexPerLayer);

                // for each node in the output layer, calculate the output error (spawn 1 thread for each neuron in the output layer)
                int outputLayerIndex = numberOfLayers - 1;

                // for each layer l between output and input, visit in reverse order, backpropagate error values and update weights
                for (int l = outputLayerIndex - 1; l > 0; l--) {

                    // for each node in layer l, use error signal (devNeuronErrors) to update the devWeights and devBiases
                    // spawn 1 block for each neuron in layer l and, in each block, spawn 1 thread for each neuron in layer l+1
                    weightUpdateKernel<<<numberOfNeuronsPerLayer[l], numberOfNeuronsPerLayer[l + 1]>>>(devNeurons, devWeights, devNeuronErrors,
                            numberOfNeuronsPerLayer[l], numberOfNeuronsPerLayer[l + 1], numberOfWeightsPerLayer[l + 1], firstNeuronIndexPerLayer[l],
                            firstNeuronIndexPerLayer[l + 1], learningRate);
                    cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel
                }
                biasUpdateKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devBiases, devNeuronErrors, numberOfNeuronsTotal, learningRate);
                cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernel

            }

            if (i % 10 == 0) {
                printf("...%d epochs complete...", i);
            }
        }

        printf("...training complete!\n");

        printf("Copying over GPU device values to Host...");

        cudaStatus = cudaMemcpy(weights, devWeights, (numberOfWeightsTotal * sizeof(double)), cudaMemcpyDeviceToHost); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("devWeights");
        }

        cudaStatus = cudaMemcpy(biases, devBiases, (numberOfNeuronsTotal * sizeof(double)), cudaMemcpyDeviceToHost); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("devBiases");
        }

        printf("...copy successful!\n");

        printf("Press enter to free dynamically allocated GPU device memory.\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

        printf("Freeing dynamically allocated GPU device memory...");

        // free the chunks of device memory that were dynamically allocated by cudaMalloc
        cudaFree(devNeurons);
        cudaFree(devWeights);
        cudaFree(devBiases);
        cudaFree(devNeuronErrors);
        cudaFree(devTrainingData);
        cudaFree(devTrainingLabels);
        printf("memory freed!\n");
    }

    // SAVE TRAINED WEIGHTS AND BIASES TO DISK
    printf("Saving trained weights and biases to disk...");
    saveWeightsToDisk(weights, numberOfWeightsTotal);
    saveBiasesToDisk(biases, numberOfNeuronsTotal);
    printf("saving complete!\n");

    printf("Press enter to free dynamically allocated host memory.\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

    // free the chunks of host memory that were dynamically allocated by malloc
    printf("Freeing dynamically allocated host memory...");
    free(numberOfNeuronsPerLayer);
    free(numberOfWeightsPerLayer);
    free(firstNeuronIndexPerLayer);
    free(firstWeightIndexPerLayer);
    free(neurons);
    free(weights);
    free(neuronErrors);
    free(outputExpected);
    printf("memory freed!\n");

    printf("Press enter to return to the main menu:\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    printf("\n");
} //end ui_train function
