/*******************************************************************************************
 * Filename: ui_evaluate.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for evaluating a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_evaluate method - user interface for evaluating a model */
void ui_evaluate() {
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
    int epochs; // store the number of epochs for training (it won't be used here, but readModel function requires it)
    double learningRate; // store the rate that our network will learn

    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    // declare variables used to generate the confusion matrix
    int mnistConfusionMatrix[10][10] = { 0 }; // store the confusion matrix (rows = actual class; cols = predicted class)
    double accuracy = 0.0; // store the accuracy of our model ( allCorrectPredictions / totalPredictions )
    double misclassificationRate = 1.0 - accuracy; // stores the model's error rate (equal to 1 - accuracy)

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

    printf("Loading MNIST test samples into memory (this might take awhile)...\n");

    // Load MNIST test data and labels into memory
    unsigned char* testData;
    char* testLabels;
    int numberOfTestSamples = 0;

    testData = (unsigned char *) malloc(sizeof(char));
    if (testData == NULL) {
        onMallocError(sizeof(char));
    }

    testLabels = (char *) malloc(sizeof(char));
    if (testLabels == NULL) {
        onMallocError(sizeof(char));
    }

    loadMnistTestSamples(&testData, &testLabels, &numberOfTestSamples);

    printf("...MNIST test samples loaded!\n");

    // get user input for running on CPU or GPU
    tempInt = 5; // assign 5 to enter loop

    while (1) {
        printf("Do you want to evaluate using the host machine or GPU device?\nEnter 1 for host or 2 for device:\n~");
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
        printf("Today we keep tradition, looks like we're evaluating on the host machine!\n");

        printf("Press enter to begin evaluation:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        printf("\n");

        printf("Beginning evaluation on host now...");

        // for each sample in batch: loadTestData, feedforward, compareOutput, then update confusion matrix
        for (int s = 0; s < numberOfTestSamples; s++) {
            // load pixel data from an MNIST sample into the input layer
            loadNextMnistSampleData(&neurons, testData, s);
            loadNextMnistSampleLabel(&outputExpected, testLabels, s);

            // feedforward the data in the input layer
            feedforwardWithHost(neurons, weights, biases, numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
                    firstWeightIndexPerLayer);

            // get the predicted MNIST class, the actual MNIST class, and then update the appropriate confusion matrix variable
            int classPrediction = getCalculatedMnistSampleClassification(neurons, firstNeuronIndexPerLayer[numberOfLayers - 1]);
            int classActual = testLabels[s];
            mnistConfusionMatrix[classActual][classPrediction] = mnistConfusionMatrix[classActual][classPrediction] + 1;
        }
        printf("evaluation complete!\n");

    } else if (tempInt == 2) {
        printf("Today we break with tradition, looks like we're evaluating on the GPU device!\n");
        // declare our device variables
        double* devNeurons;
        double* devWeights;
        double* devBiases;
        double* devOutputExpected;
        double* devNeuronErrors;
        unsigned char* devTestData;
        char* devTestLabels;

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

        cudaStatus = cudaMalloc((void **) &devTestData, (MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char));
        }

        cudaStatus = cudaMalloc((void **) &devTestLabels, (MNISTTESTSETSIZE * sizeof(char))); //cudaMalloc allocates a chunk of device memory
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(MNISTTESTSETSIZE * sizeof(char));
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

        cudaStatus = cudaMemcpy(devTestData, testData, (MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("testData");
        }

        cudaStatus = cudaMemcpy(devTestLabels, testLabels, (MNISTTESTSETSIZE * sizeof(char)), cudaMemcpyHostToDevice); //cudaMemcpy copies host values to device copies
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("testLabels");
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

        printf("Beginning evaluation on GPU device now...");

        // for each sample in batch: loadTestData, feedforward, compareOutput, then update confusion matrix
        for (int s = 0; s < numberOfTestSamples; s++) {
            // load pixel data from an MNIST sample into the input layer
            loadNextMnistSampleDataKernel<<<numBlocks, threadsPerBlock>>>(devNeurons, devTestData, s);
            loadNextMnistSampleLabelKernel<<<numBlocks, threadsPerBlock>>>(devOutputExpected, devTestLabels, s);

            cudaDeviceSynchronize(); // tell host to wait for device to finish previous kernels

            // feedforward the data in the input layer
            feedforwardWithDevice(numBlocks, threadsPerBlock, devNeurons, devWeights, devBiases, numberOfLayers, numberOfNeuronsPerLayer,
                    numberOfWeightsPerLayer, firstNeuronIndexPerLayer, firstWeightIndexPerLayer);

            // copy over the calculated neuron values (not ideal, but I'm out of ideas and I need this to work by April 21st)
            cudaStatus = cudaMemcpy(neurons, devNeurons, (numberOfNeuronsTotal * sizeof(double)), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                onCudaMemcpyError("devNeurons");
            }
            // get the predicted MNIST class, the actual MNIST class, and then update the appropriate confusion matrix variable
            int classPrediction = getCalculatedMnistSampleClassification(neurons, firstNeuronIndexPerLayer[numberOfLayers - 1]);
            int classActual = testLabels[s];
            mnistConfusionMatrix[classActual][classPrediction] = mnistConfusionMatrix[classActual][classPrediction] + 1;
        }

        printf("evaluation complete!\n");

        printf("Press enter to free dynamically allocated GPU device memory.\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

        printf("Freeing dynamically allocated GPU device memory...");

        // free the chunks of device memory that were dynamically allocated by cudaMalloc
        cudaFree(devNeurons);
        cudaFree(devWeights);
        cudaFree(devBiases);
        cudaFree(devNeuronErrors);
        cudaFree(devTestData);
        cudaFree(devTestLabels);
        printf("memory freed!\n");
    }

    // get the number of correct predictions from the confusion matrix
    int correctPredictions = 0;
    for (int i = 0; i < 10; i++) {
        correctPredictions = mnistConfusionMatrix[i][i];
    }
    accuracy = ((double) correctPredictions) / ((double) numberOfTestSamples);
    misclassificationRate = 1.0 - accuracy;

    // print out the confusion matrix (try to make it easy to read)
    printf("              predicted\n         ");
    for (int i = 0; i < 10; i++) {
        printf("%4d ", i);
    }
    printf("\n");
    for (int actualClass = 0; actualClass < 10; actualClass++) {
        printf("actual %d ", actualClass);
        for (int predictedClass = 0; predictedClass < 10; predictedClass++) {
            printf("%4d ", mnistConfusionMatrix[actualClass][predictedClass]);
        }
        printf("\n");
    }
    printf("Accuracy = %.4lf\nMisclassificationRate = %.4lf\n", accuracy, misclassificationRate);

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
} //end ui_evaluate function
