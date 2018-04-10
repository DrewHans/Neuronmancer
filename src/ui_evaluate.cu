/*******************************************************************************************
 * Filename: ui_evaluate.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for evaluating a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_evaluate method - user interface for evaluating a model */
void ui_evaluate() {
    // declare helper variables for ui_evaluate
    char inputBuffer[MAXINPUT]; // stores the user's input (gets recycled a lot)
    int tempInt -1; // stores int input from user (used for determining whether to run on host or GPU device)
    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    // declare variables needed to store the model's structure / testing information
    float learningRate; // the rate that our network will learn
    unsigned int epochs; // the number of epochs for training (in a single epoch: see all training samples then use deltas for weight/bias update)
    unsigned int numberOfLayers; // the total number of layers in the network
    unsigned int numberOfNeuronsTotal; // the total number of neurons in the network
    unsigned int numberOfWeightsTotal; // the total number of weights in the network
    unsigned int* numberOfNeuronsInLayer; // the total number of neurons in each layer (1d array of size numberOfLayers)
    unsigned int* numberOfWeightsInFrontOfLayer; // the number of weights between each layer (1d array of size numberOfLayers)
    unsigned int* indexOfFirstNeuronInLayer; // the indexes of each layer's first neuron value (1d array of size numberOfLayers)
    unsigned int* indexOfFirstWeightInFrontOfLayer; // the indexes of the first weight value in front of each layer (1d array of size numberOfLayers)
    unsigned int numberOfTestingSamples = 0; // the number of test samples in the MNIST test set

    // declare variables needed to store important model values and MNIST testing set values
    float* neurons; // the neuron values of the neural network (1d array of size numberOfNeuronsTotal)
    float* weights; // the weight values of the neural network (1d array of size numberOfWeightsTotal)
    float* biases; // the biases values of the neural network (1d array of size numberOfNeuronsTotal)
    float* expected; // the expected output values of a single sample (1D array of size numberOfNeuronsInLayer[numberOfLayers-1])
    char* testingLabels; // the labels of each testing sample (1D array of size numberOfTestingSamples)
    unsigned char* testingData; // the pixel-values of all testing samples (1d array of size numberOfTestingSamples * MNISTSAMPLEDATASIZE)

    // declare variables needed to store device copies of the important model values and MNIST testing set values
    float* devNeurons; // device copy of neurons
    float* devWeights; // device copy of weights
    float* devBiases; // device copy of biases
    float* devExpected; // device copy of expected
    char* devTestingLabels; // device copy of testingLabels
    unsigned char* devTestingData; // device copy of testingData

    // declare variables needed to generate the confusion matrix
    int* confusionMatrix; // the confusion matrix, gets filled during evaluation and printed before falling out of scope
    int classification = 0; // holds the network's calculated classification of a sample (also holds devClassification value during GPU evaluation)
    int* devClassification; // device variable that holds the network's calculated classification of a sample during GPU evaluation

    // dynamically allocate memory to store the biases and weight values
    confusionMatrix = (int*) malloc(MNISTCLASSIFICATIONS * 2 * sizeof(int));
    if (confusionMatrix == NULL) {
        onMallocError(MNISTCLASSIFICATIONS * 2 * sizeof(int));
    }

    // initialize model structure pointers to memory with malloc (will be resized and filled with values read from disk in model_read.cu)
    numberOfNeuronsInLayer = (unsigned int *) malloc(1 * sizeof(int));
    if (numberOfNeuronsInLayer == NULL) {
        onMallocError(1 * sizeof(int));
    }

    numberOfWeightsInFrontOfLayer = (int *) malloc(1 * sizeof(int));
    if (numberOfWeightsInFrontOfLayer == NULL) {
        onMallocError(1 * sizeof(int));
    }

    indexOfFirstNeuronInLayer = (int *) malloc(1 * sizeof(int));
    if (indexOfFirstNeuronInLayer == NULL) {
        onMallocError(1 * sizeof(int));
    }

    indexOfFirstWeightInFrontOfLayer = (int *) malloc(1 * sizeof(int));
    if (indexOfFirstWeightInFrontOfLayer == NULL) {
        onMallocError(1 * sizeof(int));
    }

    weights = (float *) malloc(1 * sizeof(float));
    if (weights == NULL) {
        onMallocError(sizeof(double));
    }

    biases = (float *) malloc(1 * sizeof(float));
    if (biases == NULL) {
        onMallocError(sizeof(double));
    }

    printf("Searching %s for files...", MODELDIRECTORY);

    // attempt to read in model from disk
    readModel(&learningRate, &epochs, &numberOfLayers, &numberOfNeuronsTotal, &numberOfWeightsTotal, &numberOfNeuronsInLayer, 
              &numberOfWeightsInFrontOfLayer, &indexOfFirstNeuronInLayer, &indexOfFirstWeightInFrontOfLayer, &weights, &biases);

    printf("...files found!\n"
           "Model structure read from disk:\n");

    // print out information read from disk
    printf("- epochs =================> %u\n", epochs);
    printf("- learningRate ===========> %f\n", learningRate);
    printf("- numberOfLayers =========> %u\n", numberOfLayers);
    printf("- numberOfNeuronsTotal ===> %u\n", numberOfNeuronsTotal); // remember, numberOfNeuronsTotal equals numberOfBiasesTotal
    printf("- numberOfWeightsTotal ===> %u\n", numberOfWeightsTotal);

    for(int i = 0; i < numberOfLayers; i++) {
        printf("--- numberOfNeuronsInLayer[%d] =============> %u\n", i, numberOfNeuronsInLayer[i]);
        printf("--- numberOfWeightsInFrontOfLayer[%d] ======> %u\n", i, numberOfWeightsInFrontOfLayer[i]);
        printf("--- indexOfFirstNeuronInLayer[%d] ==========> %u\n", i, indexOfFirstNeuronInLayer[i]);
        printf("--- indexOfFirstWeightInFrontOfLayer[%d] ===> %u\n", i, indexOfFirstWeightInFrontOfLayer[i]);
    }

    //printarray("biases", biases, numberOfNeuronsTotal);
    //printarray("weights", weights, numberOfWeightsTotal);
    printf("Press enter if this looks like your model structure (ctrl-c to abort):\n"
           "~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

    printf("Alright, sit tight while I do some work...\n"
           "- attempting to allocate memory for neurons and expected...");


    // malloc memory for uninitialized arrays using values we read from disk
    neurons = (float* ) malloc(numberOfNeuronsTotal * sizeof(float));
    if (neurons == NULL) {
        onMallocError(numberOfNeuronsTotal * sizeof(float));
    }

    expected = (float *) malloc(numberOfNeuronsInLayer[numberOfLayers - 1] * sizeof(float));
    if (expected == NULL) {
        onMallocError(numberOfNeuronsInLayer[numberOfLayers - 1] * sizeof(float));
    }

    printf("allocation successful!\n"
           "- initializing neurons to zero (this might take a while)...");

    initArrayToZeros(&neurons, numberOfNeuronsTotal); // cleans up any garbage we may have picked up

    printf("neurons initialized!\n"
           "- initializing expected to zero (this might take a while)...");

    initArrayToZeros(&expected, numberOfNeuronsInLayer[numberOfLayers - 1]); // cleans up any garbage we may have picked up

    printf("expected initialized!\n"
           "- loading MNIST testing samples into memory (this might take a while)...");

    // initialize MNIST testing samples pointers to memory with malloc (will be resized and filled with values read from disk in functions_mnist.cu)
    testingLabels = (char *) malloc(1 * sizeof(char));
    if (testingLabels == NULL) {
        onMallocError(1 * sizeof(char));
    }

    testingData = (unsigned char *) malloc(1 * sizeof(char));
    if (testingData == NULL) {
        onMallocError(1 * sizeof(char));
    }

    // read MNIST test data and labels from disk and load into memory
    readMnistTestSamplesFromDisk(&testingData, &testingLabels, &numberOfTestingSamples);

    printf("...samples loaded!\n"
           "Alright, we're just about ready to start evaluating!\n");

    // get user input for running on CPU or GPU
    tempInt = 5; // assign 5 to enter loop

    while (1) {
        printf("Do you want to evaluate on the host machine or GPU device?\n"
               "Enter 1 for host or 2 for device:\n"
               "~");
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
        // HOST EVALUATION LOGIC BELOW

        printf("Looks like you want to evaluate using the host machine!\n"
               "Press enter to begin evaluating on host machine (ctrl-c to abort):\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        printf("\nBeginning evaluation on host now...\n");

        // for each sample loop:
        // do (1) loadMnistSampleUsingHost, do (2) feedforwardUsingHost, 
        // do (3) getCalculatedMNISTClassificationUsingHost, then do (4) update confusion matrix
        for (unsigned int s = 0; s < numberOfTestingSamples; s++) {

            // (1) load sample s's mnistData into input-layer neurons and s's mnistlabel into expected
            loadMnistSampleUsingHost(testingLabels, testingData, s, (s * MNISTSAMPLEDATASIZE), &expected, &neurons);

            // (2) feedforward sample s's mnistData through the network (left to right)
            feedforwardUsingHost(&neurons, weights, biases, numberOfLayers, 
                                     numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer, 
                                     indexOfFirstNeuronInLayer, indexOfFirstWeightInFrontOfLayer);

            // (3) get the network's calculated classification
            classification = getCalculatedMNISTClassificationUsingHost(neurons, indexOfFirstNeuronInLayer[numberOfLayers-1]);

            // (4) update confusion matrix
            confusionMatrix[testingLabels[s] * MNISTCLASSIFICATIONS + classification] += 1;

        }//end for each sample loop

        printf("Evaluation on host is now complete!\n");

        // HOST EVALUATION LOGIC ABOVE
    } else if (tempInt == 2) {
        // GPU DEVICE EVALUATION LOGIC BELOW

        printf("Looks like you want to evaluate using the GPU!\n"
               "Alright, sit tight while I prep the device...\n"
               "- searching for cuda-enabled GPU device...");

        // declare cudaStatus variable to check for success of cuda operations
        cudaError_t cudaStatus;

        // run on GPU 0, this will need to be changed on a multi-GPU system
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            onFailToSetGPUDevice();
        }

        printf("cuda-enabled GPU detected!\n"
               "- attempting to allocate device memory for devNeurons, devWeights, devBiases, devExpected, and devClassification...");

        // allocate device memory for devNeurons
        cudaStatus = cudaMalloc((void **) &devNeurons, (numberOfNeuronsTotal * sizeof(float)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsTotal * sizeof(float));
        }

        // allocate device memory for devWeights
        cudaStatus = cudaMalloc((void **) &devWeights, (numberOfWeightsTotal * sizeof(float)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfWeightsTotal * sizeof(float));
        }

        // allocate device memory for devBiases
        cudaStatus = cudaMalloc((void **) &devBiases, (numberOfNeuronsTotal * sizeof(float)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsTotal * sizeof(float));
        }

        // allocate device memory for devExpected
        cudaStatus = cudaMalloc((void **) &devExpected, (numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(float)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(float));
        }

        // allocate device memory for devClassification
        cudaStatus = cudaMalloc((void **) &devClassification, (1 * sizeof(int)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(1 * sizeof(int));
        }

        printf("allocation successful!\n"
               "- attempting to allocate device memory for MNIST testing set...");

        // allocate device memory for devTestingLabels
        cudaStatus = cudaMalloc((void **) &devTestingLabels, (MNISTTESTSETSIZE * sizeof(char)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(MNISTTESTSETSIZE * sizeof(char));
        }

        // allocate device memory for devTestingData
        cudaStatus = cudaMalloc((void **) &devTestingData, (MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char)));
        if (cudaStatus != cudaSuccess) {
            onCudaMallocError(MNISTSAMPLEDATASIZE * MNISTTESTSETSIZE * sizeof(char));
        }

        printf("allocation successful!\n"
               "- copying neurons, weights, biases, and expected values from host memory to device memory (this might take a while)...");

        // copy neurons to device
        cudaStatus = cudaMemcpy(devNeurons, neuron, (numberOfNeuronsTotal * sizeof(float)), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("neuron");
        }

        // copy weights to device
        cudaStatus = cudaMemcpy(devWeights, weights, (numberOfWeightsTotal * sizeof(float)), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("weights");
        }

        // copy biases to device
        cudaStatus = cudaMemcpy(devBiases, biases, (numberOfNeuronsTotal * sizeof(float)), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("biases");
        }

        // copy expected to device
        cudaStatus = cudaMemcpy(devExpected, expected, (numberOfNeuronsPerLayer[numberOfLayers - 1] * sizeof(float)), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("expected");
        }

        printf("copy successful!\n"
               "- copying MNIST testing samples from host memory to device memory (this might take a while)...");

        // copy MNIST testing labels to device
        cudaStatus = cudaMemcpy(devTestingLabels, testingLabels, (MNISTTESTSETSIZE * sizeof(char)), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("testingLabels");
        }

        // copy MNIST testing data to device
        cudaStatus = cudaMemcpy(devTestingData, testingData, (MNISTTESTSETSIZE * MNISTSAMPLEDATASIZE * sizeof(char)), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            onCudaMemcpyError("testingData");
        }

        printf("copy successful!\n"
               "... device prep work complete!\n"
               "Press enter to begin evaluating on GPU device (ctrl-c to abort):\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        printf("\nBeginning evaluation on GPU now...\n");

        // for each sample loop:
        // do (1) loadMnistSampleUsingDevice, do (2) feedforwardUsingDevice, do (3) getCalculatedMNISTClassificationUsingDevice, 
        // do (4) copy devClassification to host, then do (5) update confusion matrix
        for (unsigned int s = 0; s < numberOfTestingSamples; s++) {

            // (1) load sample s's mnistData into input-layer neurons and s's mnistlabel into expected
            loadMnistSampleUsingDevice(devTestingLabels, devTestingData, s, (s * MNISTSAMPLEDATASIZE), devExpected, devNeurons);

            // (2) feedforward sample s's mnistData through the network (left to right)
            feedforwardUsingDevice(devNeurons, devWeights, devBiases, numberOfLayers, 
                                   numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer, 
                                   indexOfFirstNeuronInLayer, indexOfFirstWeightInFrontOfLayer);

            // (3) get the network's calculated classification
            getCalculatedMNISTClassificationUsingDevice(devClassification, devNeurons, indexOfFirstNeuronInLayer[numberOfLayers-1]);

            // (4) copy devClassification to host 
            classification = 0;
            cudaStatus = cudaMemcpy(classification, devClassification, (1 * sizeof(int)), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                onCudaMemcpyError("devClassification");
            }

            // (4) update confusion matrix
            confusionMatrix[testingLabels[s] * MNISTCLASSIFICATIONS + classification] += 1;

        }//end for each sample loop

        printf("Evaluation on GPU device is now complete!\n");

        // free the chunks of device memory that were dynamically allocated by cudaMalloc 
        // (do while in scope, then again before return to main just to be absolutely sure we didn't leak memory)
        cudaFree(devNeurons);
        cudaFree(devWeights);
        cudaFree(devBiases);
        cudaFree(devExpected);
        cudaFree(devTestingLabels);
        cudaFree(devTestingData);
        cudaFree(devClassification);

        // GPU DEVICE EVALUATION LOGIC ABOVE
    } else {
        // this should only execute if something goes horribly wrong with tempInt
        printf("I don't know how you did it, but you somehow broke out of my while-loop with something besides a 1 or 2...\n"
               "...as revenge I'm shutting you down. Don't mess with my program logic!");
        exit(1);
    }

    printf("Confusion Matrix:\n");
    printConfusionMatrix(confusionMatrix, MNISTCLASSIFICATIONS);

    // free the chunks of device memory that were dynamically allocated by cudaMalloc
    cudaFree(devNeurons);
    cudaFree(devWeights);
    cudaFree(devBiases);
    cudaFree(devExpected);
    cudaFree(devTestingLabels);
    cudaFree(devTestingData);

    printf("memory freed!\n"
           "- freeing dynamically allocated host memory...");

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsInLayer);
    free(numberOfWeightsInFrontOfLayer);
    free(indexOfFirstNeuronInLayer);
    free(indexOfFirstWeightInFrontOfLayer);
    free(neurons);
    free(weights);
    free(biases);
    free(expected);
    free(testingLabels);
    free(testingData);

    printf("memory freed!\n"
           "Press enter to return to the main menu:\n"
           "~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    printf("\n");
}//end ui_evaluate function

