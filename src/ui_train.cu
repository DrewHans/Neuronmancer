/*******************************************************************************************
 * Filename: ui_train.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for training a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_train method - user interface for training a model */
void ui_train() {
    // declare helper variables for ui_create
    char inputBuffer[MAXINPUT]; // stores the user's input (gets recycled a lot)
    int tempInt -1; // stores int input from user (used for determining whether to run on host or GPU device)
    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    // declare variables needed to store the model's structure / training information
    float learningRate; // the rate that our network will learn
    unsigned int epochs; // the number of epochs for training (in a single epoch: see all training samples then use deltas for weight/bias update)
    unsigned int numberOfLayers; // the total number of layers in the network
    unsigned int numberOfNeuronsTotal; // the total number of neurons in the network
    unsigned int numberOfWeightsTotal; // the total number of weights in the network
    unsigned int* numberOfNeuronsInLayer; // the total number of neurons in each layer (1d array of size numberOfLayers)
    unsigned int* numberOfWeightsInFrontOfLayer; // the number of weights between each layer (1d array of size numberOfLayers)
    unsigned int* indexOfFirstNeuronInLayer; // the indexes of each layer's first neuron value (1d array of size numberOfLayers)
    unsigned int* indexOfFirstWeightInFrontOfLayer; // the indexes of the first weight value in front of each layer (1d array of size numberOfLayers)
    unsigned int numberOfTrainingSamples = 0; // the number of training samples in the MNIST training set

    // declare variables needed to store important model values and MNIST training set values
    float* neuronDeltas; // the delta value for each neuron (used to update weights / biases) 
    float* neurons; // the neuron values of the neural network (1d array of size numberOfNeuronsTotal)
    float* weights; // the weight values of the neural network (1d array of size numberOfWeightsTotal)
    float* biases; // the biases values of the neural network (1d array of size numberOfNeuronsTotal)
    float* expected; // the expected output values of a single sample (1D array of size numberOfNeuronsInLayer[numberOfLayers-1])
    char* trainingLabels; // the labels of each training sample (1D array of size numberOfTrainingSamples)
    unsigned char* trainingData; // the pixel-values of all training samples (1d array of size numberOfTrainingSmaples * MNISTSAMPLEDATASIZE)

    // declare variables needed to store device copies of the important model values and MNIST training set values
    float* devNeuronDeltas; // device copy of neuronDeltas
    float* devNeurons; // device copy of neurons
    float* devWeights; // device copy of weights
    float* devBiases; // device copy of biases
    float* devExpected; // device copy of expected
    char* devTrainingLabels; // device copy of trainingLabels
    unsigned char* devTrainingData; // device copy of trainingData

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////BELOW IS UNDER CONSTRUCTION///////////////////////////////////////////////////////////
///                                                                                                                                 ///

    printf("Alright, sit tight while I do some work...\n"
           "- attempting to allocate memory for neuronDeltas, neurons, and expected...");


    // malloc memory for uninitialized arrays using values we read from disk
    neuronDeltas = (float *) malloc(numberOfNeuronsTotal * sizeof(float));
    if (neuronDeltas == NULL) {
        onMallocError(numberOfNeuronsTotal * sizeof(float));
    }

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
           "- initializing neuronDeltas to zero (this might take a while)...");

    initArrayToZeros(&neuronDeltas, numberOfNeuronsTotal); // cleans up any garbage we may have picked up

    printf("neuronDeltas initialized!\n"
           "- loading MNIST training samples into memory (this might take a while)...");

    // initialize MNIST training samples pointers to memory with malloc (will be resized and filled with values read from disk in functions_mnist.cu)
    trainingLabels = (char *) malloc(1 * sizeof(char));
    if (trainingLabels == NULL) {
        onMallocError(1 * sizeof(char));
    }

    trainingData = (unsigned char *) malloc(1 * sizeof(char));
    if (trainingData == NULL) {
        onMallocError(1 * sizeof(char));
    }

    // Load MNIST training data and labels into memory
    loadMnistTrainingSamples(&trainingData, &trainingLabels, &numberOfTrainingSamples);

    printf("...samples loaded!\n"
           "Alright, we're just about ready to start training!\n");

    // get user input for running on CPU or GPU
    tempInt = 5; // assign 5 to enter loop

    while (1) {
        printf("Do you want to train on the host machine or GPU device?\n"
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
        // HOST TRAINING LOGIC BELOW

        printf("Looks like you want to train using the host machine!\n"
               "Press enter to begin training on host machine (ctrl-c to abort):\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        printf("\nBeginning training on host now...");

        // for each epoch: 
        // do (A) zero out neuronDeltas, do (B) complete "for each sample" loop, then do (C) update values
        for (unsigned int i = 0; i < epochs; i++) {

            // (A) zero out neuronDeltas (start fresh)
            initArrayToZeros(&neuronDeltas, numberOfNeuronsTotal); // cleans previous epochs error values


            // (B) for each sample loop:
            // do (B1) loadMnistSampleUsingHost, do (B2) feedforwardUsingHost, 
            // do (B3) getCalculatedMNISTClassificationUsingHost, then do (B4) backpropagationUsingHost
            for (unsigned int s = 0; s < numberOfTrainingSamples; s++) {

                // (B1) load sample s's mnistData into input-layer neurons and s's mnistlabel into expected
                loadMnistSampleUsingHost(trainingLabels, trainingData, s, (s * MNISTSAMPLEDATASIZE), &expected, &neurons);

                // (B2) feedforward sample s's mnistData through the network (left to right)
                feedforwardUsingHost(&neurons, weights, biases, numberOfLayers, 
                                     numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer, 
                                     indexOfFirstNeuronInLayer, indexOfFirstWeightInFrontOfLayer);

                // (B3) get the network's calculated classification
                getCalculatedMNISTClassificationUsingHost(neurons, indexOfFirstNeuronInLayer[numberOfLayers-1]);

                // (B4) backpropagate error signals and add this sample's deltas to neuronDeltas for this epoch
                backpropagationUsingHost(&neuronDeltas, expected, neurons, weights, biases, numberOfLayers, 
                                         numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer,
                                         indexOfFirstNeuronInLayer, indexOfFirstWeightInFrontOfLayer);

            }//end for each sample loop

            // (C) update values:
            // (C1) updateBiasesUsingHost
            updateBiasesUsingHost(neuronDeltas, neurons, &biases, numberOfNeuronsTotal, learningRate);

            // (C2) updateWeightsUsingHost
            updateWeightsUsingHost(neuronDeltas, neurons, &weights, numberOfLayers, 
                                   numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer, 
                                   indexOfFirstNeuronInLayer, indexOfFirstWeightInFrontOfLayer, learningRate);

            if (s % 3000 == 0) {
                printf("--- epoch %d of %d complete ---\n", i, epochs);
            }

        }//end for each epoch loop

        printf("Training on host is now complete!\n");

        // HOST TRAINING LOGIC ABOVE
    } else if (tempInt == 2) {
        // GPU DEVICE TRAINING LOGIC BELOW

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
////////////////// everything below is especially under co. /////////////////
//                                                                         //
        printf("Looks like you want to train using the GPU!\n"
               "Press enter to begin training on GPU device (ctrl-c to abort):\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        printf("\nBeginning training on GPU now...");






//                                                                         //
////////////////// everything above is especially under co. /////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

        // GPU DEVICE TRAINING LOGIC ABOVE
    } else {
        // this should only execute if something goes horribly wrong with tempInt
        printf("I don't know how you did it, but you somehow broke out of my while-loop with something besides a 1 or 2...\n"
               "...as revenge I'm shutting you down. Don't mess with my program logic!");
        exit(1);
    }

///                                                                                                                                 ///
////////////////////////////////////////////////ABOVE IS UNDER CONSTRUCTION////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    printf("Alright, sit tight while I do some work...\n");


    // save the weights and biases to disk after training (they should be different)
    printf("- attempting to save trained weights and biases to disk...");
    saveBiasesToDisk(biases, numberOfNeuronsTotal);
    saveWeightsToDisk(weights, numberOfWeightsTotal);
    printf("weights and biases saved!\n"
           "- freeing dynamically allocated device memory...");

    // free the chunks of device memory that were dynamically allocated by cudaMalloc
    cudaFree(devNeuronDeltas);
    cudaFree(devNeurons);
    cudaFree(devWeights);
    cudaFree(devBiases);
    cudaFree(devExpected);
    cudaFree(devTrainingLabels);
    cudaFree(devTrainingData);

    printf("memory freed!\n"
           "- freeing dynamically allocated host memory...");

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsInLayer);
    free(numberOfWeightsInFrontOfLayer);
    free(indexOfFirstNeuronInLayer);
    free(indexOfFirstWeightInFrontOfLayer);
    free(neuronDeltas);
    free(neurons);
    free(weights);
    free(biases);
    free(expected);
    free(trainingLabels);
    free(trainingData);

    printf("memory freed!\n"
           "Press enter to return to the main menu:\n"
           "~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    printf("\n");
}//end ui_train function

