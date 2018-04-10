/*******************************************************************************************
 * Filename: ui_create.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for creating a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_create method - user interface for creating a model */
void ui_create() {
    // declare helper variables for ui_create
    char inputBuffer[MAXINPUT]; // stores the user's input (gets recycled a lot)
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

    // declare variables needed to store important model values
    float* weights; // the weight values of the neural network (1d array of size numberOfWeightsTotal) (1d arrays are easy for CUDA to work with)
    float* biases; // the biases values of the neural network (1d array of size numberOfNeuronsTotal) (1d arrays are easy for CUDA to work with)

    printf("Before we begin, there are some things you should know. By design, all models created with Neuronmancer will have at least"
            " two layers - one input layer at the beginning of the network and one output layer at the end of the network. These layers"
            " are not optional, they are required for this program to function correctly. You can, however, specify the number of hidden"
            " layers and the number of neurons in each of those hidden layers... I will allow you at least that much freedom.\n"
            "Remember that you can press ctrl-c at anytime to abort...\n"
            "Press enter to continue...\n"
            "~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

    // get user input for the learningRate
    learningRate = -1.0; // assign -1.0 to enter loop
    while (learningRate <= 0.0) {
        // get the learningRate
        printf("Please enter a positive floating-point number greater than 0.0 for the learning rate:\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%f", &learningRate); // format and dump the user's input
        if (learningRate < 0.0) {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    // get user input for the epochs
    epochs = -1; // assign -1 to enter loop
    while (epochs <= 0) {
        // get the epochs
        printf("Pleae enter a positive whole number greater than 0 for the number of epochs:\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%u", &epochs); // format and dump the user's input
        if (epochs < 0) {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    printf("\nFor the following please enter a positive number with no spaces, commas, or decimal points.\n");

    // get user input for the numberOfLayers
    numberOfLayers = -1; // assign -1 to enter loop
    while (numberOfLayers < 0) {
        // get the number of layers in the ANN
        printf("How many hidden layers do you want this network to have? (note, 0 is the minimum)\n"
               "~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%u", &numberOfLayers); // format and dump the user's input
        if (numberOfLayers < 0) {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    numberOfLayers = numberOfLayers + 2; // account for the input and output layers
    myPatience = 2; // restore my patience

    // dynamically allocate memory for our variables
    numberOfNeuronsInLayer = (unsigned int *) malloc(numberOfLayers * sizeof(int));
    if (numberOfNeuronsInLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    numberOfWeightsInFrontOfLayer = (unsigned int *) malloc(numberOfLayers * sizeof(int));
    if (numberOfWeightsInFrontOfLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    indexOfFirstNeuronInLayer = (unsigned int *) malloc(numberOfLayers * sizeof(int));
    if (indexOfFirstNeuronInLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    indexOfFirstWeightInFrontOfLayer = (unsigned int *) malloc(numberOfLayers * sizeof(int));
    if (indexOfFirstWeightInFrontOfLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    printf("The input layer will have 784 neurons (one for each pixel in an MNIST sample).\n"
           "The output layer will have 10 neurons (one for each possible MNIST classification).\n");


    numberOfNeuronsInLayer[0] = 784; // set input layer size (one for each pixel value in an MNIST sample)
    numberOfNeuronsInLayer[numberOfLayers - 1] = 10; // set output layer size (one for each possible MNIST classification)

    // get user input for the number of neurons in each hidden layer
    for (int i = 1; i < numberOfLayers - 1; i++) {
        // get the number of neurons for layer i
        numberOfNeuronsInLayer[i] = -1; // assign -1 to enter loop
        while (numberOfNeuronsInLayer[i] < 1) {
            printf("How many neurons do you want hidden layer %d to have? (note, 1 is the minimum)\n"
                   "~", i);
            fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
            sscanf(inputBuffer, "%u", &(numberOfNeuronsInLayer[i])); // format and dump the user's input
            if (numberOfNeuronsInLayer[i] < 1) {
                onInvalidInput(myPatience);
                myPatience--;
            }
        }
        myPatience = 2; // restore my patience
    }

    printf("Alright, sit tight while I do some work...\n");

    // Calculate the number of neuron/weight values we need space for and also the first Neuron/Weight index for each layer
    indexOfFirstNeuronInLayer[0] = 0;  // input layer's first neuron starts at 0
    indexOfFirstWeightInFrontOfLayer[0] = 0; // input layer has no weights (put 0 but don't ever try to use)
    numberOfWeightsInFrontOfLayer[0] = 0;   // input layer has no weights
    numberOfNeuronsTotal = numberOfNeuronsInLayer[0]; // count the neurons in input layer
    numberOfWeightsTotal = numberOfWeightsInFrontOfLayer[0]; // count the weights in input layer (always 0)

    for (int i = 1; i < numberOfLayers; i++) {
        indexOfFirstNeuronInLayer[i] = numberOfNeuronsTotal; // set the index of first neuron in layer i
        indexOfFirstWeightInFrontOfLayer[i] = numberOfWeightsTotal; // set the index of first weight in layer i
        numberOfWeightsPerLayer[i] = numberOfNeuronsInLayer[i - 1] * numberOfNeuronsInLayer[i]; // calculate weights needed
        numberOfNeuronsTotal = numberOfNeuronsTotal + numberOfNeuronsInLayer[i]; // update total number of neurons
        numberOfWeightsTotal = numberOfWeightsTotal + numberOfWeightsInFrontOfLayer[i]; // update total number of weights
    }

    printf("- attempting to allocate memory for weights and biases...");

    // dynamically allocate memory to store the biases and weight values
    biases = (float*) malloc(numberOfNeuronsTotal * sizeof(float)); //malloc allocates a chunk of host memory
    if (biases == NULL) {
        onMallocError(numberOfNeuronsTotal * sizeof(float));
    }

    weights = (float*) malloc(numberOfWeightsTotal * sizeof(float)); //malloc allocates a chunk of host memory
    if (weights == NULL) {
        onMallocError(numberOfWeightsTotal * sizeof(float));
    }

    printf("allocation successful!\n"
           "- initializing biases to zero (this might take a while)...");
    
    initArrayToZeros(&biases, numberOfNeuronsTotal); // cleans up any garbage we may have picked up

    printf("biases initialized!\n"
           "- initializing weights to random floating-point values in range 0.0-1.0 inclusive (this also might take a while)...");
    
    initArrayToRandomFloats(&weights, numberOfWeightsTotal); // needs to be done, also cleans up garbage :)

    printf("weights initialized!\n"
           "- attempting to save model to disk...");

    // save the model to disk
    saveModel(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer, indexOfFirstNeuronInLayer,
            indexOfFirstWeightInFrontOfLayer, weights, biases, learningRate, epochs);
    printf("model saved!\n"
           "- freeing dynamically allocated host memory...");

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsInLayer);
    free(numberOfWeightsInFrontOfLayer);
    free(indexOfFirstNeuronInLayer);
    free(indexOfFirstWeightInFrontOfLayer);
    free(biases);
    free(weights);

    printf("memory freed!\n"
           "Press enter to return to the main menu:\n"
           "~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    printf("\n");
} //end ui_create function
