/*******************************************************************************************
 * Filename: ui_create.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the user interface for creating a Neuronmancer model.
 *******************************************************************************************
 */

/* ui_create method - user interface for creating a model */
void ui_create() {
    // declare variables needed to store the model information
    char inputBuffer[MAXINPUT]; // store the user's input (gets recycled a lot)
    int numberOfLayers; // store the total number of layers in the network
    int numberOfNeuronsTotal; // store the total number of neurons in our neural network
    int numberOfWeightsTotal; // store the total number of weights in our neural network
    int* numberOfNeuronsPerLayer; // store the total number of neurons in each layer in our neural network in a 1d array of size numberOfLayers
    int* numberOfWeightsPerLayer; // store the total number of weights between each layer in our neural network in a 1d array of size numberOfLayers-1
    int* firstNeuronIndexPerLayer; // store the indexes of each layer's first neuron value
    int* firstWeightIndexPerLayer; // store the indexes of each layer's first weight value
    double* weights; // store the weight values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    double* biases; // store the biases values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    int epochs; // store the number of epochs for training
    double learningRate; // store the rate that our network will learn

    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    printf("Before we begin, there are some things you should know. By design, all models created with Neuronmancer will have at least"
            " two layers - one input layer at the beginning of the network and one output layer at the end of the network. These layers"
            " are not optional, they are required for this program to function correctly. You can, however, specify the number of hidden"
            " layers and the number of neurons in each of those hidden layers... I will allow you at least that much freedom.\n");
    printf("Press enter to continue or ctrl-c to abort...");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input

    printf("\nFor the following please enter a positive number with no spaces, commas, or decimal points.\n");

    // get user input for the numberOfLayers
    numberOfLayers = -1; // assign -1 to enter loop
    while (numberOfLayers < 0) {
        // get the number of layers in the ANN
        printf("How many hidden layers do you want this network to have? (note, 0 is the minimum)\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%d", &numberOfLayers); // format and dump the user's input
        if (numberOfLayers < 0) {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    numberOfLayers = numberOfLayers + 2; // account for the input and output layers
    myPatience = 2; // restore my patience

    // dynamically allocate memory for our variables
    numberOfNeuronsPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    if (numberOfNeuronsPerLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    numberOfWeightsPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    if (numberOfWeightsPerLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    firstNeuronIndexPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    if (firstNeuronIndexPerLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    firstWeightIndexPerLayer = (int *) malloc(numberOfLayers * sizeof(int)); //malloc allocates a chunk of host memory
    if (firstWeightIndexPerLayer == NULL) {
        onMallocError(numberOfLayers * sizeof(int));
    }

    printf("The input layer will have 784 neurons (one for each pixel in an MNIST sample).\n");
    printf("The output layer will have 10 neurons (one for each MNIST sample class).\n");

    numberOfNeuronsPerLayer[0] = 784; // set input layer size
    numberOfNeuronsPerLayer[numberOfLayers - 1] = 10; // set output layer size

    // get user input for the number of neurons in each hidden layer
    for (int i = 1; i < numberOfLayers - 1; i++) {
        // get the number of neurons for layer i
        numberOfNeuronsPerLayer[i] = -1; // assign -1 to enter loop
        while (numberOfNeuronsPerLayer[i] < 1) {
            printf("How many neurons do you want hidden layer %d to have? (note, 1 is the minimum)\n~", i);
            fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
            sscanf(inputBuffer, "%d", &(numberOfNeuronsPerLayer[i])); // format and dump the user's input
            if (numberOfNeuronsPerLayer[i] < 1) {
                onInvalidInput(myPatience);
                myPatience--;
            }
        }
        myPatience = 2; // restore my patience
    }

    printf("Alright, sit tight while I do some calculations...\n");

    // Calculate the number of neuron/weight values we need space for and also the first Neuron/Weight index for each layer
    firstNeuronIndexPerLayer[0] = 0;  // input layer's first neuron starts at 0
    firstWeightIndexPerLayer[0] = -1; // input layer has no weights, put -1 just for fun
    numberOfWeightsPerLayer[0] = 0;   // input layer has no weights
    numberOfNeuronsTotal = numberOfNeuronsPerLayer[0]; // count the neurons in input layer
    numberOfWeightsTotal = numberOfWeightsPerLayer[0]; // count the weights in input layer (always 0)
    for (int i = 1; i < numberOfLayers; i++) {
        firstNeuronIndexPerLayer[i] = numberOfNeuronsTotal; // set the index of first neuron in layer i
        firstWeightIndexPerLayer[i] = numberOfWeightsTotal; // set the index of first weight in layer i
        numberOfWeightsPerLayer[i] = numberOfNeuronsPerLayer[i - 1] * numberOfNeuronsPerLayer[i]; // calculate weights needed
        numberOfNeuronsTotal = numberOfNeuronsTotal + numberOfNeuronsPerLayer[i]; // update total number of neurons
        numberOfWeightsTotal = numberOfWeightsTotal + numberOfWeightsPerLayer[i]; // update total number of weights
    }

    printf("...attempting to allocate memory for weights and biases...\n");

    // dynamically allocate memory to store the biases and weight values
    biases = (double*) malloc(numberOfNeuronsTotal * sizeof(double)); //malloc allocates a chunk of host memory
    if (biases == NULL) {
        onMallocError(numberOfNeuronsTotal * sizeof(double));
    }

    weights = (double*) malloc(numberOfWeightsTotal * sizeof(double)); //malloc allocates a chunk of host memory
    if (weights == NULL) {
        onMallocError(numberOfWeightsTotal * sizeof(double));
    }

    printf("...allocation successful!\nBeginning value initialization (this might take a while)...\n");

    printf("...initializing biases to zero...\n");
    initArrayToZeros(biases, numberOfNeuronsTotal); // cleans up any garbage we may have picked up

    printf("...initializing weights to random double floating-point values in range 0.0-1.0 (inclusive)...\n");
    initArrayToRandomDoubles(weights, numberOfWeightsTotal);

    printf("...initialization successful!\n");

    printf("Now you need to decide the learning rate and number of epochs.\n");

    // get user input for the learningRate
    learningRate = -1.0; // assign -1.0 to enter loop
    while (learningRate <= 0.0) {
        // get the learning rate
        printf("Please enter a positive floating-point number greater than 0.0 for the learning rate:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%lf", &learningRate); // format and dump the user's input
        if (learningRate < 0.0) {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    // get user input for the epochs
    epochs = -1.0; // assign -1.0 to enter loop
    while (epochs <= 0) {
        // get the learning rate
        printf("Pleae enter a positive whole number greater than 0 for the number of epochs:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%d", &epochs); // format and dump the user's input
        if (epochs < 0) {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    // save the model to disk
    printf("Saving model to disk...");
    saveModel(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
            firstWeightIndexPerLayer, weights, biases, learningRate, epochs);
    printf("Model saved!\n");

    // free the chunks of host memory that were dynamically allocated by malloc
    free(numberOfNeuronsPerLayer);
    free(numberOfWeightsPerLayer);
    free(firstNeuronIndexPerLayer);
    free(firstWeightIndexPerLayer);
    free(biases);
    free(weights);

    printf("Press enter to return to the main menu:\n~");
    fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
    printf("\n");
} //end ui_create method
