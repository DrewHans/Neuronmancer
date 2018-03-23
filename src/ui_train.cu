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
    int numberOfLayers; // store the total number of layers in the network
    int numberOfNeuronsTotal; // store the total number of neurons in our neural network
    int numberOfWeightsTotal; // store the total number of weights in our neural network
    int* numberOfNeuronsPerLayer; // store the total number of neurons in each layer in our neural network in a 1d array of size numberOfLayers
    int* numberOfWeightsPerLayer; // store the total number of weights between each layer in our neural network in a 1d array of size numberOfLayers-1
    int* firstNeuronIndexPerLayer; // store the indexes of each layer's first neuron value
    int* firstWeightIndexPerLayer; // store the indexes of each layer's first weight value
    Activation* activationsPerLayer; // store the activation of each layer
    double* weights; // store the weight values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    double* biases; // store the biases values of our neural network in a 1d array of size weightSize (1d arrays are easy to work with in CUDA)
    int epochs; // store the number of epochs for training
    double learningRate; // store the rate that our network will learn

    int myPatience = 2; // stores the amount of patience I have for the user's nonsense

    printf("Lets train an artificial neural network!\n");
    printf("Searching ./nmModel for files...\n");

    readModel(&numberOfLayers, &numberOfNeuronsTotal, &numberOfWeightsTotal, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer,
            firstWeightIndexPerLayer, weights, biases, &learningRate, &epochs);

    printf("...files found!\n");

#ifdef DEBUG
    printf("epochs                 = %d\n", epochs);
    printf("learningRate           = %lf\n", learningRate);
    printf("numberOfLayers         = %d\n", numberOfLayers);
    printf("numberOfNeuronsTotal   = %d\n", numberOfNeuronsTotal); // remember, numberOfNeuronsTotal equals numberOfBiasesTotal
    printf("numberOfWeightsTotal   = %d\n", numberOfWeightsTotal);

    for(int i = 0; i < numberOfLayers; i++) {
        printf("numberOfNeuronsPerLayer[%d]  = %d\n", i, numberOfNeuronsPerLayer[i]);
        printf("numberOfWeightsPerLayer[%d]  = %d\n", i, numberOfWeightsPerLayer[i]);
        printf("firstNeuronIndexPerLayer[%d] = %d\n", i, firstNeuronIndexPerLayer[i]);
        printf("firstWeightIndexPerLayer[%d] = %d\n", i, firstWeightIndexPerLayer[i]);
        if(activationsPerLayer[i] == SIGMACT) {
            printf("activationsPerLayer[%d] = SIGMOID\n", i);
        } else if(activationsPerLayer[i] == RELUACT) {
            printf("activationsPerLayer[%d] = RELU\n", i);
        } else if(activationsPerLayer[i] == TANHACT) {
            printf("activationsPerLayer[%d] = TANH\n", i);
        } else {
            printf("activationsPerLayer[%d] = %d\n", i, activationsPerLayer[i]);
        }
    }

    printarray("biases", biases, numberOfNeuronsTotal);
    printarray("weights", weights, numberOfWeightsTotal);
#endif

    // get user input for running on CPU or GPU
    inputBuffer[0] = 'z'; // assign 'z' to enter loop
    while (inputBuffer[0] != 'h' || inputBuffer[0] != 'H' || inputBuffer[0] != 'd' || inputBuffer[0] != 'D') {
        // get the activation for layer i
        printf("Do you want to train on the host machine or GPU device?\nEnter h for host or d for device:\n~", i);
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        if (inputBuffer[0] != 'h' || inputBuffer[0] != 'H' || inputBuffer[0] != 'd' || inputBuffer[0] != 'D') {
            onInvalidInput(myPatience);
            myPatience--;
        }
    }
    myPatience = 2; // restore my patience

    if (inputBuffer[0] != 'h' || inputBuffer[0] != 'H') {
        printf("");
    } else if (inputBuffer[0] != 'd' || inputBuffer[0] != 'D') {

    }

} //end ui_train method
