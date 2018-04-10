/*******************************************************************************************
 * Filename: model_save.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions for saving a model to disk.
 *******************************************************************************************
 */

/*
 * saveBiasesToDisk - writes model's bias values to a file on disk
 * @params: biases - pointer to an array of float values
 * @params: numberOfBiasesTotal - int equal to numberOfNeuronsTotal
 */
void saveBiasesToDisk(const float* biases, const unsigned int numberOfBiasesTotal) {
    FILE* thefile = fopen(BIASESFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (BIASESFILELOCATION);
    }

    for (int i = 0; i < numberOfBiasesTotal; i++) {
        fprintf(thefile, "%f%s", biases[i], VALUEDELIM);   // write float value and VALUEDELIM to file
    }

    fprintf(thefile, "\n"); // write newline to file (needed for reading file with getline)
    fclose(thefile); // close the file once we're done with it
} //end saveBiasesToDisk function

/*
 * saveEpochsToDisk - writes model's epoch value to a file on disk
 * @params: epochs - the int number of training cycles (in one cycle every sample in the data set is fed in and then errors "backpropagate")
 */
void saveEpochsToDisk(const unsigned int epochs) {
    FILE* thefile = fopen(EPOCHSFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (EPOCHSFILELOCATION);
    }

    fprintf(thefile, "%u\n", epochs); // write int value and newline to file

    fclose(thefile); // close the file once we're done with it
} //end saveEpochsToDisk function

/*
 * saveLearningRateToDisk - writes model's learning rate value to a file on disk
 * @params: learningRate - the float rate at which we want our network to make adjustments to the weights
 */
void saveLearningRateToDisk(const float learningRate) {
    FILE* thefile = fopen(LEARNINGRATEFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (LEARNINGRATEFILELOCATION);
    }

    fprintf(thefile, "%f\n", learningRate); // write float value and newline to file

    fclose(thefile); // close the file once we're done with it
} //end saveLearningRateToDisk function

/*
 * saveWeightsToDisk - writes model's weight values to a file on disk
 * @params: weights - pointer to an array of float values
 * @params: numberOfWeightsTotal - the int total number of weights in the model
 */
void saveWeightsToDisk(const float* weights, const unsigned int numberOfWeightsTotal) {
    FILE* thefile = fopen(WEIGHTSFILELOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (WEIGHTSFILELOCATION);
    }

    for (int i = 0; i < numberOfWeightsTotal; i++) {
        fprintf(thefile, "%f%s", weights[i], VALUEDELIM);  // write float value and VALUEDELIM to file
    }

    fprintf(thefile, "%u\n", epochs); // write int value and newline to file

    fclose(thefile); // close the file once we're done with it
} //end saveWeightsToDisk function

/*
 * saveModelValuesToDisk - writes values describing the model's structure to a file on disk
 * @params: numberOfLayers - the int number of layers in the model
 * @params: numberOfNeuronsTotal - the int total number of neurons in the model
 * @params: numberOfWeightsTotal - the int total number of weights in the model
 * @params: numberOfNeuronsInLayer - pointer to an array of int values (the number of neurons in each layer in the model)
 * @params: numberOfWeightsInFrontOfLayer - pointer to an array of int values (the number of weights in front of each layer in the model)
 * @params: indexOfFirstNeuronInLayer - pointer to an array of int values (the indexes of the first neuron in each layer)
 * @params: indexOfFirstWeightInFrontOfLayer - pointer to an array of int values (the indexes of the first weight in front of each layer)
 */
void saveModelValuesToDisk(const unsigned int numberOfLayers, const unsigned int numberOfNeuronsTotal, const unsigned int numberOfWeightsTotal, 
                           const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
                           const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer) {
    FILE* thefile = fopen(MODELVALUESLOCATION, "w");
    if (thefile == NULL) {
        onFileOpenError (MODELVALUESLOCATION);
    }

    // write structure values for model construction to file first
    fprintf(thefile, "%u%s%u%s%u\n", numberOfLayers, VALUEDELIM, numberOfNeuronsTotal, VALUEDELIM, numberOfWeightsTotal);

    // write structure values for each layer in model to file second
    for (int i = 0; i < numberOfLayers; i++) {
        fprintf(thefile, "%u%s%u%s%u%s%u\n", numberOfNeuronsInLayer[i], VALUEDELIM, numberOfWeightsInFrontOfLayer[i], VALUEDELIM, indexOfFirstNeuronInLayer[i], VALUEDELIM, indexOfFirstWeightInFrontOfLayer[i]);
    }

    fclose(thefile); // close the file once we're done with it
} //end saveModelValuesToDisk function

/*
 * saveModel
 * @params: numberOfLayers - the int number of layers in the model
 * @params: numberOfNeuronsTotal - the int total number of neurons in the model
 * @params: numberOfWeightsTotal - the int total number of weights in the model
 * @params: numberOfNeuronsInLayer - pointer to an array of int values (the number of neurons in each layer in the model)
 * @params: numberOfWeightsInFrontOfLayer - pointer to an array of int values (the number of weights in front of each layer in the model)
 * @params: indexOfFirstNeuronInLayer - pointer to an array of int values (the indexes of the first neuron in each layer)
 * @params: indexOfFirstWeightInFrontOfLayer - pointer to an array of int values (the indexes of the first weight in front of each layer)
 * @params: weights - pointer to an array of float values
 * @params: biases - pointer to an array of float values
 * @params: learningRate - the rate at which we want our network to make adjustments to the weights
 * @params: epochs - the number of training cycles (in one cycle every sample in the data set is fed in and then errors "backpropagate")
 */
void saveModel(const unsigned int numberOfLayers, const unsigned int numberOfNeuronsTotal, const unsigned int numberOfWeightsTotal, 
               const unsigned int* numberOfNeuronsInLayer, const unsigned int* numberOfWeightsInFrontOfLayer,
               const unsigned int* indexOfFirstNeuronInLayer, const unsigned int* indexOfFirstWeightInFrontOfLayer, 
               const float* weights, const float* biases, const float learningRate, const unsigned int epochs) {
    // if directory doesn't exist, make directory to store the model
    if (stat(MODELDIRECTORY, &st) == -1) {
        mkdir(MODELDIRECTORY, 0700);
    }

    // each of the following function calls saves to a different file on disk
    saveModelValuesToDisk(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, 
                          numberOfNeuronsInLayer, numberOfWeightsInFrontOfLayer, 
                          indexOfFirstNeuronInLayer, indexOfFirstWeightInFrontOfLayer);
    saveWeightsToDisk(weights, numberOfWeightsTotal);
    saveBiasesToDisk(biases, numberOfNeuronsTotal); // num biases total is equal to num neurons total
    saveEpochsToDisk(epochs);
    saveLearningRateToDisk(learningRate);
} //end saveModel function
