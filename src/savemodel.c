/*******************************************************************************************
 * Filename: savemodel.c
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the function for saving a model to disk.
 *******************************************************************************************
 */

#define VALUEDELIM ","
#define KEYVALUESLOCATION "./model/keyvalues.csv"
#define LAYERVALUESLOCATION "./model/layervalues.csv"
#define WEIGHTSFILELOCATION "./model/weights.csv"
#define BIASESFILELOCATION "./model/biases.csv"

void saveKeyValuesToDisk(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, int epochs, double learningRate) {
    FILE* thefile = fopen(KEYVALUESLOCATION, "w");
    if (thefile == NULL) {
        printf("Error, failed to open file.");
        exit(1);
    }

    fprintf(thefile, "%d", numberOfLayers);       // write int to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file
    fprintf(thefile, "%d", numberOfNeuronsTotal); // write int to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file
    fprintf(thefile, "%d", numberOfWeightsTotal); // write int to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file
    fprintf(thefile, "%d", epochs);               // write int to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file
    fprintf(thefile, "%lf", learningRate);        // write double to file
    fprintf(thefile, "%s", VALUEDELIM);           // write delimiter to file

    fclose(thefile);

}//end saveKeyValuesToDisk function

void saveLayerValuesToDisk(int numberOfLayers, int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, \
                           int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer) {
    FILE* thefile = fopen(LAYERVALUESLOCATION, "w");
    if (thefile == NULL) {
        printf("Error, failed to open file.");
        exit(1);
    }

    for (int i = 0; i < numberOfLayers; i++) {
        fprintf(thefile, "%d", numberOfNeuronsPerLayer[i]);  // write int to file
        fprintf(thefile, "%s", VALUEDELIM);                  // write delimiter to file
        fprintf(thefile, "%d", numberOfWeightsPerLayer[i]);  // write int to file
        fprintf(thefile, "%s", VALUEDELIM);                  // write delimiter to file
        fprintf(thefile, "%d", firstNeuronIndexPerLayer[i]); // write int to file
        fprintf(thefile, "%s", VALUEDELIM);                  // write delimiter to file
        fprintf(thefile, "%d", firstWeightIndexPerLayer[i]); // write int to file
        fprintf(thefile, "\n");                              // write newline to file (indicates new layer)
    }

    fclose(thefile);

}//end saveLayerValuesToDisk function

void saveWeightsToDisk(double* weights, int numberOfWeightsTotal) {
    FILE* thefile = fopen(WEIGHTSFILELOCATION, "w");
    if (thefile == NULL) {
        printf("Error, failed to open file.");
        exit(1);
    }

    for (int i = 0; i < numberOfWeightsTotal; i++) {
        fprintf(thefile, "%lf", weights[i]);  // write double to file
        fprintf(thefile, "%s", VALUEDELIM);   // write delimiter to file
    }

    fclose(thefile);

}//end saveWeightsToDisk function

void saveBiasesToDisk(double* biases, int numberOfBiasesTotal) {
    FILE* thefile = fopen(BIASESFILELOCATION, "w");
    if (thefile == NULL) {
        printf("Error, failed to open file.");
        exit(1);
    }

    for (int i = 0; i < numberOfBiasesTotal; i++) {
        fprintf(thefile, "%lf", biases[i]);   // write double to file
        fprintf(thefile, "%s", VALUEDELIM);   // write delimiter to file
    }

    fclose(thefile);

}//end saveBiasesToDisk function

void saveModel(int numberOfLayers, int numberOfNeuronsTotal, int numberOfWeightsTotal, int epochs, double learningRate, \
               int* numberOfNeuronsPerLayer, int* numberOfWeightsPerLayer, \
               int* firstNeuronIndexPerLayer, int* firstWeightIndexPerLayer, \
               double* weights ) {
    saveKeyValuesToDisk(numberOfLayers, numberOfNeuronsTotal, numberOfWeightsTotal, epochs, learningRate);
    saveLayerValuesToDisk(numberOfLayers, numberOfNeuronsPerLayer, numberOfWeightsPerLayer, firstNeuronIndexPerLayer, firstWeightIndexPerLayer);
    saveWeightsToDisk(weights, numberOfWeightsTotal);
    // saveBiasesToDisk(biases, numberOfBiasesTotal);
}//end saveModel function

