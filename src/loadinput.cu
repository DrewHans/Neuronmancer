/*******************************************************************************************
 * Filename: neuralnetworkcore.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to load input into the network.
 *******************************************************************************************
 */

/*
 * loadInput
 * @params: neurons - a pointer to an array of double values (the input layer)
 * @params: n - the size of the input layer
 */
void loadInput(double* neurons, int n) {
    // generate random doubles for testing purposes
    for (int i = 0; i < n; i++) {
        neurons[i] = ((double) rand()) / ((double) rand());
    }
} //end loadInput method
