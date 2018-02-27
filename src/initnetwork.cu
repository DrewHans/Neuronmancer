/*******************************************************************************************
 * Filename: neuralnetworkcore.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to initialize the network.
 *******************************************************************************************
 */

/*
 * initNeurons method
 * @params: neurons - a pointer to an array of double values
 * @params: n - the size of array neurons
 */
void initNeurons(double* neurons, int n) {
    // set all neuron values to zero
    for (int i = 0; i < n; i++) {
        neurons[i] = 0;
#ifdef DEBUG
    printf("neurons[%d] = %f\n", i, neurons[i]);
#endif
    }
}//end initNeurons method

/*
 * initWeights method
 * @params: weights - a pointer to an array of double values
 * @params: n - the size of array weights
 */
void initWeights(double* weights, int n) {
    // generate random doubles in range [0, 1)
    for (int i = 0; i < n; i++) {
        weights[i] = ((double) rand()) / ((double) RAND_MAX);
#ifdef DEBUG
    printf("weights[%d] = %f\n", i, weights[i]);
#endif
    }
}//end initWeights method

