/*******************************************************************************************
 * Filename: main.cu
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This program creates a simple feed-forward artificial neural network
 *              and trains it on the CPU or GPU. The user will input (1) the number
 *              of layers (not including the input layer, which is required), (2) the
 *              number of neurons in each layer (including the input layer), and (3)
 *              whether to run on the CPU or GPU
 *******************************************************************************************
 */

#define MAXINPUT 32
#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "./neuronmancer.h"

#include "./helperfunctions.cu"
#include "./readmodel.cu"
#include "./savemodel.cu"

#include "./mnistfunctions.cu"

#include "./activationfunctions.cu"
#include "./combinationfunctions.cu"
#include "./costfunctions.cu"

#include "./feedforwardfunctions.cu"
#include "./backpropagationfunctions.cu"
#include "./updatebiasesfunctions.cu"
#include "./updateweightsfunctions.cu"

#include "./ui_create.cu"
#include "./ui_train.cu"
#include "./ui_evaluate.cu"

/* main - the program starts here
 * @params: argc - the argument count
 * @params: argv - the string of arguments
 */
int main(int argc, char * argv[]) {
    printf("Starting Neuronmancer...\n");
    char inputBuffer[MAXINPUT]; // store the user's input (gets recycled a lot)
    int myPatience = 2; // stores the amount of patience I have for the user's nonsense
    int loop = -1;

    while (loop != 0) {
        // get the number of layers in the ANN
        printf("What would you like to do?\nEnter 0 to quit, 1 to create a new model, 2 to train an existing model, or 3 to evaluate an existing model:\n~");
        fgets(inputBuffer, MAXINPUT, stdin); // read the user's input
        sscanf(inputBuffer, "%d", &loop); // format and dump the user's input
        if (loop < 0 || loop > 3) {
            onInvalidInput(myPatience);
            myPatience--;
        } else if (loop == 1) {
            printf("\nLets create an artificial neural network!\n");
            ui_create();
            myPatience = 2; // restore my patience
        } else if (loop == 2) {
            printf("\nLets train an artificial neural network!\n");
            ui_train();
            myPatience = 2; // restore my patience
        } else if (loop == 3) {
            printf("\nLets evaluate an artificial neural network!\n");
            ui_evaluate();
            myPatience = 2; // restore my patience
        }
    }

    printf("Neuronmancer will now end.\n");
} //end main function
