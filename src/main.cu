/********************************************************************************
 * Filename: main.c
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the main function and other important
 *              functions needed to evaluate a neural network.
 ********************************************************************************
 */

#include "main.h"

/**
 * evaluate - performs evaluation on a network
 * @params InputLayer* il - pointer to an InputLayer struct
 * @params HiddenLayer* hl - pointer to a HiddenLayer struct
 * @params OutputLayer* ol - pointer to an OutputLayer struct
 */
void evaluate(InputLayer* il, HiddenLayer* hl, OutputLayer* ol) {
    // initialize confusion matrix to hold the predicted vs actual classifications
    int* confusionMatrix;
    confusionMatrix = (int*) malloc(100 * sizeof(int));
    if (confusionMatrix == NULL) {
        printf("Abort! Could not allocate memory for confusion matrix!\n");
        exit(1);
    }

    // clear out any garbage we picked up before evaluating
    for (int i = 0; i < 100; i++) {
        confusionMatrix[i] = 0;
    }

    // open MNIST files
    FILE* imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGES_PATH);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABELS_PATH);

    printf("\n--- beginning evaluation ---\n");

    // for each MNIST sample in the test set
    for (int sample = 0; sample < MNIST_TESTING_SET_SIZE; sample++) {
        // read the next sample image and label
        MNIST_Image image = getImage(imageFile);
        MNIST_Label label = getLabel(labelFile);

        int prediction = feedNetwork(il, hl, ol, &image);

        // update confusion matrix
        confusionMatrix[((int) label) * 10 + prediction] += 1;

        if ((sample + 1) % 1000 == 0) {
            printf("   => sample %d of %5d complete\n", sample + 1, MNIST_TESTING_SET_SIZE);
        }
    }

    printf("--- evaluation complete ---\n\n");

    fclose(imageFile);
    fclose(labelFile);

    // print the confusion matrix
    printConfusionMatrix(confusionMatrix);

    // free dynamically allocated memory
    free(confusionMatrix);

} //end evaluate function

/*
 * printConfusionMatrix - prints out the confusion matrix, accuracy, and misclassification rate
 * @params: cm - the int pointer to the array of confusion matrix values
 * @params: n - the int size of confusion matrix rows and cols (should be equal to the number of MNIST classifications)
 */
void printConfusionMatrix(const int* cm) {
    // print out a "pretty" confusion matrix table
    printf("            +-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+-PREDICTED-+\n"
            "            |     0     |     1     |     2     |     3     |     4     |     5     |     6     |     7     |     8     |     9     |\n"
            "+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+\n");
    for (int i = 0; i < 10; i++) {
        printf("| ACTUAL %d  ", i);
        for (int j = 0; j < 10; j++) {
            printf("|  %5d    ", cm[i * 10 + j]);
        }
        printf("|\n+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+\n");
    }

    // get the number of correct predictions from the confusion matrix
    int truePositives = 0;
    for (int i = 0; i < 10; i++) {
        truePositives += cm[i * 10 + i];
    }
    float accuracy = (float) truePositives / MNIST_TESTING_SET_SIZE;
    float misclassificationRate = 1.0 - accuracy;

    // print out the accuracy and misclassification rate
    printf("Accuracy = %3.2f%%\nMisclassificationRate = %3.2f%%\n\n", accuracy * 100, misclassificationRate * 100);

} //end printConfusionMatrix function

/*
 * printTimeCollector - prints out the information stored in a TimeCollector struct
 * @params: TimeCollector* tc - pointer to a TimeCollector struct
 */
void printTimeCollector(TimeCollector* tc) {
    printf("Average time spent feedforwarding:  %4.2f seconds\n"
            "Average time spent backpropagating: %4.2f seconds\n"
            "Average time spent updating:        %4.2f seconds\n"
            "Average time spent epoching:        %4.2f seconds\n", 
            tc->averageFeedforwardTime, tc->averageBackpropagationTime, 
            tc->averageUpdateTime, tc->averageEpochTime);
} //end printTimeCollector function

/*
 * onInvalidInput - prints out insults when the user screws up (silly humans)
 * @params: myPatience - the current state of my patience, represented as an int
 */
void onInvalidInput(const int myPatience) {
    if (myPatience == 2) {
        printf("Looks like you entered an illegal value... you're testing my patience, try again!\n\n");
    } else if (myPatience == 1) {
        printf("That's the second time you've entered an illegal value... do you think this is funny? Try again!\n\n");
    } else if (myPatience == 0) {
        printf("Sigh... you just can't do anything right, can you?\n\n");
    } else {
        printf("Look dude, I've got all day. If you wanna keep wasting your time then that's fine by me. "
                "You know what you're supposed to do.\n\n");
    }
} //end onInvalidInput function

/*
 * main - the program starts here
 * @params: argc - the argument count
 * @params: argv - the string of arguments
 */
int main(int argc, const char* argv[]) {
    printf("Starting Neuronmancer 3.0...\n");
    time_t programStartTime, programEndTime, trainStartTime, trainEndTime, evalStartTime, evalEndTime;
    double executionTime = 0.0;
    programStartTime = time(NULL);

    char buffer[MAXINPUT]; // store the user's input (gets recycled a lot)
    int myPatience = 2; // stores the amount of patience I have for the user's nonsense
    int loop = -1;

    printf("Initializing neural network with the following parameters:\n"
            "--- input-layer size : %d\n"
            "--- hidden-layer size : %d\n"
            "--- output-layer size : %d\n"
            "--- learning rate : %f\n"
            "--- epochs : %d\n"
            "\n", INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, LEARNING_RATE, EPOCHS);

    printf("Note, these parameters can be changed in the main.h file.\n\n");

    // initialize the neural network
    InputLayer il;
    HiddenLayer hl;
    OutputLayer ol;
    initNetwork(&hl, &ol);

    while (loop != 0) {
        printf("What would you like to do?\n"
                "Enter 0 to quit, 1 to train on host, 2 to train on GPU device, or 3 to evaluate:\n"
                "~");
        fgets(buffer, MAXINPUT, stdin); // read the user's input from stdin into inputBuffer
        sscanf(buffer, "%d", &loop); // format and dump the user's input into loop

        if (loop < 0 || loop > 3) {
            onInvalidInput(myPatience);
            myPatience--;
        } else if (loop == 1) {
            myPatience = 2; // restore my patience

            // TRAIN ON HOST
            trainStartTime = time(NULL);

            train(&il, &hl, &ol);

            trainEndTime = time(NULL);
            executionTime = difftime(trainEndTime, trainStartTime);
            printf("Training on host completed in %.2lf seconds!\n\n", executionTime);

        } else if (loop == 2) {
            myPatience = 2; // restore my patience

            // TRAIN ON GPU DEVICE
            trainStartTime = time(NULL);

            cuda_train(&il, &hl, &ol); // comment this line out if you don't want CUDA

            trainEndTime = time(NULL);
            executionTime = difftime(trainEndTime, trainStartTime);
            printf("Training on GPU device completed in %.2lf seconds!\n\n", executionTime);

        } else if (loop == 3) {
            myPatience = 2; // restore my patience

            // EVALUATION (ALWAYS USE HOST)
            evalStartTime = time(NULL);

            evaluate(&il, &hl, &ol);

            evalEndTime = time(NULL);
            executionTime = difftime(evalEndTime, evalStartTime);
            printf("Evaluation completed in %.2lf seconds!\n\n", executionTime);
        }
    } //end while loop

    printf("Neuronmancer will now end.\n");
    programEndTime = time(NULL);
    executionTime = difftime(programEndTime, programStartTime);
    printf("===> Total time spent in Neuronmancer: %.2f seconds\n\n", executionTime);
} //end main function
