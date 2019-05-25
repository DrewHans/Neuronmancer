/*******************************************************************************
 * Filename: main.c
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the main function and other important
 *              functions needed to evaluate a neural network.
 *******************************************************************************
 */

#include "main.h"

// performs evaluation on a network
void evaluate(InputLayer *il, HiddenLayer *hl, OutputLayer *ol)
{
    // initialize confusion matrix to hold the
    // predicted vs actual classifications
    int *confusionMatrix;
    confusionMatrix = (int *)malloc(100 * sizeof(int));
    if (confusionMatrix == NULL)
    {
        printf("Abort! Could not allocate memory for confusion matrix!\n");
        exit(1);
    }

    // clear out any garbage we picked up before evaluating
    for (int i = 0; i < 100; i++)
    {
        confusionMatrix[i] = 0;
    }

    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGES_PATH);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABELS_PATH);

    printf("\n--- beginning evaluation ---\n");

    // for each MNIST sample in the test set
    for (int sample = 0; sample < MNIST_TESTING_SET_SIZE; sample++)
    {
        // read the next sample image and label
        MNIST_Image image = getImage(imageFile);
        MNIST_Label label = getLabel(labelFile);

        int prediction = feedNetwork(il, hl, ol, &image);

        // update confusion matrix
        confusionMatrix[((int)label) * 10 + prediction] += 1;

        if ((sample + 1) % 1000 == 0)
        {
            printf("   => sample %d of %5d complete\n",
                   sample + 1, MNIST_TESTING_SET_SIZE);
        }
    }

    printf("--- evaluation complete ---\n\n");

    fclose(imageFile);
    fclose(labelFile);

    printConfusionMatrix(confusionMatrix);
    free(confusionMatrix);
}

// prints confusion matrix, accuracy, and misclassification rate
void printConfusionMatrix(const int *confusionMatrix)
{
    // print out a "pretty" confusion matrix table
    printf("            ");
    for (int i = 0; i < 10; i++)
    {
        printf("+-PREDICTED-+");
    }
    printf("\n");

    printf("            ");
    for (int i = 0; i < 10; i++)
    {
        printf("|     %d     |", i);
    }
    printf("\n");

    printf("+-----------");
    for (int i = 0; i < 10; i++)
    {
        printf("+-----------+");
    }
    printf("\n");

    for (int i = 0; i < 10; i++)
    {
        printf("| ACTUAL %d  ", i);
        for (int j = 0; j < 10; j++)
        {
            printf("|  %5d    ", confusionMatrix[i * 10 + j]);
        }
        printf("|\n+");

        for (int i = 0; i < 11; i++)
        {
            printf("-----------+");
        }
        printf("\n");
    }

    // get the number of correct predictions from the confusion matrix
    int truePositives = 0;
    for (int i = 0; i < 10; i++)
    {
        truePositives += confusionMatrix[i * 10 + i];
    }
    float accuracy = (float)truePositives / MNIST_TESTING_SET_SIZE;
    float misclassificationRate = 1.0 - accuracy;

    printf("Accuracy = %3.2f%%\nMisclassificationRate = %3.2f%%\n\n",
           accuracy * 100,
           misclassificationRate * 100);
}

// prints out the information stored in a TimeCollector struct
void printTimeCollector(TimeCollector *tc)
{
    printf("Average time spent feedforwarding:  %4.2f seconds\n"
           "Average time spent backpropagating: %4.2f seconds\n"
           "Average time spent updating:        %4.2f seconds\n"
           "Average time spent epoching:        %4.2f seconds\n",
           tc->averageFeedforwardTime,
           tc->averageBackpropagationTime,
           tc->averageUpdateTime,
           tc->averageEpochTime);
}

// prints out insults when the user screws up (silly humans)
// myPatience - the current state of my patience, represented as an int
void onInvalidInput(const int myPatience)
{
    if (myPatience == 2)
    {
        printf("Looks like you entered an illegal value... "
               "you're testing my patience, try again!\n\n");
    }
    else if (myPatience == 1)
    {
        printf("That's the second time you've entered an illegal value... "
               "do you think this is funny? Try again!\n\n");
    }
    else if (myPatience == 0)
    {
        printf("Sigh... you just can't do anything right, can you?\n\n");
    }
    else
    {
        printf("Look dude, I've got all day. "
               "If you wanna keep wasting your time then that's fine by me. "
               "You know what you're supposed to do.\n\n");
    }
}

// main - the program starts here
int main(int argc, const char *argv[])
{
    printf("Starting Neuronmancer 3.0...\n");
    time_t programStartTime;
    time_t programEndTime;
    time_t trainStartTime;
    time_t trainEndTime;
    time_t evalStartTime;
    time_t evalEndTime;
    double executionTime = 0.0;
    programStartTime = time(NULL);

    char buffer[MAXINPUT];
    int myPatience = 2;
    int loop = -1;
    InputLayer il;
    HiddenLayer hl;
    OutputLayer ol;

    printf("Initializing neural network with the following parameters:\n"
           "--- input-layer size : %d\n"
           "--- hidden-layer size : %d\n"
           "--- output-layer size : %d\n"
           "--- learning rate : %f\n"
           "--- epochs : %d\n"
           "\n"
           "Note, these parameters can be changed in the main.h file.\n\n",
           INPUT_LAYER_SIZE,
           HIDDEN_LAYER_SIZE,
           OUTPUT_LAYER_SIZE,
           LEARNING_RATE,
           EPOCHS);

    // initialize the neural network

    initNetwork(&hl, &ol);

    while (loop != 0)
    {
        printf("What would you like to do?\n"
               "Enter 0 to quit, "
               "1 to train on host, "
               "2 to train on GPU device, or "
               "3 to evaluate:\n"
               "~");
        fgets(buffer, MAXINPUT, stdin); // load stdin into buffer
        sscanf(buffer, "%d", &loop);    // load buffer into loop

        if (loop < 0 || loop > 3)
        {
            onInvalidInput(myPatience);
            myPatience--;
        }
        else if (loop == 1)
        {
            // TRAIN ON HOST
            myPatience = 2; // restore my patience
            trainStartTime = time(NULL);

            train(&il, &hl, &ol);

            trainEndTime = time(NULL);
            executionTime = difftime(trainEndTime, trainStartTime);

            printf("Training on host completed in %.2lf seconds!\n\n",
                   executionTime);
        }
        else if (loop == 2)
        {
            // TRAIN ON GPU DEVICE
            myPatience = 2; // restore my patience
            trainStartTime = time(NULL);

            cuda_train(&il, &hl, &ol); // comment out if you don't want CUDA

            trainEndTime = time(NULL);
            executionTime = difftime(trainEndTime, trainStartTime);
            printf("Training on GPU device completed in %.2lf seconds!\n\n",
                   executionTime);
        }
        else if (loop == 3)
        {
            // EVALUATION (ALWAYS USE HOST)
            myPatience = 2; // restore my patience
            evalStartTime = time(NULL);

            evaluate(&il, &hl, &ol);

            evalEndTime = time(NULL);
            executionTime = difftime(evalEndTime, evalStartTime);
            printf("Evaluation completed in %.2lf seconds!\n\n",
                   executionTime);
        }
    }

    printf("Neuronmancer will now end.\n");
    programEndTime = time(NULL);
    executionTime = difftime(programEndTime, programStartTime);
    printf("===> Total time spent in Neuronmancer: %.2f seconds\n\n",
           executionTime);
}
