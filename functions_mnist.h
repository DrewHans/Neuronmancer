/********************************************************************************
 * Filename: functions_mnist.h
 * Author: Drew Hans (github.com/drewhans555)
 * Description: functions_mnist.c's header file - contains function prototypes.
 ********************************************************************************
 */

#ifndef FUNCTIONS_MNIST_H
#define FUNCTIONS_MNIST_H

/////////////////////////////////////////////////////////////////////////////////
// define function prototypes for functions_mnist.c //////////////////////////////

MNIST_Image getImage(FILE* imageFile);
MNIST_Label getLabel(FILE* labelFile);
FILE* openMNISTImageFile(char* filePath);
FILE* openMNISTLabelFile(char* filePath);
void readImageFileHeader(FILE* imageFile, MNIST_ImageFileHeader* ifh);
void readLabelFileHeader(FILE* imageFile, MNIST_LabelFileHeader* lfh);
uint32_t reverseBytes(uint32_t n);

/////////////////////////////////////////////////////////////////////////////////

#endif /* FUNCTIONS_MNIST_H */
