/********************************************************************************
 * Filename: functions_mnist.c
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to get the images and
 *              labels from the MNIST database files.
 ********************************************************************************
 */

#include "main.h"

/*
 * getImage - returns the next image in the MNIST image file
 * @params File* imageFile - the file pointer pointing to the MNIST image we want
 */
MNIST_Image getImage(FILE* imageFile) {

    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result != 1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;

} //end getImage function

/*
 * getLabel - returns the next label in the MNIST label file
 * @params File* labelFile - the file pointer pointing to the MNIST label we want
 */
MNIST_Label getLabel(FILE* labelFile) {

    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result != 1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;

} //end getLabel function

/*
 * openMNISTImageFile - returns the file pointer pointing to the first MNIST image in the file
 * @params char* filePath - the path to the MNIST image file
 */
FILE* openMNISTImageFile(char* filePath) {

    // open the MNIST image file
    FILE* imageFile;
    imageFile = fopen(filePath, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n", filePath);
        exit(1);
    }

    // read the header to move the pointer to the position of the first image
    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;

} //end openMNISTImageFile function

/*
 * openMNISTLabelFile - returns the file pointer pointing to the first MNIST label in the file
 * @params char* filePath - the path to the MNIST label file
 */
FILE* openMNISTLabelFile(char* filePath) {

    // open the MNIST label file
    FILE* labelFile;
    labelFile = fopen(filePath, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n", filePath);
        exit(1);
    }

    // read the header to move the pointer to the position of the first label
    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;

} //end openMNISTLabelFile function

/*
 * readImageFileHeader - reads an MNIST image file header from the imageFile
 * @params FILE* imageFile - pointer to a file containing MNIST images
 * @params MNIST_ImageFileHeader* ifh - pointer to an MNIST_ImageFileHeader struct
 */
void readImageFileHeader(FILE* imageFile, MNIST_ImageFileHeader* ifh) {

    ifh->magicNumber = 0;
    ifh->maxImages = 0;
    ifh->imgWidth = 0;
    ifh->imgHeight = 0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = reverseBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = reverseBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = reverseBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = reverseBytes(ifh->imgHeight);

} //end readImageFileHeader function

/*
 * readLabelFileHeader - reads an MNIST label file header from the imageFile
 * @params FILE* imageFile - pointer to a file containing MNIST labels
 * @params MNIST_LabelFileHeader* ifh - pointer to an MNIST_LabelFileHeader struct
 */
void readLabelFileHeader(FILE* imageFile, MNIST_LabelFileHeader* lfh) {

    lfh->magicNumber = 0;
    lfh->maxImages = 0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = reverseBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = reverseBytes(lfh->maxImages);

} //end readLabelFileHeader function

/*
 * reverseBytes - reverses the byte order of 32-bit numbers
 * @params uint32_t n - some 32-bit number
 */
uint32_t reverseBytes(uint32_t n) {

    uint32_t b0, b1, b2, b3;

    b0 = (n & 0x000000ff) << 24u;
    b1 = (n & 0x0000ff00) << 8u;
    b2 = (n & 0x00ff0000) >> 8u;
    b3 = (n & 0xff000000) >> 24u;

    return (b0 | b1 | b2 | b3);

} //end reverseBytes function
