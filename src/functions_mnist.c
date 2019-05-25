/******************************************************************************
 * Filename: functions_mnist.c
 * Author: Drew Hans (github.com/drewhans555)
 * Description: This file contains the functions needed to get the images and
 *              labels from the MNIST database files.
 ******************************************************************************
 */

#include "main.h"

// returns the next image in the MNIST image file
MNIST_Image getImage(FILE *imageFile)
{
    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result != 1)
    {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;
}

// returns the next label in the MNIST label file
MNIST_Label getLabel(FILE *labelFile)
{
    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result != 1)
    {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

// returns the file pointer pointing to the first MNIST image in the file
FILE *openMNISTImageFile(char *filePath)
{
    // open the MNIST image file
    FILE *imageFile;
    imageFile = fopen(filePath, "rb");
    if (imageFile == NULL)
    {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n", filePath);
        exit(1);
    }

    // read the header to move the pointer to the position of the first image
    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}

// returns the file pointer pointing to the first MNIST label in the file
FILE *openMNISTLabelFile(char *filePath)
{
    // open the MNIST label file
    FILE *labelFile;
    labelFile = fopen(filePath, "rb");
    if (labelFile == NULL)
    {
        printf("Abort! Could not find MNIST LABEL file: %s\n", filePath);
        exit(1);
    }

    // read the header to move the pointer to the position of the first label
    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}

// reads an MNIST image file header from the imageFile
void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh)
{
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
}

// reads an MNIST label file header from the imageFile
void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh)
{
    lfh->magicNumber = 0;
    lfh->maxImages = 0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = reverseBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = reverseBytes(lfh->maxImages);
}

// reverses the byte order of 32-bit numbers
uint32_t reverseBytes(uint32_t n)
{
    uint32_t b0, b1, b2, b3;

    b0 = (n & 0x000000ff) << 24u;
    b1 = (n & 0x0000ff00) << 8u;
    b2 = (n & 0x00ff0000) >> 8u;
    b3 = (n & 0xff000000) >> 24u;

    return (b0 | b1 | b2 | b3);
}
