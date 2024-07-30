# Matrix_Based_Homomorphic_Encryption

This repository contains a Jupyter Notebook that implements orthogonal/inverted matrix-based homomorphic encryption for secure data processing and applies deep learning models for credit scoring using the UCI credit dataset.

## Overview

This project demonstrates the use of homomorphic encryption techniques to secure data and then applies both a vanilla deep learning model and a homomorphic encryption-based model to perform credit scoring. The notebook includes functions for data encryption, decryption, model training, and evaluation.

## Dataset
The dataset used in this project is the UCI Credit Card Dataset. The dataset should be in the same directory as the notebook or specify the correct path to the dataset in the notebook.

 ## Functions
# Encryption and Decryption
encryption_train(X, y): Encrypts the training data using orthogonal matrices.
decryption_train(X, y, U1, U2): Decrypts the training data.
encryption_test(X, U2): Encrypts the test data.
decryption_test(y_enc, U3): Decrypts the test data.

# Models
vanillaModel(x_data, y_data): Trains a vanilla deep learning model using TensorFlow.
homomorphicEncryptionModel(X_enc, y_enc, x_data, H_enc): Trains a deep learning model on encrypted data.

# Data Loading
dataload(cci_data, input_cols, output_cols): Loads and preprocesses the data, performing encryption on a subset of the data.

##Results
#Vanilla Model: Outputs the accuracy of the model on the test set.
Test Set Accuracy :.779

#Homomorphic Encryption Model: Outputs the accuracy and RMSE of the model on the encrypted data.
HE accuracy:  1.0
RMSE:  0.5970361989630348

#An accuracy is 1.0 indicates that there is no loss of insight during encryption
