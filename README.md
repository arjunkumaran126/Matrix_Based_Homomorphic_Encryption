# Matrix_Based_Homomorphic_Encryption

This repository contains a Jupyter Notebook that implements orthogonal/inverted matrix-based homomorphic encryption for secure data processing and applies deep learning models for credit scoring using the UCI credit dataset.

## Overview

This project demonstrates the use of homomorphic encryption techniques to secure data and then applies both a vanilla deep learning model and a homomorphic encryption-based model to perform credit scoring. The notebook includes functions for data encryption, decryption, model training, and evaluation.

## Dataset

The dataset used in this project is the UCI Credit Card Dataset. 

## Functions

### Encryption and Decryption

- **`encryption_train(X, y)`**: Encrypts the training data using orthogonal matrices.
  - **Homomorphic Encryption (HME)**: This function generates two orthogonal matrices, `U1` and `U2`, and uses them to encrypt the input data `X` and the target data `y`. The encryption is performed by matrix multiplication: `X_enc = U1.dot(X).dot(U2)` and `y_enc = U1.dot(y)`. The function returns the encrypted data and the orthogonal matrices used for encryption.

- **`decryption_train(X, y, U1, U2)`**: Decrypts the training data.
  - **HME**: This function takes the encrypted data `X` and `y`, along with the orthogonal matrices `U1` and `U2`, and decrypts the data by performing inverse operations: `X_dec = U1.T.dot(X).dot(np.linalg.inv(U2))` and `y_dec = U1.T.dot(y)`. The function returns the decrypted data.

- **`encryption_test(X, U2)`**: Encrypts the test data.
  - **HME**: This function generates a new orthogonal matrix `U3` and uses it, along with `U2`, to encrypt the input test data `X`. The encryption is performed by matrix multiplication: `X_enc = U3.dot(X).dot(np.linalg.inv(U2))`. The function returns the encrypted data and the orthogonal matrix `U3`.

- **`decryption_test(y_enc, U3)`**: Decrypts the test data.
  - **HME**: This function takes the encrypted target data `y_enc` and the orthogonal matrix `U3`, and decrypts the data by performing the inverse operation: `y_dec = np.linalg.inv(U3).dot(y_enc)`. The function returns the decrypted data.

### Models

- **`vanillaModel(x_data, y_data)`**: Trains a vanilla deep learning model using TensorFlow.
  - This function defines and trains a deep neural network (DNN) classifier using the input features `x_data` and the target values `y_data`. The model architecture includes multiple hidden layers with specified units. The training process includes early stopping based on the loss metric.

- **`homomorphicEncryptionModel(X_enc, y_enc, x_data, H_enc)`**: Trains a deep learning model on encrypted data.
  - This function defines and trains a DNN regressor using the encrypted input features `X_enc` and the encrypted target values `y_enc`. The model architecture includes multiple hidden layers with specified units. The training process includes evaluation of the model's performance on encrypted data and calculation of RMSE and accuracy.

### Data Loading

- **`dataload(cci_data, input_cols, output_cols)`**: Loads and preprocesses the data, performing encryption on a subset of the data.
  - This function reads the credit card dataset and selects the input and output columns for training. It performs encryption on a subset of the data using the `encryption_train` function and returns the original and encrypted datasets.

## Results

### Vanilla Model

Outputs the accuracy of the model on the test set.
- **Test Set Accuracy**: 0.779

### Homomorphic Encryption Model

Outputs the accuracy and RMSE of the model on the encrypted data.
- **HE Accuracy**: 1.0
- **RMSE**: 0.564387940124446

An accuracy of 1.0 indicates that there is no loss of insight during the encryption process.
