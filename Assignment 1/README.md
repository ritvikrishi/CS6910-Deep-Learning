# Assignment 1

In this assignment, we implement a feedforward neural network with backpropogation. We used wandb.ai to search (or in wandb terms "sweep") for the best hyperparameters.
The neural network was trained on Fashion-MNIST data and we obtained an accuracy of around 87% on the test data.

## Instructions to run code:

Change the values of hyperparameters as needed in FFNN.py and run 

$ python FFNN.py

to get the test accuracy on the dataset. The jupyter notebook was used on google colab to run wandb sweeps. A wandb account would be required to run those sweeps.

### Packages required:
numpy
pandas
sklearn
keras
matplotlib
