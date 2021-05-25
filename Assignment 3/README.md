# Assignment 3 - Recurrent neural networks

## Instructions to run

It is recommended to run the jupyter notebook using google colab to utilise the freely available GPUs. 
- Run all cells in rnn.ipynb which import the data and define the functions for building the model. 
- Skip the wandb cells and run the manual training cell. You can change the values of the hyperparameters by varying the arguments of make_model().
- Compile the model using model.build(), and run model.fit().
- The character-level training and validation accuracies will be printed.
- Run model.predict() to get character-level accuracies for test dataset and model.word_level(filename) to get word level accuracies and save predictions for test data 
- The connectivity visualiations can be done by running the following cells.

predictions-vanilla.csv and predictions-attention.csv contain the predicted outputs for the test file of hi/lexicons from the dakshina dataset.
