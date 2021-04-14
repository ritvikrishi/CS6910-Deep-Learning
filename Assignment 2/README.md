# Assignment 2
In this assignment, we implemented a CNN using keras and trained it on the 
[inaturalist dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) in part A and fine-tuned pretrained models (like EfficientNetB3,
InceptionResnetV2, VGG19, Xception, InceptionV3, ResNet50) to run on the same dataset in Part B.
[wandb.ai](wandb.ai) was the tool used to log runs for hyperparameter tuning.

## Instructions to run Part A
It is recommended to run the jupyter notebook for part A using google colab to utilise the freely available GPUs. 
- Run all cells in PartA.ipynb which import the data and define the functions for building cnn. 
- Skip the wandb cells and go to the manual training cell. You can change the values of number of filters in each layer by varying nFilters, the filter size by varying ksize, and other parameters by varying the arguments of BuildCNN.
- Compile the model, and run model.fit(). 
- The evaluation and visualiations can be done by running the following cells.

model-best-A.h5 contains the trained keras model with hyperparameter choices that performs best for this dataset.

The accuracy obtained on running the best model on the test data was 42%.
