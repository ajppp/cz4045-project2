# cz4045-project2
Source Code for Project 2 of CZ4045

For this project, there are two questions.
The first is on implementing a neural language model using a feedforward network instead of an RNN.
The second is on Named Entity Recognition (NER). The aim is to see how a CNN can be used as an encoder instead of the bidirectional LSTM which has already been implemented.

The source code for both questions are in the SourceCode folder, which contains 2 subfolders, neural-probabilistic-model (Question 1) and named-entity-recognition (Question 2). 

## Question 1 (Neural Language Model)
`FNNModel.py` is the class file for the FNN model that was built from scratch
`FNNModel1.py` and `FNNModel2.py` were classes made for convenience so that we could train the model with multiple layers
Line 65 of the class should be uncommented while lines 68 and 69 are commented when training the model
The opposite is needed when generating text using `generate.py`

In order to train the model 
`python fnn_main.py --cuda --model FNN`

In order to generate text based on some random phrase,
`python generate.py`



## Question 2 (Named Entity Recognition)
There are 4 notebooks, CNN_1layer, CNN_2layers, and CNN_3layers. 
_ is the original notebook from https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial/blob/master/Named_Entity_Recognition-LSTM-CNN-CRF-Tutorial.ipynb.
CNN_1layer, CNN_2layers and CNN_3layers are edited from the original notebook to implement a CNN layer (or layers) in replacement of the LSTM layer. 
For all notebooks, to train the model from scratch, running the notebook will be sufficient and no modification of the code is needed.

The models directory contains our trained CNN models. To evaluate our models without training them, simply comment out 'parameters['reload']' in the notebook.

The logs directory contains the logs of training each model, which shows the losses at every 2000 steps. The F-score for the train, validation, and test set is also computed regularly.
