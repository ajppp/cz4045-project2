# cz4045-project2
Source Code for Project 2 of CZ4045

For this project, there are two questions.
The first is on implementing a neural language model using a feedforward network instead of an RNN.
The second is on Named Entity Recognition (NER). The aim is to see how a CNN can be used as an encoder instead of the bidirectional LSTM which has already been implemented.
The source code for each of the questions are in their respective directories.
The latex file for the report will be in the directory called report. 
The PDF file of the report will also be in that directory.

# Question 1

`FNNModel.py` is the class file for the FNN model that was built from scratch
`FNNModel1.py` and `FNNModel2.py` were classes made for convenience so that we could train the model with multiple layers
Line 65 of the class should be uncommented while lines 68 and 69 are commented when training the model
The opposite is needed when generating text using `generate.py`

In order to train the model 
`python fnn_main.py --cuda --model FNN`

In order to generate text based on some random phrase,
`python generate.py`
