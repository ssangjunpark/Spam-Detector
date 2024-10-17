# Spam Detector
The goal of this project is to build a model utizling PyTorch's LSTM and Embedding layer. 

Custom text preprocessing and tokenization methods were written to process the data.

The model performs 98% accuracy in predicting correct lables for test data (20% of total dataset).



## Test.py
To use the pre-trained model, execute Test.py. It will request for a string input on the commandline. 

## Train.py
To train the model with custom dataset, change line 5 with appropriate path for the train data in .csv format. 

The train data must have string label on its first column and corresponding document on its second column. Data must be encoded in ISO-8859-1 format.

## Util.py
Contains methods for Test.py and Train.py
