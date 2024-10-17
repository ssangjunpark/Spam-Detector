# Spam Detector
The goal of this project is to build a model utizling PyTorch's LSTM and Embedding layer. 

Custom text preprocessing and tokenization methods were written to process the data.

The model performs 98% accuracy in predicting correct lables for test data (20% of total dataset).

Successfully classifying it as Spam 
<img width="1439" alt="Screenshot 2024-10-17 at 3 46 38 PM" src="https://github.com/user-attachments/assets/f40c54b2-6136-4723-b0b4-e9c350091b50">
<img width="1439" alt="Screenshot 2024-10-17 at 3 46 23 PM" src="https://github.com/user-attachments/assets/4bd18db6-cdea-4226-9728-733d54026bb9">

Successfully classifying as Ham
<img width="1439" alt="Screenshot 2024-10-17 at 3 46 30 PM" src="https://github.com/user-attachments/assets/de26377c-590e-4427-971a-4204d20c7563">
<img width="1439" alt="Screenshot 2024-10-17 at 3 45 19 PM" src="https://github.com/user-attachments/assets/1f73638c-6a0e-4d2e-b7b6-018fedb17431">

Unsuccessful classification
<img width="1439" alt="Screenshot 2024-10-17 at 3 46 45 PM" src="https://github.com/user-attachments/assets/aa9349f4-2a9b-46f7-af00-66f549b380f7">


## Test.py
To use the pre-trained model, execute Test.py. It will request for a string input on the commandline. 

## Train.py
To train the model with custom dataset, change line 5 with appropriate path for the train data in .csv format. 

The train data must have string label on its first column and corresponding document on its second column. Data must be encoded in ISO-8859-1 format.

## Util.py
Contains methods for Test.py and Train.py
