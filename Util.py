import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import string


from sklearn.utils import shuffle


def preprocess_test(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df = df.drop(columns=[df.columns[2],df.columns[3],df.columns[4]], axis=1)
    df.columns = ['labels', 'document']
    df['target'] = df['labels'].map({'ham': 0, 'spam': 1})

    return df


def transform_word_to_id(df) -> dict:
    uid = 1
    word_id = {'<PAD>':0}

    for _, row in df.iterrows():
        pure_string = row['document'].translate(str.maketrans('', '',string.punctuation))
        tokens = pure_string.lower().split()
        for token in tokens:
            if token not in word_id:
                word_id[token] = uid
                uid += 1


    return word_id


def convert_senentces_to_id(df, word_to_id):
    df_sentences_to_id = []
    for _, row in df.iterrows():
        pure_string = row['document'].translate(str.maketrans('', '',string.punctuation))
        tokens = pure_string.lower().split()
        sentence_to_ids = [word_to_id[token] for token in tokens if token in word_to_id]
        df_sentences_to_id.append(sentence_to_ids)

    return df_sentences_to_id


def data_generator(X, Y, batch_size=32):
    X, Y = shuffle(X, Y)
    number_batches = int(np.ceil(len(X) / batch_size))

    for i in range(number_batches):
        end = min((i+1) * batch_size, len(X))

        X_batch = X[i*batch_size:end]
        Y_batch = Y[i*batch_size:end]

        max_sentence_len = np.max([len(x) for x in X_batch])
        
        for j in range(len(X_batch)):
            x = X_batch[j]
            padding = [0] * (max_sentence_len - len(x))
            X_batch[j] = padding + x 
    

        X_batch = torch.from_numpy(np.array(X_batch)).long()
        Y_batch = torch.from_numpy(np.array(Y_batch)).long()

        yield X_batch, Y_batch

class RNN(nn.Module):
    def __init__(self, device, number_of_embeddings, embedded_dimension, hidden_size, number_of_rnnlayers, output_size):
        super(RNN, self).__init__()
        self.device = device
        self.nE = number_of_embeddings
        self.eD = embedded_dimension
        self.nH = hidden_size
        self.nK = output_size
        self.nL = number_of_rnnlayers

        self.embeding = nn.Embedding(self.nE, self.eD)
        self.rnn = nn.LSTM(
            input_size= self.eD,
            hidden_size= self.nH,
            num_layers=self.nL,
            batch_first=True
        )

        self.fc = nn.Linear(self.nH, self.nK)

    def forward(self, X):
        h0 = torch.zeros(self.nL, X.size(0), self.nH).to(self.device)
        c0 = torch.zeros(self.nL, X.size(0), self.nH).to(self.device)

        o = self.embeding(X)

        o, _ = self.rnn(o, (h0, c0))
        
        o = o[:, -1, :]

        o = self.fc(o)

        return o
    


def mini_batch_gd(device, model, train_generator, test_generator, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        t_start = datetime.now()
        
        train_loss = []
        for x, y in train_generator():
            y = y.view(-1, 1).float()

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            o = model(x)
            loss = criterion(o, y)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        
        
        train_loss_average = np.mean(train_loss)
        train_losses.append(train_loss_average)

        test_loss = []
        for x,y in test_generator():
            x, y = x.to(device), y.to(device)
            y = y.view(-1, 1).float()
            o = model(x)
            loss = criterion(o, y)
            test_loss.append(loss.item())

        test_loss_average = np.mean(test_loss)
        test_losses.append(test_loss_average)

        duration = datetime.now() - t_start

        print(f"Epoch: {epoch}/{epochs}, Train Acc: {train_loss_average}, Test Acc: {test_loss_average}, Duration: {duration}")
    
    return train_losses, test_losses


def calculate_accuracy(device, model, train_generator, test_generator):
    with torch.no_grad():
        train_correct = 0
        train_total = 0

        for x, y in train_generator():
            y = y.view(-1, 1).float()
            x, y = x.to(device), y.to(device)
            
            #calculate the output 
            o = model(x)

            predictions = (o > 0)

            train_correct += (predictions == y).sum().item()
            train_total += len(predictions)

        test_correct = 0
        test_total = 0

        for x, y in test_generator():
            y = y.view(-1, 1).float()
            x, y = x.to(device), y.to(device)

            o = model(x)
            predictions  = (o > 0)

            test_correct += (predictions == y).sum().item()
            test_total += len(predictions)

        return (train_correct/train_total), (test_correct/test_total)