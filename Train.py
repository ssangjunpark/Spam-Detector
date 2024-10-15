from sklearn.model_selection import train_test_split

from Util import preprocess_test, transform_word_to_id, convert_senentces_to_id, data_generator, RNN, mini_batch_gd, calculate_accuracy

df = preprocess_test('spam.csv')

# split test and train data
df_train, df_test = train_test_split(df, test_size=.2)

# tokenize specific words in df_train
word_to_id = transform_word_to_id(df_train)


# tokenize words in df_train using word_to_id
train_sentences_to_id = convert_senentces_to_id(df_train, word_to_id)
# tokenize words in df_test using word_to_id
test_sentences_to_id = convert_senentces_to_id(df_test, word_to_id)

import torch
# set device to mps
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# instantiate model 
model = RNN(device=device,number_of_embeddings=len(word_to_id), embedded_dimension=20, hidden_size=15, number_of_rnnlayers=2, output_size=1)
model.to(device)

# create batch data generator 
train_generator = lambda: data_generator(train_sentences_to_id, df_train.target)
test_generator = lambda: data_generator(test_sentences_to_id, df_test.target)

# train the model 
tel, trl = mini_batch_gd(device, model, train_generator, test_generator, epochs=15)

# plot the loss BCE with logits (no activation)
import matplotlib.pyplot as plt
plt.plot(tel, label='train_losses')
plt.plot(trl, label='test_losses')
plt.legend()
plt.show()

# calculate accrucay for train and test dataset
train_accuracy, test_accuracy = calculate_accuracy(device, model, train_generator, test_generator)
print(f"Train Acc: {train_accuracy}, Test Acc: {test_accuracy}")

# save the model
torch.save(model, 'call/test.pth')

# save word_to_id into csv file for later retrival
import csv
with open('data.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in word_to_id.items():
        writer.writerow([key, value])
