import torch
import csv
from Util import convert_senentces_to_id

model = torch.load('call/test.pth', weights_only=False)
model.eval()


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


word_to_id = {}
with open('data.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        key, value = row
        word_to_id[key] = int(value)


import string 

userIn = str(input("Please insert your text: "))

tokens = userIn.translate(str.maketrans('', '',string.punctuation))
tokens = tokens.lower().split()

text_to_int = [word_to_id[token] for token in tokens if token in word_to_id]

if not len(text_to_int):
    raise ValueError("Invalid Input! Please try again with new sample")

import numpy as np
text_tensor = torch.from_numpy(
    np.array([text_to_int])
)

with torch.no_grad():
    o = model(text_tensor.to(device))

n = torch.sigmoid(o).cpu().detach().numpy()
predic_spam_percentage = f"{(n[0][0]*100):.4f}%"
predic_ham_percentage = f"{((1-n[0][0])*100):.4f}%"

if n > 0.8: 
    print('Prediction: Spam ', predic_spam_percentage, " certain")
else:
    print("Prediction: Ham", (predic_ham_percentage), " certain")