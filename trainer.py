import torch
import torch.optim as optim
import torch.nn as nn
from LSTM import LSTMSentiment
import pickle

def load(pic_file):
    with open(pic_file, 'rb') as f:
        return pickle.load(f)

train = load("train.pic")
dev = load("dev.pic")
test = load("test.pic")
inputs = load("inputs.pic")
answers = load("answers.pic")

print(test)
print()
print(inputs)
print()
print(answers)


# TODO: modify vocab size with actual vocab size
model = LSTMSentiment(embedding_dim=300, hidden_dim=128, vocab_size=300, label_size=2)
for epoch in range(5):
    for sentence, tag in train:
        model.zero_grad()
        model.hidden = model.init_hidden()

        # Forward pass
        label_scores = model(sentence)

        # Backprop
        loss = nn.NLLLoss(label_scores, tag)

