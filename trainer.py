import torch
import torch.optim as optim
import torch.nn as nn
from LSTM import LSTMSentiment
import pickle
import preprocessing

torch.cuda.set_device(0)

print("Downloading data")
train_iter, dev_iter, test_iter, answers, inputs = preprocessing.get_data()

# TODO: modify vocab size with actual vocab size
print("Creating model")
model = LSTMSentiment(embedding_dim=300, hidden_dim=128, vocab_size=300, label_size=2)
model.word_embeddings.weight.data = inputs.vocab.vectors
model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Beginning training")
for epoch in range(5):
    train_iter.init_epoch()
    print("Epoch {}".format(epoch))
    for batch_idx, batch in enumerate(train_iter):
        model.train(); optimizer.zero_grad()
        model.hidden = model.init_hidden()

        # Forward pass
        label_scores = model(batch)

        # Backprop
        loss = loss_function(label_scores, batch.label)
        loss.backward()
        optimizer.step()

# calculate accuracy on validation set
n_test_correct, test_loss = 0, 0
for dev_batch_idx, dev_batch in enumerate(test_iter):
    answer = model(test_iter)
    n_test_correct += (torch.max((answer, 1))[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()

test_acc = 100. * n_test_correct / len(test_iter)
print(test_acc)