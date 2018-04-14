import torch
import torch.optim as optim
import torch.nn as nn
from LSTM import LSTMSentiment
import preprocessing
import sys
from tqdm import tqdm

# Select GPU we will use
DEVICE = int(sys.argv[1])
torch.cuda.set_device(DEVICE)

print("Downloading data")
train_iter, dev_iter, test_iter, answers, inputs = preprocessing.get_data(device=DEVICE)

print("Creating model")
model = LSTMSentiment(embedding_dim=300, hidden_dim=168, vocab_size=300, label_size=2, gpu_device=DEVICE)
model.word_embeddings.weight.data = inputs.vocab.vectors
model.cuda(device=DEVICE)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Beginning training")
early_stop_test = 0
for epoch in tqdm(range(5)):
    train_iter.init_epoch()
    print("Epoch {}".format(epoch))
    for batch in tqdm(train_iter):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        label_scores = model(batch)

        # Backprop
        loss = loss_function(label_scores, batch.label)
        loss.backward()
        optimizer.step()

    # Early stopping: save model if this one was better
    num_correct = 0
    for val_batch in dev_iter:
        answer = model(val_batch)
        num_correct += (torch.max((answer, 1))[1].view(val_batch.label.size()).data == val_batch.label.data).sum()
    val_acc = 100. * num_correct / len(dev_iter)

    if val_acc > early_stop_test:
        # Save model
        print("Found new best model with dev accuracy: {}".format(val_acc))
        torch.save("data/model.pt", model)


# calculate accuracy on testing set
n_test_correct = 0
for test_batch in test_iter:
    answer = model(test_batch)
    n_test_correct += (torch.max((answer, 1))[1].view(test_batch.label.size()).data == test_batch.label.data).sum()

test_acc = 100. * n_test_correct / len(test_iter)
print(test_acc)
