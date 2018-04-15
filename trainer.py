import torch
import torch.optim as optim
import torch.nn as nn
from model import LSTMSentiment
import preprocessing
import sys
import os
from tqdm import tqdm

from torchtext import data
from torchtext import datasets

import evaluate_lstm

# Select GPU we will use
DEVICE = int(sys.argv[1])
torch.cuda.set_device(DEVICE)

# Select to resume checkpoint or not
RESUME_CKPT = bool(int(sys.argv[2]))

###########################################
# PREPROCESSING
print("Downloading data")
# train_iter, dev_iter, test_iter, answers, inputs = preprocessing.get_data(device=DEVICE)

inputs = data.Field(lower='preserve-case')
answers = data.Field(sequential=False, unk_token=None)

train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
										filter_pred=lambda ex: ex.label != 'neutral')
inputs.build_vocab(train, dev, test)

inputs.vocab.load_vectors('glove.6B.300d')
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=50, repeat=False, device=DEVICE)


############################################
print("Creating model")
model = LSTMSentiment(embedding_dim=300, hidden_dim=168, vocab_size=300, label_size=2, gpu_device=DEVICE)
model.word_embeddings.weight.data = inputs.vocab.vectors
model.cuda(device=DEVICE)

# Load previously checkpointed model if it exists
model_path = "model.pt"
if os.path.exists(model_path) and RESUME_CKPT:
    print("Loading previously stored model")
    model = torch.load(model_path)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Beginning training")
early_stop_test = 0
tqdm_epoch = tqdm(range(5), desc="Epoch")
for epoch in tqdm_epoch:
    train_iter.init_epoch()

    tqdm_batch = tqdm(train_iter, desc="Batch")
    loss = None
    for id, batch in enumerate(tqdm_batch):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        label_scores = model(batch)

        # Backprop
        loss = loss_function(label_scores, batch.label)
        loss.backward()
        optimizer.step()

        if id % (len(tqdm_batch)/10) == 0:
            tqdm_batch.set_postfix(loss=loss.data[0])

    print("Finished training for epoch {}".format(epoch))

    # Early stopping: save model if this one was better
    num_correct = 0
    model.eval()
    dev_iter.init_epoch()
    for val_batch in dev_iter:
        answer = model(val_batch)
        num_correct += (torch.max(answer, 1)[1].view(val_batch.label.size()).data == val_batch.label.data).sum()
    val_acc = 100. * num_correct / len(dev)

    tqdm_epoch.set_postfix(loss=loss.data[0], acc=val_acc)

    if val_acc > early_stop_test:
        # Save model
        print("Found new best model with dev accuracy: {}".format(val_acc))
        torch.save(model_path, model)

# Evaluate test data
evaluate_lstm.evaluate()