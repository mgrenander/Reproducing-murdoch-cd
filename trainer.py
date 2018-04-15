import torch
import torch.optim as optim
import torch.nn as nn
from model import LSTMSentiment
import sys
import os
from tqdm import tqdm

from torchtext import data
from torchtext import datasets

#########################################
# INITIALIZING
# Select GPU we will use
DEVICE = int(sys.argv[1])
torch.cuda.set_device(DEVICE)

# Select to resume checkpoint or not
RESUME_CKPT = bool(int(sys.argv[2]))

###########################################
# PREPROCESSING
print("Downloading and preprocessing data")
inputs = data.Field(lower='preserve-case')
answers = data.Field(sequential=False, unk_token=None)
train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
inputs.build_vocab(train, dev, test)
inputs.vocab.load_vectors('glove.6B.300d')
answers.build_vocab(train)
train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=50, repeat=False, device=DEVICE)

############################################
# CREATING MODEL
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


#############################################
# TRAINING
print("Beginning training")
early_stop_test = 0
tqdm_epoch = tqdm(range(5), desc="Epoch")
for epoch in tqdm_epoch:
    train_iter.init_epoch()

    tqdm_batch = tqdm(train_iter, desc="Batch")
    loss = None
    for b_id, batch in enumerate(tqdm_batch):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        label_scores = model(batch)

        # Backprop
        loss = loss_function(label_scores, batch.label)
        loss.backward()
        optimizer.step()

        if b_id % (len(tqdm_batch)/10) == 0:
            tqdm_batch.set_postfix(loss=loss.data[0])

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
        early_stop_test = val_acc
        torch.save(model, model_path)

###########################################
# TEST EVALUATION
print("Evaluating on test data")
num_correct = 0
model.eval()
test_iter.init_epoch()
for test_batch in test_iter:
    answer = model(test_batch)
    num_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()

test_acc = 100. * num_correct / len(test)
print("Test accuracy: {}".format(test_acc))
