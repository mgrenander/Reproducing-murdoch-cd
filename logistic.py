from torchtext import data, datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch.autograd as autograd
import sys

# GPU device
DEVICE = int(sys.argv[1])
torch.cuda.set_device(DEVICE)

class LogisticRegression(nn.Module):
    def __init__(self, embedding_dim, vocab_size, label_size, gpu_device):
        super(LogisticRegression, self).__init__()
        self.gpu_device = gpu_device
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(vocab_size, label_size)

    def forward(self, batch):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.

        # Clear hidden state
        # embeds = self.word_embeddings(batch.text)
        return self.linear(batch.text.type('torch.cuda.FloatTensor'))

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
print("Creating model")
model = LogisticRegression(embedding_dim=300, vocab_size=300, label_size=2, gpu_device=DEVICE)
# model.word_embeddings.weight.data = inputs.vocab.vectors
model.cuda(device=DEVICE)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

tqdm_epoch = tqdm(range(100), desc="Epoch")
for epoch in tqdm_epoch:
    train_iter.init_epoch()
    tqdm_batch = tqdm(train_iter, desc="Batch")
    for batch in tqdm_batch:
        # Clear gradient before each new instance
        model.train()
        opt.zero_grad()

        log_probs = model(batch)

        # Compute the loss and gradients and update the parameters by opt.step()
        loss = loss_fn(log_probs, batch.label)
        loss.backward()
        opt.step()

# Evaluate Logistic Regression
print("Evaluating on test data")
num_correct = 0
model.eval()
test_iter.init_epoch()
for test_batch in test_iter:
    answer = model(test_batch)
    num_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()

test_acc = 100. * num_correct / len(test)
print("Test accuracy: {}".format(test_acc))

# Save model
torch.save(model, "logis-model.pt")