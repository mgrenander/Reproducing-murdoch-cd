from ContextualDecomposition import CD
import torch
from torchtext import data, datasets

# Load model and data
DEVICE = 0
model = torch.load("model.pt")
inputs = data.Field(lower='preserve-case')
answers = data.Field(sequential=False, unk_token=None)
train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
inputs.build_vocab(train, dev, test)
inputs.vocab.load_vectors('glove.6B.300d')
answers.build_vocab(train)
train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=1, repeat=False, device=DEVICE)

for batch in train_iter:
    print(CD(batch, model, 0, len(batch.text)-1))
    break