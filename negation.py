from ContextualDecomposition import CD
import torch
from torchtext import data, datasets


def filter_neg(example):
    pass


# Load model and data
print("Loading model")
model = torch.load("model.pt", map_location=lambda storage, loc: storage)

print("Fetching data and creating splits")
inputs = data.Field(lower='preserve-case')
answers = data.Field(sequential=False, unk_token=None)
train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True, filter_pred=None)
inputs.build_vocab(train, dev, test, vectors="glove.6B.100d")
answers.build_vocab(train)
vocab = inputs.vocab

ok = [vocab.stoi[i] for i in test[0].text]
print(CD(ok, model, 0, len(ok)-1))


# print("Computing CD scores")
# for ex in train:

    # filtered_tuples = filter_neg(ex)
    # print(filtered_tuples)
    # break