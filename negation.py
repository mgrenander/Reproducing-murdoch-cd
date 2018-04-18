from ContextualDecomposition import CD
import torch
from torchtext import data, datasets
import numpy as np


def rolling_window(phrase, sub):
    tups = []
    for i in range(phrase.shape[0]):
        if i + len(sub) > phrase.shape[0]:
            break
        else:
            if np.array_equal(phrase[i:i+len(sub)], sub):
                tups.append((i, i+len(sub)-1))
    return tups


def format_indices(ls):
    formatted_ls = []
    for tup in ls:
        phrase = inputs.numericalize([tup[0].text], device=-1, train=False)
        subphrases = [inputs.numericalize([sub.text], device=-1, train=False) for sub in tup[1]]
        np_phrase = phrase.data.numpy()
        idx_tups = []
        for sub in subphrases:
            np_sub = sub.data.numpy()
            idx_tups += rolling_window(np_phrase, np_sub)
        formatted_ls.append((phrase, idx_tups))
    return formatted_ls


def filterTrees(trees):
    pos_trees = []
    neg_trees = []
    neut_trees = []
    for tree in trees:
        phrase, subs = tree
        if phrase.label == 'positive':
            pos_trees.append(tree)
        elif phrase.label == 'neutral':
            neut_trees.append(tree)
        elif phrase.label == 'negative':
            neg_trees.append(tree)
    return pos_trees, neg_trees, neut_trees


def parseTrees(train):
    phrases = []
    i = 0  # index where we put phrase in phrases
    while i < len(train):
        phrase = train[i]
        i += 1
        subs = []
        sub = train[i]
        while set(sub.text).issubset(set(phrase.text)):
            subs.append(sub)
            i += 1
            if i >= len(train):
                break
            sub = train[i]
        phrases.append((phrase, subs))
    pos, neg, neut = filterTrees(phrases)
    pos = format_indices(pos)
    neg = format_indices(neg)
    neut = format_indices(neut)
    return pos, neg, neut


def get_cd_scores(trees, model, label):
    hist = []
    for tree in trees:
        phrase, tups = tree
        for start, stop in tups:
            score_array = CD(phrase, model, start, stop)

            if label == "pos":
                score = score_array[0]
            elif label == "neg":
                score = score_array[1]
            else:
                score = np.max(score_array, axis=1)

            hist.append(score)
    return hist

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

# ok = inputs.numericalize([test[0].text], device=-1, train=False) ## Why does this not work???

print("Filtering data")
pos, neg, neut = parseTrees(train)
pos_hist = get_cd_scores(pos, model, "pos")
neg_hist = get_cd_scores(neg, model, "neg")
neut_hist = get_cd_scores(neut, model, "neut")
print(pos_hist)
print(neg_hist)
print(neut_hist)
