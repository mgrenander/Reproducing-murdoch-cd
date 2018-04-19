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
    all_trees = []
    for tree in trees:
        all_trees.append(tree)
        phrase, subs = tree
        if phrase.label == 'positive':
            pos_trees.append(tree)
        elif phrase.label == 'negative':
            neg_trees.append(tree)
    return pos_trees, neg_trees, all_trees


def get_next_tree(train, i):
    phrase = train[i]
    i += 1
    sub = train[i]
    while set(sub.text).issubset(set(phrase.text)):
        i+=1
        if i >= len(train):
            break
        sub = train[i]
    return i


def get_first_child(train, i):
    return i + 1


def get_second_child(train, i):
    first_child = train[i+1]
    sub = train[i+2]
    while set(sub.text).issubset(set(first_child.text)):
        i+= 1
        sub = train[i]
    return i


def get_next_word(train, i):
    sub = train[i+1]
    while len(sub.text) != 1:
        i+=1
        sub = train[i]
    return i


def sc_find_nneut(train, i):
    sc = get_second_child(train, i)
    j = sc+1
    if j >= len(train):
        return -1
    sub = train[j]
    while set(sub.text).issubset(set(train[sc].text)):
        if sub.label != "neutral":
            return j
        j += 1
        if j >= len(train):
            return -1
        sub = train[j]
    return -1

def parseTrees(train):
    negation_words = ["not", "n't", "lacks", "nobody", "nor", "nothing", "neither", "never", "none", "nowhere", "remotely"]
    phrases = []
    i = 0
    while i < len(train):
        phrase = train[i]
        if len(phrase) >= 10:  # Phrase is too long
            i = get_next_tree(train, i)
            continue
        fc = train[get_first_child(train, i)]
        sc = train[get_second_child(train, i)]
        fw = train[get_next_word(train, i)]
        sw = train[get_next_word(train, fw)]
        if fw.text in negation_words:
            scnn = sc_find_nneut(train, i)
            if sc.label != "neutral":
                phrases.append((phrase, sc, fw))
            if scnn != -1:
                phrases.append((phrase, train[scnn], fw))
        elif sw.text in negation_words and sw.text in fc.text:
            scnn = sc_find_nneut(train, i)
            if sc.label != "neutral":
                phrases.append((phrase, sc, sw))
            elif scnn != -1:
                phrases.append((phrase, train[scnn], sw))
    print("Filtering parsed trees")
    pos, neg, all = filterTrees(phrases)
    print("Formatting indices")
    pos = format_indices(pos)
    neg = format_indices(neg)
    all = format_indices(all)
    return pos, neg, all


def get_cd_scores(trees, model, label):
    hist = []
    for tree in trees:
        phrase, tups = tree
        for start, stop in tups:
            score_array = CD(phrase, model, start, stop)
            if label == "pos":
                score = score_array[0] - score_array[1]
            elif label == "neg":
                score = score_array[0] - score_array[1]
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
train, dev, test = datasets.SST.splits(inputs, answers, fine_grained=False, train_subtrees=True, filter_pred=None)
inputs.build_vocab(train, dev, test, vectors="glove.6B.100d")
answers.build_vocab(train)
vocab = inputs.vocab

# ok = inputs.numericalize([test[0].text], device=-1, train=False) ## Why does this not work???

print("Parsing trees")
pos, neg, neut = parseTrees(train)
print("Computing CD scores")
pos_hist = get_cd_scores(pos, model, "pos")
neg_hist = get_cd_scores(neg, model, "neg")
neut_hist = get_cd_scores(neut, model, "neut")
print(pos_hist)
print(neg_hist)
print(neut_hist)
