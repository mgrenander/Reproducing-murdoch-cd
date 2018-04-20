from ContextualDecomposition import CD
import torch
from torchtext import data, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rolling_window(phrase, sub):
    for i in range(phrase.shape[0]):
        if np.array_equal(phrase[i:i+len(sub)], sub):
            return i, i+len(sub)-1
    return -1  # Error!


def format_indices(ls):
    formatted_ls = []
    for tup in ls:
        phrase = inputs.numericalize([tup[0].text], device=-1, train=False)
        subphrase = inputs.numericalize([tup[1].text], device=-1, train=False)
        idx_tup = rolling_window(phrase.data.numpy(), subphrase.data.numpy())
        if idx_tup == -1:
            raise ValueError("Could not find subphrase in phrase!")
        else:
            formatted_ls.append((phrase, idx_tup, tup[2]))
    return formatted_ls


def filterTrees(trees):
    pos_trees = []
    neg_trees = []
    all_trees = []
    for tree in trees:
        all_trees.append(tree)
        phrase, _, _ = tree
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
    i += 1
    first_child = train[i]
    i += 1
    sub = train[i]
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
    print("Filtering trees")
    negation_words = ["not", "n't", "lacks", "nobody", "nor", "nothing", "neither", "never", "none", "nowhere", "remotely"]
    phrases = []  # Format: [(sentence, phrase being negated, negation term index)]
    i = 0
    while i < len(train):
        phrase = train[i]
        if len(phrase.text) > 10:  # Phrase is too long
            i = get_next_tree(train, i)
            continue
        fc = train[get_first_child(train, i)]  # First child
        sc = train[get_second_child(train, i)]  # Second child
        fw_idx = get_next_word(train, i)  # Index for first word
        fw = train[fw_idx].text[0]  # First word
        sw = train[get_next_word(train, fw_idx)].text[0]  # Second word
        if fw in negation_words:  # First word is in negation words
            scnn = sc_find_nneut(train, i)
            if sc.label != "neutral":
                phrases.append((phrase, sc, 0))
            elif scnn != -1 and phrase.label != "neutral":
                phrases.append((phrase, train[scnn], 0))
        elif sw in negation_words: # Second word is in negation words
            scnn = sc_find_nneut(train, i)
            if sc.label != "neutral":
                phrases.append((phrase, sc, 1))
            elif scnn != -1 and phrase.label != "neutral":
                phrases.append((phrase, train[scnn], 1))
        i = get_next_tree(train, i)
    print("Sorting trees by label")
    pos, neg, all = filterTrees(phrases)
    print("Formatting indices")
    pos = format_indices(pos)
    neg = format_indices(neg)
    all = format_indices(all)
    return pos, neg, all


def get_cd_scores(trees, model):
    hist = []
    for tree in trees:
        phrase, (start, stop), neg_word_id = tree
        phrase_score, _ = CD(phrase, model, 0, len(phrase)-1)
        sub_score, _ = CD(phrase, model, start, stop)
        neg_score, _ = CD(phrase, model, neg_word_id, neg_word_id)
        final_score = phrase_score - sub_score - neg_score
        hist.append(final_score[0] - final_score[1])
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

print("Parsing trees")
p, n, a = parseTrees(train)

print(len(p))
print(len(n))
print(len(a))

print("Computing CD scores")
p_scores = get_cd_scores(p, model)
n_scores = get_cd_scores(n, model)
a_scores = get_cd_scores(a, model)
print("Plotting results")

_, ax = plt.subplots()
sns.distplot(p_scores, hist=False, color='blue', kde_kws={"shade":True}, ax=ax, label="Positive")
sns.distplot(n_scores, hist=False, color='green', kde_kws={"shade":True}, ax=ax, label="Negative")
sns.distplot(a_scores, hist=False, color='red', kde_kws={"shade":True}, ax=ax, label="All")

ax.set(xlabel='Contextual Decomposition Score', ylabel='Density')
ax.set_title('Contextual Decomposition for Negation Phrases')

fig = ax.get_figure()
fig.savefig('negation.png')

