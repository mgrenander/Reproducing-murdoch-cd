from torchtext import data, datasets
import torch
import os
import pickle

def get_data():
    # preserves case of words
    inputs = data.Field(lower='preserve-case')

    # No tokenization applied because the data is not seq
    # unk_token=None: ignore out of vocabulary tokens, since these are grades
    answers = data.Field(sequential=False, unk_token=None) # y: floats

    # fine_grained=False - use the following grade mapping { 0,1 -> negativ; 2 -> neutral; 3,4 -> positive }
    # filter=... - remove the neutral class to reduce the problem to binary classification
    # train_subtrees=False - Use only complete review instead of also using subsentences (subtrees)
    train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
                                           filter_pred=lambda ex: ex.label != 'neutral')

    # build the initial vocabulary from the SST dataset
    inputs.build_vocab(train, dev, test)

    # then enhance it with the pre-trained glove model
    inputs.vocab.load_vectors('glove.6B.300d')

    # build the vocab for the labels (only consists of 'positive','negative')
    answers.build_vocab(train)

    # You can use these iterators to train/test/validate the network :)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=1, device=2)
    return train_iter, dev_iter, test_iter, answers, inputs

# Pickle values
# def dump_pickle(obj, file):
#     with open(file, 'wb') as f:
#         pickle.dump(obj, f)
#
# print(type(train_iter))
# print(type(inputs))
# print(type(answers))
#
#
# dump_pickle(train_iter, "train.pic")
# dump_pickle(dev_iter, "dev.pic")
# dump_pickle(test_iter, "test.pic")
# dump_pickle(inputs, "inputs.pic")
# dump_pickle(answers, "answers.pic")

