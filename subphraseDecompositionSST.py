# Decomposing SST Reviews into subphrases of opposite sentiments

from torchtext import data, datasets
import torch
import os
from collections import Counter
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn


#-------------------- Extract Data -------------------------#

inputs = data.Field(lower='preserve-case')
answers = data.Field(sequential=False, unk_token=None) # y: floats

train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
                                       filter_pred=lambda ex: ex.label != 'neutral')


#-------------------- Parse the reviews -------------------#

# List of tuples: (sentence, (+,-)) where + are positive and - are negative subsentences.
# The subsentences are of lengths 1/3 and 2/3 of the given sentence
def parse(train):
	
	sentencels=[]	
	for _,sub in enumerate(train):
	    # check if sub is subsentence of curr_sentence
	    if set(sub.text).issubset(set(cur_sentence)):
		l = len(sub.text)
		# Check if length of subtree is between 1/3 and 2/3
		if (l <= cur_length*2/3.0) and (l >= cur_length/3.0) :
		    # get sentiment of subsentence
		    sentiment = sub.label
		    # add subsentence to corresponding list
		    if sentiment == 'positive':
		        sentencels[-1][1][0].append(sub)
		    elif sentiment == 'negative':
		        sentencels[-1][1][1].append(sub)

	    else:
		sentencels.append((sub, ([],[])))
		cur_sentence = sub.text
		cur_length = len(cur_sentence)

	return sentencels
