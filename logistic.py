
# coding: utf-8

# In[1]:


from __future__ import print_function

from torchtext import data, datasets
import torch
import os
from collections import Counter
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn


# ## Preprocessing SST into Glove Vectors

# In[2]:


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
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=100, device=-1)


# ## Bag of Words Representation

# In[4]:


def preprocessingBOW(batch,vocab,istrain):
    tensor = torch.FloatTensor(len(vocab),len(batch))
    tensor.zero_()
    
    for i in xrange(len(batch)):
        # update frequency
        c = Counter()
        c.update(batch[i])
        
        localtensor = torch.FloatTensor(len(vocab))
        localtensor.zero_()
        
        localtensor[c.keys()] = torch.FloatTensor(c.values())
        tensor[:,i] = localtensor
    
    return tensor

inputsBOW = data.Field(lower='preserve-case', tensor_type=torch.FloatTensor, postprocessing=preprocessingBOW)

# No tokenization applied because the data is not seq
# unk_token=None: ignore out of vocabulary tokens, since these are grades
answersBOW = data.Field(sequential=False, unk_token=None)

# fine_grained=False - use the following grade mapping { 0,1 -> negativ; 2 -> neutral; 3,4 -> positive }
# filter=... - remove the neutral class to reduce the problem to binary classification
# train_subtrees=False - Use only complete review instead of also using subsentences (subtrees)
trainBOW, devBOW, testBOW = datasets.SST.splits(inputsBOW, answersBOW, fine_grained = False, train_subtrees = True,
                                       filter_pred=lambda ex: ex.label != 'neutral')
# build the initial vocabulary from the SST dataset
inputsBOW.build_vocab(trainBOW, devBOW, testBOW)

# build the vocab for the labels (only consists of 'positive','negative')
answersBOW.build_vocab(trainBOW)


# ## Logistic Regression Classifier

# In[5]:


class LogisticRegression(nn.Module): 
    
    def __init__(self, num_labels, vocab_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vector):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vector), dim=1)


# ### Setting Parameters

# In[79]:


num_labels = 2
vocab_size = len(inputsBOW.vocab)
model = LogisticRegression(num_labels, vocab_size)
learning_rate = 0.0001
num_epochs = 100


# In[80]:


train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (trainBOW, devBOW, testBOW), repeat=False, batch_size=100, device=-1)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    print('epoch #%s [' % epoch,end='.')
    i=0
    
    for _,batch in enumerate(train_iter):
        # Clear gradient before each new instance
        model.zero_grad()
        log_probs = model(batch.text)

        # Compute the loss and gradients and update the parameters by opt.step()
        loss = loss_fn(log_probs, batch.label)
        loss.backward()
        opt.step()
        
        i += 1
        if i % 50 == 0:
            print('.',end='')
    print(']')


# ## Evaluate Logistic Regression

# In[81]:


correct = 0
dev_loss = 0
for idx, dev_batch in enumerate(dev_iter):
    labels = dev_batch.label
    prediction = model(dev_batch.text)
    correct += (torch.max(prediction, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
    dev_loss = loss_fn(prediction, dev_batch.label)
dev_acc = 100. * correct / len(devBOW)


# In[82]:


print(dev_acc)


# In[316]:


# Step 5: Validate
   logloss = 0
   for _, batch in enumerate(dev_iter):
       log_probs = model(batch.text.data)
       logloss += loss_fn(log_probs,batch.label.data)
   print("Test log loss: " % )
   
   
# Validate
for _, batch in enumerate(dev_iter):
   log_probs = model(batch.text)
   loss = loss_fn()
   print(log_probs)
   
# CODE FROM PAPER
# calculate accuracy on validation set
n_dev_correct, dev_loss = 0, 0
for dev_batch_idx, dev_batch in enumerate(dev_iter):
    answer = model(dev_batch)
    n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
    dev_loss = criterion(answer, dev_batch.label)
dev_acc = 100. * n_dev_correct / len(dev)



# In[255]:


b.text.data

