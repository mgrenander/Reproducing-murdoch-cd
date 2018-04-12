import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda(device=2)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(device=2))

    def forward(self, batch):
        embeds = self.word_embeddings(batch.text)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        # label_scores = F.log_softmax(label_space)
        return label_space
