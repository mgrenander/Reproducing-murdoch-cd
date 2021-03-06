import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class LSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, gpu_device):
        super(LSTMSentiment, self).__init__()
        self.gpu_device = gpu_device
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda(device=self.gpu_device)),
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(device=self.gpu_device))

    def forward(self, batch):
        # Clear hidden state
        self.hidden = self.init_hidden(batch.text.size()[1])

        embeds = self.word_embeddings(batch.text)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        # label_scores = F.log_softmax(label_space)
        return label_space
