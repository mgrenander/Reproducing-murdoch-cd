import torch
import torch.optim as optim
from LSTM import LSTMSentiment

model = LSTMSentiment(embedding_dim=300, hidden_dim=128, vocab_size=300)