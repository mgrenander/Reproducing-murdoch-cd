import torch
import preprocessing
import sys
import os

try:
    _, _, test_iter, _, _ = preprocessing.get_data(device=int(sys.argv[1]))
except:
    raise("Failed to get test data. Check you selected the right GPU.")

if not os.path.exists("data/model.pt"):
    raise("You need to run trainer.py first!")

# calculate accuracy on testing set
model = torch.load("data/model.pt")
n_test_correct = 0
for test_batch in test_iter:
    answer = model(test_batch)
    n_test_correct += (torch.max((answer, 1))[1].view(test_batch.label.size()).data == test_batch.label.data).sum()

test_acc = 100. * n_test_correct / len(test_iter)
print("Test accuracy: {}".format(test_acc))
