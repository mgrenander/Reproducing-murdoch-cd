# Reproducing-murdoch-cd
Reproducing results from "Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs"

The report can be viewed on Overleaf at: <https://www.overleaf.com/15602350wgfqzhjhqpvy#/59266173/>

First run ``trainer.py <DEVICE #> <PREV MODEL>`` - this will generate ``model.pt``, which is the trained LSTM classifier. The parameters are:
* DEVICE #: the device number of the GPU. If none, then use -1.
* PREV MODEL: if you already have a ``model.pt`` file and want to further train it, set this flag to a nonzero integer (otherwise use 0)
