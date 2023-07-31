
from torch import nn
import numpy as np
import torch.nn.functional as F

# This is not the exact neural network architecture for our metricNN. This is just the trial version:
class metricNN(nn.Module):

    # In the __init__ function we define all the layers that we want to have:
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(metricNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size2, 1)


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        y_pred = self.linear3(out)
        return y_pred

'''The above method is one way of defining the neural network. 
   The second method is to use the activation functions directly in the forward pass function to the output of the linear layers 
   using the torch API.'''


        

