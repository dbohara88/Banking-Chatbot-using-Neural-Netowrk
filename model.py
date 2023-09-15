import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


#The given code represents a feedforward neural network model. 
# It is a basic three-layer neural network with fully connected layers. 
# The model takes an input tensor and passes it through two hidden layers with ReLU activation functions, 
# and then finally through an output layer with no activation function applied. The output layer provides the 
# raw scores or logits for the classes, without applying softmax activation.