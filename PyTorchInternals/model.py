import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import jit



class FizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size=10, output_size=4):
        super(FizBuzNet, self).__init__()
        hidden_size = 100
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.settrace()

    def forward(self, batch):
        self.settrace()
        hidden = self.hidden(batch)
        activated = F.sigmoid(hidden)
        out = self.out(activated)
        return F.sigmoid(out)
    
    @staticmethod
    def settrace():
        pass # import pdb; pdb.set_trace()


@jit.compile(nderivs=0)
class JITFizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size=10, output_size=4):
        super(JITFizBuzNet, self).__init__()
        hidden_size = 100
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        hidden = self.hidden(batch)
        activated = F.sigmoid(hidden)
        out = self.out(activated)
        return F.sigmoid(out)