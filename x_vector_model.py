import torch
import torch.nn as nn
from torch.nn import functional as F

class TDNNLayer(nn.Module):

    def __init__(self, input_dim, output_dim, context):
        super().__init__()
    
    def forward(self, x):

        return x




class XVectorModel(nn.Module):

    def __init__(self):
        
        frame1 = nn.Linear(in_features = 120, out_features = 512)





class AAMSoftmaxLoss(nn.Module):

    def __init__(self, margin, scale):
        pass

