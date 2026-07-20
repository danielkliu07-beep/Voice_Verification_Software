import torch
import torch.nn as nn
from torch.nn import functional as F


class Res2DilatedConv1D(nn.Module):

    def __init__(self, channels, kernel_size = 3, dilation = 1, scale = 8): #Scale = 8 -> Split input into 8 chunks

        super().__init__()

        assert channels % scale == 0

        self.width = channels // scale #Width - how much input and output channels each of the 8 chunks have 
        self.scale = scale

        padding = dilation * (kernel_size - 1) // 2 #Padding is calculated to ensure temporal dimension (T) remains unchanged

        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels = self.width, out_channels = self.width, kernel_size = kernel_size, dilation = dilation, padding = padding) for _ in range(self.scale - 1)])

    def forward(self, x): #x shape = (batch, channel, time)

        chunks = torch.split(x, self.width, dim = 1) #Splits the input vector into 8 different chunks of dimension (batch, time)
        out = []

        for i in range(self.scale):
            if i == 0:
                out.append(chunks[i])
            elif i == 1:
                out.append(self.convolutions[i - 1](chunks[i]))
            else:
                residual_connection = chunks[i] + out[i - 1]
                out.append(self.convolutions[i - 1](residual_connection))

        return torch.cat(out, dim = 1) #Adds back the channel dimension 



class SE_Block(nn.Module):

    #1-Dimensional Squeeze-Excitation Res2Blocks

    def __init__(self, reduced_dimension, channels):

        super().__init__()

        #Note that in_features = col, out_features = row

        self.W1 = nn.Linear(in_features = channels, out_features = reduced_dimension)
        self.W2 = nn.Linear(in_features = reduced_dimension, out_features = channels)

    
    def forward(self, x): #x shape = (batch, channel, time)

        z = torch.mean(x, dim = 2) #z shape = (batch, channel)

        s = self.W1(z)
        s = F.relu(s)

        s = self.W2(s)

        s = torch.sigmoid(s)

        s = s.unsqueeze(2) #Turns s back into a (batch, channel, time) shape so we can multiply it with the orignal input
        out = s * x

        return out


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size = 3, dilation = 2, scale = 8):

        #kernel size = 3, dilation = 1, scale = 8
        
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block2 = nn.Sequential(
            Res2DilatedConv1D(channels, kernel_size = kernel_size, dilation = dilation, scale = scale),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size = 1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block4 = SE_Block(reduced_dimension = 128, channels = channels)


    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = out + x

        return out

class AttentiveStatPooling(nn.Module):

    def __init__(self, in_dim, bottleneck_dim):

        super().__init__()

        #Input dimension must be multiplied by 3 since we concatenate local input (1x), global mean (1x), and global std (1x)
        self.W1 = nn.Conv1d(in_channels = in_dim * 3, out_channels = bottleneck_dim, kernel_size = 1)
        self.W2 = nn.Conv1d(in_channels = bottleneck_dim, out_channels = in_dim, kernel_size = 1)
    
    def forward(self, x): #x shape: (batch, channels, time)

        t = x.size(2) #time dimension

        #Calculates the global mean of batch and channels for x while also expanding it to match the time dimension 
        global_mean = torch.mean(x, dim = 2, keepdim = True).expand(-1, -1, t)
        global_std = torch.std(x, dim = 2, keepdim = True).expand(-1, -1, t)

        context_vector = torch.cat([x, global_mean, global_std], dim = 1)

        e = self.W1(context_vector)
        e = F.relu(e)
        e = self.W2(e)

        alpha = F.softmax(e, dim = 2)

        #Weighted mean (Summation over time)
        mean = torch.sum(alpha * x, dim = 2)

        #Weighted standard deviation
        std = torch.sum(alpha * x ** 2, dim = -1) - mean ** 2
        std = torch.sqrt(std.clamp(min = 1e-9))

        out = torch.cat([mean, std], dim = 1)

        return out




class ECAPA_TDNN(nn.Module):

    def __init__(self, in_channels = 80, channels = 512, embd_dim = 192):
        
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = channels, kernel_size = 5, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block2 = SE_Res2Block(channels = channels, kernel_size = 3, dilation = 2, scale = 8)

        self.block3 = SE_Res2Block(channels = channels, kernel_size = 3, dilation = 3, scale = 8)

        self.block4 = SE_Res2Block(channels = channels, kernel_size = 3, dilation = 4, scale = 8)

        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels = channels * 3, out_channels = 1536, kernel_size = 1, dilation = 1),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            AttentiveStatPooling(in_dim = 1536, bottleneck_dim = 128),
            nn.BatchNorm1d(num_features = 3072)
        )

        self.block7 = nn.Sequential(
            nn.Linear(in_features = 3072, out_features = embd_dim),
            nn.BatchNorm1d(num_features = embd_dim)
        )

    def forward(self, x):
        
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out1 + out2)
        out4 = self.block4(out1 + out2 + out3)

        out5 = self.block5(torch.cat([out2, out3, out4], dim = 1))

        output = self.block6(out5)
        output = self.block7(output)

        return output





