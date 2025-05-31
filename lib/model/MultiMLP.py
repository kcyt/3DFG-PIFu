
import torch
import torch.nn as nn
import torch.nn.functional as F 

class MultiMLP(nn.Module):
    def __init__(self, 
                 filter_channels, # [257, 1024, 512, 256, 128, 1]
                 res_layers=[], # [2,3,4]
                 last_op=None): # nn.Sigmoid()
        super(MultiMLP, self).__init__()

        self.filters = nn.ModuleList()
        self.res_layers = res_layers
        self.last_op = last_op

        for l in range(0, len(filter_channels)-1):

            if l<3:
                residual_channel = filter_channels[0]
            else:
                residual_channel = filter_channels[3]

            if l in self.res_layers:
                self.filters.append(nn.Conv1d(
                    filter_channels[l] + residual_channel,
                    filter_channels[l+1],
                    1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l+1],
                    1))


    def forward(self, feature1, feature2):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''
        y1 = feature1
        tmpy1 = feature1
        # runs through the filters until the features reach the dimension of 256 (is referred to the author as the 4th layer).
        for i, f in enumerate(self.filters[0:3]):
            y1 = f(
                y1 if i not in self.res_layers
                else torch.cat([y1, tmpy1], 1)
            )
            y1 = F.leaky_relu(y1) # Shape of [B, 256, N] 

        # do the same for feature2
        y2 = feature2
        tmpy2 = feature2
        # runs through the filters until the features reach the dimension of 256 (is referred to the author as the 4th layer).
        for i, f in enumerate(self.filters[0:3]):
            y2 = f(
                y2 if i not in self.res_layers
                else torch.cat([y2, tmpy2], 1)
            )
            y2 = F.leaky_relu(y2) # Shape of [B, 256, N] 


        # averaging the two features
        y = torch.add(y1,y2)
        y = torch.mul(y, 0.5)

        # finish running through the rest of the filters
        tmpy = y
        for i, f in enumerate(self.filters[3:], start=3):
            y = f(
                y if i not in self.res_layers
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters)-1:
                y = F.leaky_relu(y) # Shape of [B, 256, N] 

        if self.last_op is not None:
            y = self.last_op(y)

        return y  # y is the output 




