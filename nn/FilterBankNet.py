import torch
import torch.nn as nn
import math


class Filter_bank_net(nn.Module):

    def __init__(self):
        filter_size = 3
        super(Filter_bank_net, self).__init__()
        self.dsac1 = nn.Conv1d(4, 128, filter_size, dilation=1)
        self.dsac2 = nn.Conv1d(128, 128, filter_size, dilation=2)
        self.dsac3 = nn.Conv1d(128, 128, filter_size, dilation=4)
        self.dsac4 = nn.Conv1d(128, 128, filter_size, dilation=8)
        self.dsac5 = nn.Conv1d(128, 128, filter_size, dilation=16)
        self.dsac6 = nn.Conv1d(128, 128, filter_size, dilation=32)

        self.tan_h = nn.Tanh()

        self.normalization_layer = nn.LayerNorm(1)

    def forward(self, nn_input):
        dsac1_output = self.dsac1(nn_input)
        layer_norm1 = self.normalization_layer(dsac1_output)
        tan_h1 = self.tan_h(layer_norm1)
        dsac2_output = self.dsac2(tan_h1)

        return dsac2_output


