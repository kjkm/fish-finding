import torch.nn as nn

class Filter_bank_net(nn.Module):

    def __init__(self):
        filter_size = 3
        super(Filter_bank_net, self).__init__()

        # a depthwise separable convolution is the combination of a depthwise 
        # convolution and a pointwise convolution
        # https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
        self.depth_conv_1 = nn.Conv1d(4, 4, filter_size, dilation=1, groups=4)
        self.point_conv_1 = nn.Conv1d(4, 128, kernel_size=1)
        self.dsac_1 = nn.Sequential(self.depth_conv_1, self.point_conv_1)

        self.dsac2 = nn.Conv1d(128, 128, filter_size, dilation=2)
        self.dsac3 = nn.Conv1d(128, 128, filter_size, dilation=4)
        self.dsac4 = nn.Conv1d(128, 128, filter_size, dilation=8)
        self.dsac5 = nn.Conv1d(128, 128, filter_size, dilation=16)
        self.dsac6 = nn.Conv1d(128, 128, filter_size, dilation=32)

        self.tan_h = nn.Tanh()

        self.normalization_layer = nn.LayerNorm(1)

    def forward(self, nn_input):
        dsac1_output = self.dsac_1(nn_input) # Should produce 1D array

        # dsac1_output = nn.Flatten(dsac1_output) <- this returns a Flatten object
        # https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch

        # Error from 4/28/22
        # Given normalized_shape=[1], expected input with shape [*, 1], but got 
        # input of size[1, 128, 1022]
        # Kieran's idea: drop some dimensions?
        # Anastasia's idea: flatten it? (kieran says yeah)
        layer_norm1 = self.normalization_layer(dsac1_output)
        tan_h1 = self.tan_h(layer_norm1)
        dsac2_output = self.dsac2(tan_h1)

        return dsac2_output


