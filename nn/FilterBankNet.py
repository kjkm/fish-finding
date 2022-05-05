import torch.nn as nn
import torch

class Filter_bank_net(nn.Module):

    def __init__(self,input_length):
        filter_size = 3
        super(Filter_bank_net, self).__init__()
        self.input_length = input_length
        self.filters = 128

        # a depthwise separable convolution is the combination of a depthwise 
        # convolution and a pointwise convolution
        # https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
        self.depth_conv_1 = nn.Conv1d(4, 4, filter_size, dilation=1, groups=4,padding=1)
        self.point_conv_1 = nn.Conv1d(4, self.filters, kernel_size=1)
        self.dsac_1 = nn.Sequential(self.depth_conv_1, self.point_conv_1)

        self.point_conv_generic = nn.Conv1d(self.filters, self.filters, kernel_size=1)

        self.depth_conv_2 = nn.Conv1d(self.filters, self.filters, filter_size, dilation=2,padding=2)
        self.dsac_2 = nn.Sequential(self.depth_conv_2, self.point_conv_generic)

        self.depth_conv_3 = nn.Conv1d(self.filters, self.filters, filter_size, dilation=4,padding=4)
        self.dsac_3 = nn.Sequential(self.depth_conv_3, self.point_conv_generic)

        self.depth_conv_4 = nn.Conv1d(self.filters, self.filters, filter_size, dilation=8,padding=8)
        self.dsac_4 = nn.Sequential(self.depth_conv_4, self.point_conv_generic)

        self.depth_conv_5 = nn.Conv1d(self.filters, self.filters, filter_size, dilation=16,padding=16)
        self.dsac_5 = nn.Sequential(self.depth_conv_5, self.point_conv_generic)

        self.depth_conv_6 = nn.Conv1d(self.filters, self.filters, filter_size, dilation=32,padding=32)
        self.dsac_6 = nn.Sequential(self.depth_conv_6, self.point_conv_generic)


        self.tan_h = nn.Tanh()

        self.normalization_layer = nn.LayerNorm([self.filters,self.input_length])



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
        dsac2_output = self.dsac_2(tan_h1)

        catted_12 = torch.cat((dsac1_output,dsac2_output),1)

        return catted_12


