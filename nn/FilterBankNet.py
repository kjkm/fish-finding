import torch.nn as nn
import torch

class Filter_bank_net(nn.Module):

    def __init__(self, input_length):
        filter_size = 3
        super(Filter_bank_net, self).__init__()
        self.input_length = input_length
        self.filters = 128

        # a depthwise separable convolution is the combination of a depthwise 
        # convolution and a pointwise convolution
        # https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/
        self.depth_conv_1 = nn.Conv1d(4, 4, filter_size, dilation=1, groups=4, padding=1)
        self.point_conv_1 = nn.Conv1d(4, self.filters, kernel_size=1)
        self.dsac_1 = nn.Sequential(self.depth_conv_1, self.point_conv_1)

        self.point_conv_generic = nn.Conv1d(self.filters, self.filters, kernel_size=1)

        self.depth_conv_2 = nn.Conv1d(self.filters, self.filters, filter_size, dilation=2, padding=2)
        self.dsac_2 = nn.Sequential(self.depth_conv_2, self.point_conv_generic)

        self.depth_conv_3 = nn.Conv1d(2 * self.filters, self.filters, filter_size, dilation=4, padding=4)
        self.dsac_3 = nn.Sequential(self.depth_conv_3, self.point_conv_generic)

        self.depth_conv_4 = nn.Conv1d(2 * self.filters, self.filters, filter_size, dilation=8, padding=8)
        self.dsac_4 = nn.Sequential(self.depth_conv_4, self.point_conv_generic)

        self.depth_conv_5 = nn.Conv1d(2 * self.filters, self.filters, filter_size, dilation=16, padding=16)
        self.dsac_5 = nn.Sequential(self.depth_conv_5, self.point_conv_generic)

        self.depth_conv_6 = nn.Conv1d(2 * self.filters, self.filters, filter_size, dilation=32, padding=32)
        self.dsac_6 = nn.Sequential(self.depth_conv_6, self.point_conv_generic)

        self.tan_h = nn.Tanh()

        self.first_normalization = nn.LayerNorm([self.filters, self.input_length])
        self.normalization_layer = nn.LayerNorm([2 * self.filters, self.input_length])

        self.average_pool = nn.AvgPool1d(2)


    def forward(self, nn_input):
        # Layer 1
        dsac1_output = self.dsac_1(nn_input)  # Should produce 1D array
        layer_norm1 = self.first_normalization(dsac1_output)
        tan_h1 = self.tan_h(layer_norm1)

        # Layer 2
        dsac2_output = self.dsac_2(tan_h1)
        catted_12 = torch.cat((dsac1_output, dsac2_output), 1)
        layer_norm2 = self.normalization_layer(catted_12)
        tan_h2 = self.tan_h(layer_norm2)

        # Layer 3
        dsac3_output = self.dsac_3(tan_h2)
        catted_23 = torch.cat((dsac2_output, dsac3_output), 1)
        layer_norm3 = self.normalization_layer(catted_23)
        tan_h3 = self.tan_h(layer_norm3)

        # Layer 4
        dsac4_output = self.dsac_4(tan_h3)
        catted_34 = torch.cat((dsac3_output, dsac4_output), 1)
        layer_norm4 = self.normalization_layer(catted_34)
        tan_h4 = self.tan_h(layer_norm4)

        # Layer 5
        dsac5_output = self.dsac_5(tan_h4)
        catted_45 = torch.cat((dsac4_output, dsac5_output), 1)
        layer_norm5 = self.normalization_layer(catted_45)
        tan_h5 = self.tan_h(layer_norm5)

        # Layer 6
        dsac6_output = self.dsac_6(tan_h5)
        catted_56 = torch.cat((dsac5_output, dsac6_output), 1)
        layer_norm6 = self.normalization_layer(catted_56)
        tan_h6 = self.tan_h(layer_norm6)

        # Pseudo Energy
        square = torch.square(tan_h6)
        avg = self.average_pool(square[:, :, 30:991])

        # TODO: Make these do something
        energy_layer_norm = 0
        selu = 0

        return tan_h6


