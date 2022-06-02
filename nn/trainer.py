import torch
import torch.nn as nn
from NeuralNetwork import Neural_net

class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_filter_bank = Neural_net(1024)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        neural_net = 1
        for s in list(p.size()):
            neural_net = neural_net*s
        pp += neural_net
    return pp


if __name__ == "__main__":
    my_trainer = Trainer()
    data = torch.zeros([1, 4, 1024])
    my_trainer.my_filter_bank(data)
    print(get_n_params(my_trainer.my_filter_bank))
