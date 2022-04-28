import torch
import torch.nn as nn
from FilterBankNet import Filter_bank_net

class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_filter_bank = Filter_bank_net()

if __name__ == "__main__":
    my_trainer = Trainer()
    data = torch.zeros([1, 4, 1024])
    my_trainer.my_filter_bank(data)
