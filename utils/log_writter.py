import os 
from torch.utils.tensorboard import SummaryWriter

class log_writter:
    def __init__(self, path) -> None:

        self.writer = SummaryWriter(path)
        
    def add_train_loss_scalar(self, name, data, n):
        self.writer.add_scalar(name+"/loss/train", data ,n)

    def add_train_rew_scalar(self, name, data, n):
        self.writer.add_scalar(name+"/reward/train", data ,n)

    def add_test_loss_scalar(self, name, data, n):
        self.writer.add_scalar(name+"/loss/test", data ,n)

    def add_test_rew_scalar(self, name, data, n):
        self.writer.add_scalar(name+"/reward/test", data ,n)