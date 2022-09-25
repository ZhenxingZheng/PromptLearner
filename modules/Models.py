import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
import r2plus1d

class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.dataset = dataset

        if self.dataset == 'hmdb':
            self.num_class = 51
        elif self.dataset == 'ucf':
            self.num_class = 101
        elif self.dataset == 'kinetics':
            self.num_class = 400
        self.net = r2plus1d.r2plus1d_34_8_kinetics(num_classes=400)
        self.prepare_basemodel('./data/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth')


    def forward(self, x):
        output = self.net(x)

        return output

    def prepare_basemodel(self, pretrained_weights):
        state_dicts = torch.load(pretrained_weights)
        self.net.load_state_dict(state_dicts, strict=False)


if __name__ == '__main__':
    print('haha')
    fake_data = torch.randn(2, 3, 32, 112, 112)
    net = Net(dataset='kinetics')
    net = torch.nn.DataParallel(net).cuda()
    print(net)
    # net.load_state_dict(torch.load('/home/stormai/userfile/zhengzhenxing/IG-65M/model/2020-06-06 09_16_4523.pkl'), strict=True)
    print (net(fake_data).size())


