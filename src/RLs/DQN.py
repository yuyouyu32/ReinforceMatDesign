import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self,d_input,d_output, layer1=64, layer2=64):
        '''
        :param d_input: 输入state的维度
        :param d_output: 输出action的个数（其值为Q(s,a）)
        :param layer1: DQN网络有两层隐藏层都是64维的
        :param layer2: 隐藏层第二层
        '''
        super(Net,self).__init__()
        self.fc1 = nn.Linear(d_input,layer1)
        # self.fc2 = nn.Linear(layer1,layer2)
        self.out = nn.Linear(layer1, d_output)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = self.fc2(x)
        output = self.out(x)

        return output

if __name__ == '__main__':
    net = Net(4,2,8,8)
    state = np.array([[1,2,3,4],[5,6,7,8]],dtype=np.float)
    if type(state) == type(np.array([1])):
        print('TRUE')
    output = net(state)
    print(output)
    print(output.max(1)[0])