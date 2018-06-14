
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
from functools import reduce
from torch.autograd import Variable
'''
class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn
    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input
class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))
class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))
'''

completionnet_places2 = nn.Sequential( # Sequential,
	nn.Conv2d(4,64,(5, 5),(1, 1),(2, 2)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3),(2, 2),(1, 1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(2, 2),(2, 2),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(4, 4),(4, 4),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(8, 8),(8, 8),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(16, 16),(16, 16),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),(1, 1),1),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.ConvTranspose2d(256,128,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.ConvTranspose2d(128,64,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,32,(3, 3),(1, 1),(1, 1)),
	nn.BatchNorm2d(32),
	nn.ReLU(),
	nn.Conv2d(32,3,(3, 3),(1, 1),(1, 1)),
	nn.Sigmoid(),
)
completionnet_places2.load_state_dict(torch.load('completionnet_places2.pth'))

torch.save(completionnet_places2, "inpaint2.pkl")

