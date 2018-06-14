import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
from collections import OrderedDict
model=nn.Sequential(
	nn.Conv2d(4,64,(5,5),stride=(1,1),padding=(2,2)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,128,(3,3),stride=(2,2),padding=(1,1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3,3),stride=(1,1),padding=(1,1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,256,(3,3),stride=(2,2),padding=(1,1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(1,1),dilation=(1,1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(1,1),dilation=(1,1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(2,2),dilation=(2,2)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(4,4),dilation=(4,4)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(8,8),dilation=(8,8)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(16,16),dilation=(16,16)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(1,1),dilation=(1,1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(1,1),dilation=(1,1)),
	nn.BatchNorm2d(256),
	nn.ReLU(),
	nn.ConvTranspose2d(256,128,(4,4),stride=(2,2),padding=(1,1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.Conv2d(128,128,(3,3),stride=(1,1),padding=(1,1)),
	nn.BatchNorm2d(128),
	nn.ReLU(),
	nn.ConvTranspose2d(128,64,(4,4),stride=(2,2),padding=(1,1)),
	nn.BatchNorm2d(64),
	nn.ReLU(),
	nn.Conv2d(64,32,(3,3),stride=(1,1),padding=(1,1)),
	nn.BatchNorm2d(32),
	nn.ReLU(),
	nn.Conv2d(32,3,(3,3),stride=(1,1),padding=(1,1)),
	nn.Sigmoid()
)

model.load_state_dict(torch.load('completionnet_places2.pth'))

#print(Inpaint)
#b=list(Inpaint.parameters())
#print(Inpaint.state_dict())

torch.save(model, "inpaint.pkl")
