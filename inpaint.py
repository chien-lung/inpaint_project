from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch
import argparse

def img_resize(image):
    width = int(image.size[0]/4)*4
    height = int(image.size[1]/4)*4
    return image.resize((width,height),Image.BILINEAR)

def load_gray_img(image):
    image = image.convert('YCbCr')
    y,cb,cr = image.split()
    return y

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input image to use')
parser.add_argument('--mask', type=str, required=True, help='mask to use')
opt = parser.parse_args()

sub_M = ToTensor()
maxdim = 500
im = Image.open(opt.input)
mk = Image.open(opt.mask)
im = img_resize(im)
mk = img_resize(mk)

mk = load_gray_img(mk)

sub = ToTensor()
I = sub(im)
M = sub(mk)
M = M.ge(0.2).to(torch.float32)

model = torch.load("inpaint.pkl")
model.eval()
datamean = torch.Tensor([0.45603489875793, 0.44721376895905, 0.41546452045441])
I_in = I.clone()

for i in range(3):
    I_in[i] = I[i].add(-datamean[i])

I_in.masked_fill_( (M.to(torch.uint8).repeat(3,1,1)) , 0 )

input_ts = torch.cat( (I_in,M) ,0)
input_ts = torch.reshape(input_ts, (1, input_ts.size(0), input_ts.size(1), input_ts.size(2) ))

res = model.forward(input_ts).to(torch.float32)[0]
minusM = torch.mul(M,-1).add(1)
out = torch.mul(I,minusM.repeat(3,1,1)).add(torch.mul(res,M.repeat(3,1,1)))

I_out = out.clone()

I_img = ToPILImage()
#I_img(I_out).show()
out_pic = I_img(I_out)
out_pic.save('out.png')

