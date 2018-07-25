from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch
import argparse

#resize image size to multiple of 4
def img_resize(image):
    width = int(image.size[0]/4)*4
    height = int(image.size[1]/4)*4
    return image.resize((width,height),Image.BILINEAR)

#convert image to gray scale
def load_gray_img(image):
    image = image.convert('YCbCr')
    y,cb,cr = image.split()
    return y

#defined datas
datamean = torch.Tensor([0.45603489875793, 0.44721376895905, 0.41546452045441])
maxdim = 500

#input command
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input image to use')
parser.add_argument('--mask', type=str, required=True, help='mask to use')
opt = parser.parse_args()

#initialize the input image and mask
im = Image.open(opt.input)
mk = Image.open(opt.mask)
im = img_resize(im)
mk = img_resize(mk)
mk = load_gray_img(mk)

#convert image and mask to pytorch tensor
sub = ToTensor()
I = sub(im)
M = sub(mk)
M = M.ge(0.2).to(torch.float32)

#load network
model = torch.load("inpaint.pkl")
model.eval()
I_in = I.clone()

#image's RGB channel minus all training data's RGB mean
for i in range(3):
    I_in[i] = I[i].add(-datamean[i])

#draw the mask on the input image
I_in.masked_fill_( (M.to(torch.uint8).repeat(3,1,1)) , 0 )

#concat mask to I_in ; dimemsion: [3, height, width] -> [4, height, width]
input_ts = torch.cat( (I_in,M) ,0)
#reshape tensor dimesion: [4, height, width] -> [1, 4, height, width]
input_ts = torch.reshape(input_ts, (1, input_ts.size(0), input_ts.size(1), input_ts.size(2) ))

#resolution tensor dim:[1, 3, height, width]
res = model.forward(input_ts).to(torch.float32)[0]
oneMinusM = torch.mul(M,-1).add(1)   #1-M
#out = original image outside mask + resolution image inside mask
out = torch.mul(I,oneMinusM.repeat(3,1,1)).add(torch.mul(res,M.repeat(3,1,1)))

#save output and resolution image
I_out = out.clone()
I_img = ToPILImage()
out_pic = I_img(I_out)
out_pic.save('out.png')
out_pic = I_img(res)
out_pic.save('res.png')
