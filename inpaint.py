from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch


im = Image.open('example.png')
mk = Image.open('example_mask.png')

sub_I = ToTensor()
I = sub_I(im)
sub_M = ToTensor()
M = sub_M(mk)



model = torch.load("inpaint.pkl")
model.eval()
datamean = torch.Tensor([0.45603489875793, 0.44721376895905, 0.41546452045441])
I_in = I.clone()
I1 = I.clone()
for i in range(3):
    I1[i] = torch.gt( I[i].add(-datamean[i]) , 0 )
    I_in[i] = torch.mul( I[i].add(-datamean[i]) , I1[i] )

I_in.masked_fill_( (M.to(torch.uint8).repeat(3,1,1)) , 0 )
#I_img = ToPILImage()
#I_img(I_in).show()

input_ts = torch.cat( (I_in,M) ,0)
input_ts = torch.reshape(input_ts, (1, input_ts.size(0), input_ts.size(1), input_ts.size(2) ))

res = model.forward(input_ts).to(torch.float32)[0]
print(res.size())
res_sub = res.clone()
res_img = ToPILImage()
#res_img(res).show()
'''
for i in range(3):
    res_sub[i] = torch.lt( res[i] , 1 )
    res[i] = torch.mul( res_sub[i] , res[i] )
'''
#print(res)
#res_img = ToPILImage()
res_img(res).show()

minusM = torch.mul(M,-1).add(1)
out = torch.mul(I,minusM.repeat(3,1,1)).add(torch.mul(res,M.repeat(3,1,1)))

I_out = out.clone()


I_img = ToPILImage()
I_img(I_out).show()

#plt.plot(res[0].detach().numpy())
#plt.show()
plt.plot(res[1].detach().numpy())
plt.show()
#plt.plot(res[2].detach().numpy())
#plt.show()
