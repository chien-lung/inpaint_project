# inpaint_project

## Overview

Using pytorch to implement [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)

Using [`convert_torch.py`](https://github.com/clcarwin/convert_torch_to_pytorch) to convert torch model(`.t7`) to pytorch model(`.py`,` .pth`):

   `completionnet_places2.t7`  : torch(lua) model 

   -> [`completionnet_places2.py`  : pytorch model , `completionnet_places2.pth` : pytorch model's parameters]

[`save_network.py`](https://github.com/chien-lung/inpaint_project/blob/master/save_network.py) : load the params from `completionnet_places2.pth` and save whole network in `inpaint.pkl`

## Usage

Basic usage is 

```
python inpaint.py --input <input_image> --mask <mask_image>
```

For example:

```
python inpaint.py --input example.png --mask example_mask.png
```

## Reference

[torch library](https://github.com/torch/torch7) and [gnuplot of torch](https://github.com/torch/gnuplot) : 
how to use`torch` and `lua`

[pyTorch](https://pytorch.org/docs/master/index.html) : how to implement torch model by pyTorch

## Future Work

Using [Python captcha.image.ImageCaptcha() Examples](https://www.programcreek.com/python/example/98386/captcha.image.ImageCaptcha) to train 1k pictures.
