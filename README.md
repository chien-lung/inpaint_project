# inpaint_project

## Overview

Using pytorch to implement [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)

`completionnet_places2.t7`  : torch(lua) model

`completionnet_places2.py`  : pytorch model

`completionnet_places2.pth` : pytorch model's parameters

Using [`convert_torch.py`](https://github.com/clcarwin/convert_torch_to_pytorch) to convert torch model(.t7) to pytorch model(.py, .pth)

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
