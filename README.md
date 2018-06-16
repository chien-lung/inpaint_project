# inpaint_project
Using pytorch to implement [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)

`completionnet_places2.t7`  : torch(lua) model

`completionnet_places2.py`  : pytorch model

`completionnet_places2.pth` : pytorch model's param

Using [`convert_torch.py`](https://github.com/clcarwin/convert_torch_to_pytorch) to convert torch model(.t7) to pytorch model(.py, .pth)

Refer to [torch library](https://github.com/torch/torch7) and [gnuplot of torch](https://github.com/torch/gnuplot) to understand 
how to use`torch` and `lua`

Refet to [pyTorch](https://pytorch.org/docs/master/index.html) to understand how to implement by pyTorch
