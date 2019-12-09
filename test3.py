import torchvision.models as models
import torch
from help_func.ptflops import get_model_complexity_info
#
# with torch.cuda.device(0):
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def invest_net_pad(model, input_size, batch_size=-1, set_zero = 0,device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            if class_name == 'Conv2d':
                if module.kernel_size[0] != 1 and module.padding[0] == 0:
                    no_height_pad.append((module.kernel_size[0]+0.5)//2)
                if module.kernel_size[1] != 1 and module.padding[1] == 0:
                    no_width_pad.append((module.kernel_size[1]+0.5)//2)
                h, w = module.padding
                width_pad.append(w)
                height_pad.append(h)
                if set_zero:
                    module.padding = (0,0)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    width_pad = []
    height_pad = []
    no_height_pad = []
    no_width_pad = []
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    return int(np.sum(height_pad)), int(np.sum(width_pad)), int(np.sum(no_height_pad)), int(np.sum(no_width_pad))



def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class PlainNetwork(nn.Sequential):
    def __init__(self, num_input_features, output_dim):
        super(PlainNetwork, self).__init__()
        convlist = []
        convlist.append(Conv3x3Bn(num_input_features, 32, 1, 'ReLU',0))
        convlist.append(Conv3x3Bn(32, 32, 1, 'ReLU',0))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,32,1))
        convlist.append(Conv3x3Bn(32,output_dim,1))
        self.layers = nn.Sequential(*convlist)

    def forward(self, x):
        x = self.layers(x)
        return x
net = PlainNetwork(3, 1)
net.cpu()
print(invest_net_pad(net,(3,250,250), set_zero=1))# flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print(invest_net_pad(net,(3,250,250)))
# print('Flops:  ' + flops)
# print('Params: ' + params)