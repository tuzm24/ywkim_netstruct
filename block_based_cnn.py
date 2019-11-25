import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from help_func.ptflops import get_model_complexity_info
import torch
from collections import OrderedDict
import copy
import time
from help_func.help_torch import myUtil
from PIL import Image
from torchsummary import summary
pad_require = 10
input_height = 1080
input_width = 1920
batch_size = 64
channel_num = 3
block_size = 128
iscuda = 1


def invest_net_pad(model, input_size, batch_size=-1, set_zero=0, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            if class_name == 'Conv2d':
                if module.kernel_size[0] != 1 and module.padding[0] == 0:
                    no_height_pad.append((module.kernel_size[0] + 0.5) // 2)
                if module.kernel_size[1] != 1 and module.padding[1] == 0:
                    no_width_pad.append((module.kernel_size[1] + 0.5) // 2)
                h, w = module.padding
                width_pad.append(w)
                height_pad.append(h)
                if set_zero:
                    module.padding = (0, 0)
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


def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
        nn.ReLU(inplace=True)
    )


class Conv3x3_Recycle(nn.Module):
    def __init__(self, in_channels, out_channels, nth_layer):
        super(Conv3x3_Recycle, self).__init__()
        self.featuremap_size = block_size
        self.upper_buffer = {}
        for x_pos in range(0, input_width, block_size):
            self.upper_buffer[x_pos] = torch.zeros((batch_size, channel_num, 2, self.featuremap_size))
        self.left_buffer = torch.zeros((batch_size, channel_num, self.featuremap_size + 2, 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x_pos, y_pos):
        if y_pos == 0:
            x = F.pad(x, [0, 0, 1, 0])
        else:
            x = torch.cat((self.upper_buffer[x_pos], x), 2)
        self.upper_buffer[x_pos] = x[:, :, -2:, :]
        if x_pos == 0:
            x = F.pad(x, [1, 0, 0, 0])
        else:
            x = torch.cat((self.left_buffer, x), 3)
        self.left_buffer = x[:, :, :, -2:]

        return self.relu(self.conv(x))


class PlainNetwork(nn.Sequential):
    def __init__(self, num_input_features, output_dim):
        super(PlainNetwork, self).__init__()
        convlist = []
        convlist.append(Conv3x3Bn(num_input_features, 32, 1, 'ReLU', 1))
        convlist.append(Conv3x3Bn(32, 32, 1, 'ReLU', 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, 32, 1))
        convlist.append(Conv3x3Bn(32, output_dim, 1))
        self.layers = nn.Sequential(*convlist)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        return x


class PlainNetwork_nopad(nn.Sequential):
    def __init__(self, num_input_features, output_dim):
        super(PlainNetwork_nopad, self).__init__()
        convlist = []
        convlist.append(Conv3x3Bn(num_input_features, 32, 1, 'ReLU', 0))
        for i in range(8):
            convlist.append(Conv3x3Bn(32, 32, 1, 'ReLU', 0))

        convlist.append(Conv3x3Bn(32, output_dim, 1, 'ReLU', 0))
        self.layers = nn.Sequential(*convlist)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(x)
        return x


class PlainNetwork_nopad_recycle(nn.Module):
    def __init__(self, num_input_features, output_dim):
        super(PlainNetwork_nopad_recycle, self).__init__()
        convlist = []
        convlist.append(Conv3x3_Recycle(num_input_features, 32, 0))
        convlist.append(Conv3x3_Recycle(32, 32, 1))
        convlist.append(Conv3x3_Recycle(32, 32, 2))
        convlist.append(Conv3x3_Recycle(32, 32, 3))
        convlist.append(Conv3x3_Recycle(32, 32, 4))
        convlist.append(Conv3x3_Recycle(32, 32, 5))
        convlist.append(Conv3x3_Recycle(32, 32, 6))
        convlist.append(Conv3x3_Recycle(32, 32, 7))
        convlist.append(Conv3x3_Recycle(32, 32, 8))
        convlist.append(Conv3x3_Recycle(32, output_dim, 9))
        self.layers = nn.ModuleList(convlist)
        self.layers.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x, x_pos, y_pos):
        for layer in self.layers:
            x = layer(x, x_pos, y_pos)
        return x


def Copy_and_divide_Recycle(inputs, net, pad_required=10, devide_size=64):
    b, c, h, w = inputs.shape
    pos_list = [
        np.array((devide_size if (x + devide_size) <= w else (w - x),
                  devide_size if (y + devide_size) <= h else (h - y),
                  x, y))
        for y in range(0, h, devide_size)
        for x in range(0, w, devide_size)]
    pos_list = np.array(pos_list)

    pos_list[:, 0] += pos_list[:, 2]
    pos_list[:, 1] += pos_list[:, 3]

    output = torch.zeros(b, c, h - pad_require, w - pad_require)
    if iscuda:
        output = output.cuda()
    # output = np.full(inputs.shape, np.nan, dtype='float64')
    # output = np.full(inputs.shape, np.nan)
    # temp_input = copy.deepcopy(inputs)

    for dx, dy, x, y in pos_list:
        temp = net(inputs[:, :, y:dy, x:dx], x, y)
        if y:
            y -= pad_require
        dy -= pad_require
        if x:
            x -= pad_require
        dx -= pad_require
        output[:, :, y:dy, x:dx] = temp
        # output[:,:, y:dy-two_devide_size, x] = 2
        # output[:,:,y, x:dx-two_devide_size] = 2

    return output


def Calc_Copy_and_divide_Recycle(inputs, net, pad_required=10, devide_size=64):
    b, c, h, w = inputs.shape
    pos_list = [
        np.array((devide_size if (x + devide_size) <= w else (w - x),
                  devide_size if (y + devide_size) <= h else (h - y),
                  x, y))
        for y in range(0, h, devide_size)
        for x in range(0, w, devide_size)]
    pos_list = np.array(pos_list)

    # pos_list[:, 0] += pos_list[:, 2]
    # pos_list[:, 1] += pos_list[:, 3]


    # output = np.full(inputs.shape, np.nan, dtype='float64')
    # output = np.full(inputs.shape, np.nan)
    # temp_input = copy.deepcopy(inputs)
    flops = 0
    for dx, dy, x, y in pos_list:
        temp, _ = get_model_complexity_info(net, (channel_num, dy, dx), pos=(0,0), as_strings=False, print_per_layer_stat=False)
        flops += temp

    return flops

def Copy_and_divide_inputs(inputs, net, pad_required=10, devide_size=64):
    b, c, h, w = inputs.shape
    two_devide_size = pad_required * 2
    pos_list = [
        np.array((devide_size if (x + devide_size) <= w else (w - x),
                  devide_size if (y + devide_size) <= h else (h - y),
                  x, y))
        for y in range(0, h, devide_size)
        for x in range(0, w, devide_size)]
    pos_list = np.array(pos_list)
    pos_list[pos_list[:, 2] == 0, 0] += pad_required
    pos_list[pos_list[:, 3] == 0, 1] += pad_required
    pos_list[pos_list[:, 2] != 0, 0] += 2 * pad_required
    pos_list[pos_list[:, 2] != 0, 2] -= pad_required
    pos_list[pos_list[:, 3] != 0, 1] += 2 * pad_required
    pos_list[pos_list[:, 3] != 0, 3] -= pad_required
    # pos_list[pos_list[:, 3] == max(pos_list[:, 3]), 1] += pad_required
    # pos_list[pos_list[:, 2] == max(pos_list[:, 2]), 0] += pad_required
    pos_list[:, 0] += pos_list[:, 2]
    pos_list[:, 1] += pos_list[:, 3]

    output = torch.zeros(b, c, h-pad_require, w-pad_require)
    if iscuda:
        output = output.cuda()
    # output = np.full(inputs.shape, np.nan)
    img = F.pad(inputs, [pad_required, 0, pad_required, 0])
    # temp_input = copy.deepcopy(inputs)
    for dx, dy, x, y in pos_list:
        temp = net(img[:, :, y:dy, x:dx])
        output[:, :, y:dy - two_devide_size, x:dx - two_devide_size] = temp
        # output[:,:, y:dy-two_devide_size, x] = 2
        # output[:,:,y, x:dx-two_devide_size] = 2
    return output


def Calc_Copy_and_divide_inputs(inputs, net, pad_required=10, devide_size=64):
    b, c, h, w = inputs.shape
    two_devide_size = pad_required * 2
    pos_list = [
        np.array((devide_size if (x + devide_size) <= w else (w - x),
                  devide_size if (y + devide_size) <= h else (h - y),
                  x, y))
        for y in range(0, h, devide_size)
        for x in range(0, w, devide_size)]
    pos_list = np.array(pos_list)
    pos_list[pos_list[:, 2] == 0, 0] += pad_required
    pos_list[pos_list[:, 3] == 0, 1] += pad_required
    pos_list[pos_list[:, 2] != 0, 0] += 2 * pad_required
    pos_list[pos_list[:, 2] != 0, 2] -= pad_required
    pos_list[pos_list[:, 3] != 0, 1] += 2 * pad_required
    pos_list[pos_list[:, 3] != 0, 3] -= pad_required
    # pos_list[pos_list[:, 3] == max(pos_list[:, 3]), 1] += pad_required
    # pos_list[pos_list[:, 2] == max(pos_list[:, 2]), 0] += pad_required
    # pos_list[:, 0] += pos_list[:, 2]
    # pos_list[:, 1] += pos_list[:, 3]

    # output = torch.zeros(b, c, h-pad_require, w-pad_require)
    # if iscuda:
    #     output = output.cuda()
    # output = np.full(inputs.shape, np.nan)
    img = F.pad(inputs, [pad_required, pad_required, pad_required, pad_required])
    # temp_input = copy.deepcopy(inputs)
    flops = 0.0
    for dx, dy, x, y in pos_list:
        temp, _ = get_model_complexity_info(net, (channel_num, dy, dx), pos=None, as_strings=False, print_per_layer_stat=False)
        flops += temp
    return flops



def basic_Copy_and_devide_input(inputs, net, pad_required=1, devide_size=64):
    b, c, h, w = inputs.shape
    two_devide_size = pad_required * 2
    pos_list = [
        np.array((devide_size if (x + devide_size) <= w else (w - x),
                  devide_size if (y + devide_size) <= h else (h - y),
                  x, y))
        for y in range(0, h, devide_size)
        for x in range(0, w, devide_size)]
    pos_list = np.array(pos_list)
    pos_list[:, 0] += pad_required * 2
    pos_list[:, 1] += pad_required * 2
    pos_list[:, 0] += pos_list[:, 2]
    pos_list[:, 1] += pos_list[:, 3]

    output = torch.zeros(b, c, h - two_devide_size, w - two_devide_size)
    # output = np.full(inputs.shape, np.nan)
    img = F.pad(inputs, [pad_required, pad_required, pad_required, pad_required])
    for dx, dy, x, y in pos_list:
        output[:, :, y:dy - two_devide_size, x:dx - two_devide_size] = net(img[:, :, y:dy, x:dx])
    return output


torch.manual_seed(42)
with torch.no_grad():
    inputs = torch.randn(1, 3, input_height, input_width)
    inputs = torch.autograd.Variable(inputs, requires_grad=False)

    net = PlainNetwork(3, 3)
    net.eval()

    net_nopad = PlainNetwork_nopad(3, 3)
    net_nopad.eval()
    # print(invest_net_pad(net,(3,100,100)))

    net_recycle = PlainNetwork_nopad_recycle(3, 3)
    net_recycle.eval()

    if iscuda:
        net.cuda()
        net_nopad.cuda()
        net_recycle.cuda()
        inputs = inputs.cuda()
    # flops, params = get_model_complexity_info(net_recycle, (channel_num, input_height, input_width), pos=(0,0),as_strings=False, print_per_layer_stat=False)
    # print(flops)
    # flops = Calc_Copy_and_divide_Recycle(copy.deepcopy(inputs), net_recycle, devide_size=block_size)
    # print(flops)
    # flops = Calc_Copy_and_divide_inputs(copy.deepcopy(inputs), net_nopad,pad_required=pad_require,devide_size=block_size)
    flops = get_model_complexity_info(net_nopad, (channel_num, input_height, input_width), pos=None, as_strings=False, print_per_layer_stat=False)
    print(flops)


    # startTime = time.time()
    # c = Copy_and_divide_Recycle(copy.deepcopy(inputs), net_recycle, devide_size=block_size)
    #
    # print(time.time() - startTime)
    #
    # startTime = time.time()
    # # b = net_nopad(F.pad(copy.deepcopy(inputs), [pad_require, pad_require, pad_require ,pad_require]))
    # # b = net_nopad(copy.deepcopy(inputs))
    # # b = net(copy.deepcopy(inputs))
    # b = net_recycle(copy.deepcopy(inputs), 0, 0)
    #
    # # c = net_nopad(F.pad(copy.deepcopy(inputs), [10, 10, 10 ,10]))
    # # print(torch.all(torch.eq(b, c)))
    # print(time.time() - startTime)
    # # invest_net_pad(net,(3,100,100), batch_size=-1, set_zero=1, device='cpu')
    #
    # import timeit
    # print(net_nopad)
    # start = timeit.default_timer()
    # a = Copy_and_divide_inputs(copy.deepcopy(inputs), net_nopad, pad_required=pad_require,devide_size=block_size)
    # stop = timeit.default_timer()
    # print(stop - start)

    # start = timeit.default_timer()
    # ct = net_nopad(F.pad(inputs, [pad_require, 0, pad_require, 0]))
    # stop = timeit.default_timer()
    # print(stop - start)

    # print('오차 : %s' %torch.abs(a-b).sum())
    # print('PSNR : %s' %myUtil.psnr((torch.abs(a-b)**2).mean()))
    # print('오차 : %s' % torch.abs(c - b).sum())
    #
    # # for i in range(3):
    # #     mat = torch.eq(c,b).detach().numpy()[0,i]
    # #     mat = np.clip(mat, 0, 255)
    # #     # for i in range(0, 255, 64):
    # #     #     for j in range(0, 255, 64):
    # #     #         mat[i:i+64, j] = 0.0
    # #     #         mat[i, j:j+64] = 0.0
    # #     # reshape to 2d
    # #
    # #     # Creates PIL image
    # #     img = Image.fromarray(np.uint8(mat * 128) , 'L')
    # #     img.show()
    # # flops, params = get_model_complexity_info(net, (3, 1920, 1080), as_strings=True, print_per_layer_stat=True)
    # # print('Flops:  ' + flops)
    # # print('Params: ' + params)