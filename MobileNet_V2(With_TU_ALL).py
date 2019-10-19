from help_func.__init__ import ExecFileName
import os
ExecFileName.filename = os.path.splitext(os.path.basename(__file__))[0]

import torch

import torch.nn as nn

import torch.optim as optim

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

from CfgEnv.loadCfg import NetManager
from CfgEnv.loadData import DataBatch
from help_func.logging import LoggingHelper
from help_func.help_python import myUtil
from help_func.CompArea import LearningIndex
from help_func.CompArea import PictureFormat
from CfgEnv.loadData import TestDataBatch

from itertools import cycle
from visual_tool.Tensorboard import Mytensorboard
from help_func.help_torch_parallel import DataParallelModel, DataParallelCriterion
import numpy as np
import os
import math
logger = LoggingHelper.get_instance().logger
filename = os.path.basename(__file__)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_in_channels = _SplitChannels(in_channels, self.num_groups)
        self.split_out_channels = _SplitChannels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size[i],
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()

        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)

        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i]//2,
                groups=self.split_channels[i],
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)

        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x

class MixNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=[3],
        expand_ksize=[1],
        project_ksize=[1],
        stride=1,
        expand_ratio=1,
        non_linear='ReLU',
        se_ratio=0.0
    ):

        super(MixNetBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                GroupedConv2d(in_channels, expand_channels, expand_ksize),
                nn.BatchNorm2d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            GroupedConv2d(expand_channels, out_channels, project_ksize),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MixNet(nn.Module):
    mymixnet = [(24, 24, [3],       [1],    [1],  1, 1, 'ReLU', 0.0),
                (24, 24, [3,5,7],   [1,1], [1,1], 1, 3, 'ReLU', 0.0),
                (24, 24, [3],       [1,1], [1,1], 1, 3, 'ReLU', 0.0),
                (24, 24, [3,5,7,9], [1],    [1],  1, 3, 'Swish', 0.5),
                (24, 24, [3,5],     [1,1], [1,1], 1, 3, 'Swish', 0.5),
                (24, 24, [3,5,7], [1],      [1], 1, 3, 'Swish', 0.5),
                (24, 24, [3,5,7,9], [1],   [1,1], 1, 3, 'Swish', 0.5),
                (24, 24, [3,5,7,9], [1],   [1,1], 1, 3, 'Swish', 0.5)]


    def __init__(self, input_channel=3, output_channel=1, stem_channel=24, depth_multiplier=1.0):
        super(MixNet, self).__init__()
        config = self.mymixnet
        stem_channel=24
        depth_multiplier=1.0
        # dropout_rate = 0.25

        if depth_multiplier != 1.0:
            stem_channel = _RoundChannels(stem_channel*depth_multiplier)
        for i, conf in enumerate(config):
            conf_ls = list(conf)
            conf_ls[0] = _RoundChannels(conf_ls[0]*depth_multiplier)
            conf_ls[1] = _RoundChannels(conf_ls[1]*depth_multiplier)
            config[i] = tuple(conf_ls)

        self.stem_conv = Conv3x3Bn(input_channel, stem_channel, 1)
        layers = []
        for in_channels, out_channels, kernel_size, expand_ksize, \
            project_ksize, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(MixNetBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                expand_ksize=expand_ksize,
                project_ksize=project_ksize,
                stride=stride,
                expand_ratio=expand_ratio,
                non_linear=non_linear,
                se_ratio=se_ratio
            ))
        self.layers = nn.Sequential(*layers)
        self.last_conv = Conv3x3Bn(config[-1][1], output_channel, stride=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        return x



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

def _make_divisble(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+divisor/2)//divisor*divisor)
    if new_v < 0.9*v:
        new_v +=divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, ispad = True):
        if ispad:
            padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, input_dim = 3, output_dim = 1, width_mult = 1.0, inverted_residual_setting = None, round_nearest = 8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s - expand_ratio, output_channel, number_of_layers, stride
                [1, 24, 1, 1],
                [6, 24, 6, 1],
            ]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        input_channel =  _make_divisble(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(input_dim, input_channel, stride=1, ispad = False),
                    ConvBNReLU(input_channel, input_channel, stride=1, ispad = False)]


        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisble(c*width_mult, round_nearest)
            for i in range(n):
                stride = s if i==0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, output_dim, 3, stride=1, padding=1, bias=False))
        self.features = nn.Sequential(*features)

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


    def forward(self,x):
        x = self.features(x)
        return x


class COMMONDATASETTING():
    DATA_CHANNEL_NUM = 9
    OUTPUT_CHANNEL_NUM = 1
    qplist = [22,27,32,37]
    depthlist = [i for i in range(1,7)]
    modelist = [i for i in range(0,67)]
    translist = [0,2]
    mean = NetManager.cfg.DATAMEAN[:DATA_CHANNEL_NUM]
    std = NetManager.cfg.DATASTD[:DATA_CHANNEL_NUM]
    alflist = [i for i in range(0,17)]
    if not mean and not std:
        mean = 0
        std = 1
    else:
        mean = (np.array(list(mean))).reshape((len(mean), 1 , 1))
        std = (np.array(list(std))).reshape((len(std), 1, 1))
        std[std == 0] = 1



class _DataBatch(DataBatch, COMMONDATASETTING):
    def __init__(self, istraining, batch_size):
        DataBatch.__init__(self, istraining, batch_size)
        self.data_channel_num = COMMONDATASETTING.DATA_CHANNEL_NUM
        self.output_channel_num = COMMONDATASETTING.OUTPUT_CHANNEL_NUM
        self.data_padding = 2

    def getInputDataShape(self):
        return (self.batch_size, self.data_channel_num, self.batch[0][2], self.batch[0][1])


    def getOutputDataShape(self):
        return (self.output_channel_num, self.batch[0][2]- self.data_padding*2, self.batch[0][1] - self.data_padding*2)


    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
    def unpackData(self, info):
        DataBatch.unpackData(self, info)
        qpmap = self.tulist.getTuMaskFromIndex(0, info[2], info[1])
        modemap = self.tulist.getTuMaskFromIndex(1, info[2], info[1])
        modemap[np.all([modemap>1, modemap<34], axis = 0)] = 2
        modemap[modemap>=34] = 3
        depthmap = self.tulist.getTuMaskFromIndex(2, info[2], info[1])
        hortrans = self.tulist.getTuMaskFromIndex(3, info[2], info[1])
        vertrans = self.tulist.getTuMaskFromIndex(4, info[2], info[1])
        alfmap = self.ctulist.getTuMaskFromIndex(0, info[2], info[1])
        data = np.stack([*self.reshapeRecon(), qpmap, modemap, depthmap,hortrans,vertrans,alfmap], axis=0)
        # data = np.stack([*self.reshapeRecon(),qpmap], axis=0)
        gt = self.dropPadding(np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis=0), 2)
        recon = self.dropPadding(data[:self.output_channel_num], 2, isDeepCopy=True)
        data = (data - self.mean) / self.std
        recon /= 1023.0
        gt /= 1023.0
        gt -= recon
        return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def ReverseNorm(self, x, idx):
        mean = torch.from_numpy(np.array(self.mean[idx], dtype='float32'))
        std = torch.from_numpy(np.array(self.std[idx], dtype='float32'))
        return x * std + mean

class _TestSetBatch(TestDataBatch, COMMONDATASETTING):
    def __init__(self):
        TestDataBatch.__init__(self)
        self.data_channel_num = COMMONDATASETTING.DATA_CHANNEL_NUM
        self.output_channel_num = COMMONDATASETTING.OUTPUT_CHANNEL_NUM

    def unpackData(self, testFolderPath):
        TestDataBatch.unpackData(self, testFolderPath=testFolderPath)
        self.pic.setReshape1dTo2d(PictureFormat.RECONSTRUCTION)
        self.pic.setReshape1dTo2d(PictureFormat.ORIGINAL)
        qpmap = self.tulist.getTuMaskFromIndex(0, self.pic.area.height, self.pic.area.width)
        modemap = self.tulist.getTuMaskFromIndex(1, self.pic.area.height, self.pic.area.width)
        modemap[np.all([modemap>1, modemap<34], axis = 0)] = 2
        modemap[modemap>=34] = 3
        depthmap = self.tulist.getTuMaskFromIndex(2, self.pic.area.height, self.pic.area.width)
        hortrans = self.tulist.getTuMaskFromIndex(3, self.pic.area.height, self.pic.area.width)
        vertrans = self.tulist.getTuMaskFromIndex(4, self.pic.area.height, self.pic.area.width)
        alfmap = self.ctulist.getTuMaskFromIndex(0, self.pic.area.height, self.pic.area.width)
        # qpmap = np.full(qpmap.shape, qpmap[100,100])
        data = np.stack([*self.pic.pelBuf[PictureFormat.RECONSTRUCTION], qpmap, modemap, depthmap, hortrans,vertrans, alfmap], axis = 0)
        orig = np.stack([*self.pic.dropPadding(np.array(self.pic.pelBuf[PictureFormat.ORIGINAL][0])[np.newaxis,:,:], 2, isDeepCopy=False)])
        recon = self.dropPadding(data[:self.output_channel_num], pad=2, isDeepCopy=True)
        data = (data - self.mean) / self.std
        orig /= 1023.0
        recon /= 1023.0
        orig -= recon
        return self.cur_path,recon.astype('float32'), data.astype('float32'), orig.astype('float32')


class myDataBatch(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_channel_num = dataset.data_channel_num
        self.output_channel_num = dataset.output_channel_num

    def __getitem__(self, index):
        return self.dataset.unpackData(self.dataset.batch[index])

    def __len__(self):
        return len(self.dataset.batch)



if '__main__' == __name__:

    logger.info(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device %s" % device)
    dataset = _DataBatch(LearningIndex.TRAINING, _DataBatch.BATCH_SIZE)

    pt_dataset = myDataBatch(dataset=dataset)
    train_loader = DataLoader(dataset=pt_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=True,
                              num_workers=NetManager.NUM_WORKER)
    if NetManager.SET_NEW_MEAN_STD:
        dataset.loadMeanStd(train_loader, isGetNew=True)
        exit()

    valid_dataset = _DataBatch(LearningIndex.VALIDATION, _DataBatch.BATCH_SIZE)
    test_dataset = _TestSetBatch()
    pt_valid_dataset = myDataBatch(dataset=valid_dataset)
    pt_test_dataset = myDataBatch(dataset=test_dataset)

    valid_loader = DataLoader(dataset=pt_valid_dataset, batch_size=dataset.batch_size, drop_last=True,
                              shuffle=False, num_workers=NetManager.NUM_WORKER)
    test_loader = DataLoader(dataset=pt_test_dataset, batch_size=1, drop_last=True, shuffle=False, num_workers=0)
    # net = DenseNet(dataset.data_channel_num, 1, growth_rate=12, block_config=(4,4,4,4), drop_rate=0.2)
    iter_training = cycle(train_loader)
    iter_valid = cycle(valid_loader)
    iter_test = cycle(test_loader)

    # net = MobileNetV2(input_dim=dataset.data_channel_num, output_dim=1)
    net = MobileNetV2(dataset.data_channel_num, 1)
    # net.to(device)
    summary(net, (dataset.data_channel_num,132,132), device='cpu')
    cuda_device_count = torch.cuda.device_count()
    criterion = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    recon_MSE_loss = nn.MSELoss()
    if torch.cuda.is_available():
        recon_MSE_loss = recon_MSE_loss.cuda()
        if cuda_device_count > 1:
            net = DataParallelModel(net).cuda()
            criterion = DataParallelCriterion(criterion).cuda()
            MSE_loss = DataParallelCriterion(nn.MSELoss()).cuda()
        else:
            net.cuda()
            criterion.cuda()
            MSE_loss.cuda()

    optimizer = optim.Adam(net.parameters(), lr=dataset.cfg.INIT_LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=[int(dataset.cfg.OBJECT_EPOCH * 0.5),
                                                              int(dataset.cfg.OBJECT_EPOCH * 0.75)],
                                                  gamma=0.1, last_epoch=-1)

    object_step = dataset.batch_num * dataset.cfg.OBJECT_EPOCH
    tb = Mytensorboard(os.path.splitext(os.path.basename(__file__))[0])
    if NetManager.cfg.LOAD_SAVE_MODEL:
        PATH = './'+NetManager.MODEL_PATH + '/'+ os.path.splitext(os.path.basename(__file__))[0] +'_model.pth'
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        net.eval()

    logger.info('Training Start')

    for epoch_iter, epoch in enumerate(range(NetManager.OBJECT_EPOCH), 1):
        running_loss = 0.0
        for i in range(dataset.batch_num):
            (recons, inputs, gts) = next(iter_training)

            outputs = net(inputs)
            loss = criterion(outputs, gts)
            MSE = MSE_loss(outputs, gts)
            recon_MSE = torch.mean((gts) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += MSE.item()

            tb.SetLoss('CNN', MSE.item())
            tb.SetLoss('Recon', recon_MSE.item())
            tb.plotScalars()

            if i % dataset.PRINT_PERIOD == dataset.PRINT_PERIOD - 1:
                logger.info('[Epoch : %d, %5d/%d] loss: %.7f' %
                                    (epoch_iter,i+1,
                                     dataset.batch_num, running_loss / dataset.PRINT_PERIOD))
                running_loss = 0.0
            del recons, inputs, gts
            tb.step += 1  # Must Used

        mean_loss_cnn = 0
        mean_psnr_cnn = 0
        mean_loss_recon = 0
        mean_psnr_recon = 0
        cumsum_valid = torch.zeros(valid_dataset.getOutputDataShape()).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        for i in range(valid_dataset.batch_num):
            with torch.no_grad():
                (recons, inputs, gts) = next(iter_valid)
                outputs = net(inputs)
                loss = criterion(outputs, gts)
                recon_loss = torch.mean(torch.abs(gts))
                MSE = MSE_loss(outputs, gts)
                recon_MSE = torch.mean((gts) ** 2)
                mean_psnr_cnn += myUtil.psnr(MSE.item())
                mean_psnr_recon += myUtil.psnr(recon_MSE.item())
                mean_loss_cnn += loss.item()
                mean_loss_recon += recon_loss.item()
                if cuda_device_count > 1:
                    outputs = torch.cat(outputs, dim=0)
                cumsum_valid += (outputs**2).sum(dim=0)
                if i == 0:
                    tb.batchImageToTensorBoard(tb.Makegrid(recons), tb.Makegrid(outputs), 'CNN_Reconstruction')
                    tb.plotDifferent(tb.Makegrid(outputs), 'CNN_Residual')
                    if epoch_iter==1:
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[3], idx=3).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'QP_Map', [22, 37], 4)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[4], idx=4).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Mode_Map', [0, 3], 3)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[5], idx=5).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Depth_Map', [1, 6], 6)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[6], idx=6).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Hor_Trans', [0, 2], 2)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[7], idx=7).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Ver_Trans', [0, 2], 2)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[8], idx=8).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'ALF_IDX', [0, 16], 17)
                    logger.info("[epoch:%d] Finish Plot Image" % epoch_iter)
        cumsum_valid /= (valid_dataset.batch_num * valid_dataset.batch_size)
        tb.plotMSEImage(cumsum_valid, 'Error_Mean')
        logger.info('[epoch : %d] Recon_loss : %.7f, Recon_PSNR : %.7f' % (
        epoch_iter, mean_loss_recon / len(valid_loader), mean_psnr_recon / len(valid_loader)))
        logger.info('[epoch : %d] CNN_loss   : %.7f, CNN_PSNR :   %.7f' % (
        epoch_iter, mean_loss_cnn / len(valid_loader), mean_psnr_cnn / len(valid_loader)))
        torch.save({
            'epoch': epoch_iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'TensorBoardStep':tb.step
        }, NetManager.MODEL_PATH + '/'+ os.path.splitext(os.path.basename(__file__))[0] +'_model.pth')
        lr_scheduler.step()
        logger.info('Epoch %d Finished' % epoch_iter)

    MSE_loss = nn.MSELoss()
    recon_MSE_loss = nn.MSELoss()
    if torch.cuda.is_available():
        MSE_loss.cuda()
        recon_MSE_loss.cuda()
    mean_test_psnr = 0
    mean_testGT_psnr = 0
    for i in range(len(test_loader)):

        with torch.no_grad():
            (path, recons, inputs, gts) = next(iter_test)

            if torch.cuda.is_available():
                recons = recons.cuda()
                inputs = inputs.cuda()
                gts = gts.cuda()
            outputs = net(inputs)
            # if cuda_device_count > 1:
            #     outputs = torch.cat(outputs, dim=0)

            MSE = MSE_loss(outputs[0], gts)
            recon_MSE = torch.mean((gts) ** 2)
            logger.info('%s(%s) : %s %s' %(path, i, myUtil.psnr(MSE.item()), myUtil.psnr(recon_MSE.item())))
            mean_test_psnr += myUtil.psnr(MSE.item())
            mean_testGT_psnr += myUtil.psnr(recon_MSE.item())
    logger.info("%s %s" % (mean_test_psnr / len(test_loader), mean_testGT_psnr / len(test_loader)))