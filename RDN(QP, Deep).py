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
import torch.nn.functional as F
logger = LoggingHelper.get_instance().logger
filename = os.path.basename(__file__)
from collections import OrderedDict
import copy
from collections import namedtuple
import torch.utils.checkpoint as cp
from help_func.warmup_scheduler import GradualWarmupScheduler
from help_func.help_torch import NetTrainAndTest


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    return torch.cat((x, F.relu(self.conv(x))), 1)


def _function_factory(relu, conv):
    def _function(*inputs):
        return relu(conv(torch.cat(inputs, 1)))
    return _function

# Residual dense block (RDB) architecture

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _function_factory(self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(bottleneck_output))
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, bn = False):
        super(_Transition, self).__init__()
        if bn:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=2, padding=1, bias=False))


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, memory_efficient=False):
        super(_ResidualDenseBlock, self).__init__()
        nchannels = num_input_features
        for i in range(num_layers):
            layer = _DenseLayer(
                nchannels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                memory_efficient=memory_efficient,
            )
            nchannels += growth_rate
            self.add_module('denselayer%d' % (i + 1), layer)
        self.add_module('1x1conv', nn.Conv2d(nchannels, num_input_features, kernel_size=1, padding=0, bias=False))
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name == '1x1conv':
                new_features = layer(torch.cat(features, 1))
                break
            new_features = layer(*features)
            features.append(new_features)
        # return torch.cat(features, 1)
        return init_features + new_features






class ResidualEncoder(nn.Module):

    def __init__(self, num_input_channel=1, num_output_channel=64, bn_size=4, growth_rate=32, memory_efficient=False,
                 middle_output_channel=32, block_config = (4,4,4,4,4)):
        super(ResidualEncoder, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_channel, middle_output_channel, kernel_size=5, stride=2, padding=2, bias=False)),
            ('norm0', nn.BatchNorm2d(middle_output_channel)),
            ('relu0', nn.ReLU(inplace=False)),
            ('conv1', nn.Conv2d(middle_output_channel, middle_output_channel, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(middle_output_channel)),
            ('relu1', nn.ReLU(inplace=False))
        ]))
        for i, num_layers in enumerate(block_config):
            block = _ResidualDenseBlock(
                num_layers=num_layers,
                num_input_features=middle_output_channel,
                bn_size=bn_size,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' %(i+1), block)
            num_features = middle_output_channel
            if i!=len(block_config)-1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features)
                self.features.add_module('transition%d' %(i+1), trans)

        self.channel_wise_features = nn.Sequential()
        self.channel_wise_features.add_module('c_GobalAveragePooling', nn.AdaptiveAvgPool2d((1,1)))
        self.channel_wise_features.add_module('c_Linear1', nn.Conv2d(middle_output_channel, middle_output_channel*2, 1,1,0,bias=True))
        self.channel_wise_features.add_module('c_relu1', nn.ReLU(inplace=False))
        self.channel_wise_features.add_module('c_Linear2', nn.Conv2d(middle_output_channel*2, middle_output_channel*2, 1,1,0,bias=True))
        self.channel_wise_features.add_module('c_relu2', nn.ReLU(inplace=False))
        self.channel_wise_features.add_module('c_Linear3', nn.Conv2d(middle_output_channel*2, num_output_channel, 1,1,0,bias=True))

        # self.spatial_wise_features = nn.Sequential()
        # self.spatial_wise_features.add_module('s_1x1conv1', nn.Conv2d(middle_output_channel, middle_output_channel*2, 1,1,0, bias=True))
        # self.spatial_wise_features.add_module('s_relu1', nn.ReLU(inplace=False))
        # self.spatial_wise_features.add_module('s_1x1conv2', nn.Conv2d(middle_output_channel*2, middle_output_channel*2, 1,1,0, bias=True))
        # self.spatial_wise_features.add_module('s_relu2', nn.ReLU(inplace=False))
        # self.spatial_wise_features.add_module('s_1x1conv3', nn.Conv2d(middle_output_channel*2, 1, 1,1,0, bias=True))
        # self.features.add_module('relu2', nn.ReLU(inplace=False))
        self.features.apply(self._init_weights)
        self.channel_wise_features.apply(self._init_weights)
        # self.spatial_wise_features.apply(self._init_weights)
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

    def forward(self, x):
        # x = self.features(x)
        # return self.channel_wise_features(x), self.spatial_wise_features(x)
        return self.channel_wise_features(self.features(x))
# Residual Dense Network

class GroupOfResidualDenseBlock(nn.Module):
    def __init__(self, num_input_channel, num_output_channel, bn_size=4, growth_rate=32, memory_efficient=False,
                 block_config=(8,10,8)):
        super(GroupOfResidualDenseBlock, self).__init__()
        for i, num_layers in enumerate(block_config):
            layer = _ResidualDenseBlock(
                num_layers=num_layers,
                num_input_features=num_input_channel,
                bn_size=bn_size,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient
            )
            self.add_module('residualdenselayer%d' % (i+1), layer)
        self.add_module('group_1x1conv', nn.Conv2d(num_input_channel*len(block_config), num_output_channel, kernel_size=1, padding=0, bias=False))

    def forward(self, init_features):
        features = []
        new_features = init_features
        for name, layer in self.named_children():
            if name=='group_1x1conv':
                new_features = layer(torch.cat(features,1))
                break
            new_features = layer(new_features)
            features.append(new_features)
        return init_features + new_features


class PixToPixReidualDenseNet(nn.Module):

    def __init__(self, num_input_channel=3, num_output_channel=1, bn_size=4, growth_rate=24, memory_efficient=False,
                 middle_output_channel=32, block_config=(4,4,4,4), additional_vector_channels = 64):
        super(PixToPixReidualDenseNet, self).__init__()
        self.preprocess = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_channel, middle_output_channel, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu0', nn.ReLU(inplace=False)),
            ('conv1', nn.Conv2d(middle_output_channel, additional_vector_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu0', nn.ReLU(inplace=False))
        ]))
        # F-1
        self.features = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(additional_vector_channels, middle_output_channel, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu2', nn.ReLU(inplace=False))
        ]))
        self.features.add_module('GroupOfResidualDenseBlock', GroupOfResidualDenseBlock(middle_output_channel, middle_output_channel,
                                                                                        bn_size=bn_size, growth_rate=growth_rate,
                                                                                        memory_efficient=memory_efficient, block_config=block_config))
        self.features.add_module('Finalconv', nn.Conv2d(middle_output_channel, num_output_channel, kernel_size=3,
                                                        stride=1, padding=1, bias=False))
        self.preprocess.apply(self._init_weights)
        self.features.apply(self._init_weights)


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

    def forward(self, x):
        # x = self.preprocess(x)
        return self.features(self.preprocess(x))

class COMMONDATASETTING():
    DATA_CHANNEL_NUM = 4
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
        self.data_padding = 0

    def getInputDataShape(self):
        return (self.batch_size, self.data_channel_num, self.batch[0][2], self.batch[0][1])


    def getOutputDataShape(self):
        return (self.output_channel_num, self.batch[0][2]- self.data_padding*2, self.batch[0][1] - self.data_padding*2)


    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
    # def unpackData(self, info):
    #     DataBatch.unpackData(self, info)
    #     qpmap = self.tulist.getTuMaskFromIndex(0, info[2], info[1])
    #     # modemap = self.tulist.getTuMaskFromIndex(1, info[2], info[1])
    #     # modemap[np.all([modemap>1, modemap<34], axis = 0)] = 2
    #     # modemap[modemap>=34] = 3
    #     # depthmap = self.tulist.getTuMaskFromIndex(2, info[2], info[1])
    #     # hortrans = self.tulist.getTuMaskFromIndex(3, info[2], info[1])
    #     # vertrans = self.tulist.getTuMaskFromIndex(4, info[2], info[1])
    #     # alfmap = self.ctulist.getTuMaskFromIndex(0, info[2], info[1])
    #     data = np.stack([*self.reshapeRecon(), qpmap], axis=0)
    #     # data = np.stack([*self.reshapeRecon(),qpmap], axis=0)
    #     gt = self.dropPadding(np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis=0), 2)
    #     recon = self.dropPadding(data[:self.output_channel_num], 2, isDeepCopy=True)
    #     data = (data - self.mean) / self.std
    #     recon /= 1023.0
    #     gt /= 1023.0
    #     gt -= recon
    #     return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def unpackData(self, info):
        DataBatch.unpackData(self, info)
        qpmap = self.tulist.getTuMaskFromIndex(0, info[2], info[1])
        # modemap = self.tulist.getTuMaskFromIndex(1, info[2], info[1])
        # modemap[np.all([modemap>1, modemap<34], axis = 0)] = 2
        # modemap[modemap>=34] = 3
        # depthmap = self.tulist.getTuMaskFromIndex(2, info[2 ], info[1])
        # hortrans = self.tulist.getTuMaskFromIndex(3, info[2], info[1])
        # vertrans = self.tulist.getTuMaskFromIndex(4, info[2], info[1])
        # alfmap = self.ctulist.getTuMaskFromIndex(0, info[2], info[1])
        data = np.stack([*self.reshapeRecon(), qpmap], axis=0)
        # data = np.stack([*self.reshapeRecon(),qpmap], axis=0)
        gt = (np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis=0))
        recon = copy.deepcopy(data[:self.output_channel_num])
        data = (data - self.mean) / self.std
        recon /= 1023.0
        gt /= 1023.0
        gt -= recon
        return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def ReverseNorm(self, x, idx):
        if torch.cuda.is_available():
            mean = torch.from_numpy(np.array(self.mean[idx], dtype='float32')).cuda()
            std = torch.from_numpy(np.array(self.std[idx], dtype='float32')).cuda()
        else:
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
        # alfmap = self.ctulist.getTuMaskFromIndex(0, self.pic.area.height, self.pic.area.width)
        # qpmap = np.full(qpmap.shape, qpmap[100,100])
        data = np.stack([*self.pic.pelBuf[PictureFormat.RECONSTRUCTION], qpmap, modemap, depthmap, hortrans, vertrans], axis = 0)
        orig = np.stack([*(np.array(self.pic.pelBuf[PictureFormat.ORIGINAL][0])[np.newaxis,:,:])])
        recon = copy.deepcopy(data[:self.output_channel_num])
        data = (data - self.mean) / self.std
        orig /= 1023.0
        recon /= 1023.0
        orig -= recon
        return recon.astype('float32'), data.astype('float32'), orig.astype('float32')

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

    Input_info = namedtuple('Input_info', 'input_channel output_channel encoded_channel')


    myinfo = Input_info(COMMONDATASETTING.DATA_CHANNEL_NUM, COMMONDATASETTING.OUTPUT_CHANNEL_NUM, 64)

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
    net = PixToPixReidualDenseNet(dataset.data_channel_num, 1, memory_efficient=False)
    # netmanage = NetTrainAndTest(net, train_loader, valid_loader, test_loader=None)

    netmanage = NetTrainAndTest(net, train_loader = train_loader, valid_loader = valid_loader, test_loader=None)
    netmanage.train()