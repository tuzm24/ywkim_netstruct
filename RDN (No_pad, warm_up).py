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
from help_func.help_torch import NetTrainAndTest


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Residual Dense Network
class RDN(nn.Module):
    nDenselayer = 3
    nFeaturemaps = 32
    growthRate = 24

    def __init__(self, input_channels, output_channels):
        super(RDN, self).__init__()
        nChannel = input_channels
        nDenselayer = self.nDenselayer
        nFeat = self.nFeaturemaps
        growthRate = self.growthRate

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=False)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB4 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*4, nFeat, kernel_size=1, padding=0, bias=False)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        # conv
        self.conv3 = nn.Conv2d(nFeat, output_channels, kernel_size=3, padding=1, bias=False)


    def forward(self, x):
        firstlayer  = F.relu(self.conv1(x))
        secondlayer = F.relu(self.conv2(firstlayer))
        F_1 = self.RDB1(secondlayer)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = self.RDB4(F_3)
        FF = torch.cat((F_1, F_2, F_3, F_4), 1)
        FdLF = F.relu(self.GFF_1x1(FF))
        FGF = F.relu(self.GFF_3x3(FdLF))
        FDF = FGF + secondlayer
        finallayer = self.conv_up(FDF)
        output = self.conv3(finallayer)
        return output

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
        # depthmap = self.tulist.getTuMaskFromIndex(2, info[2], info[1])
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
        # modemap = self.tulist.getTuMaskFromIndex(1, self.pic.area.height, self.pic.area.width)
        # modemap[np.all([modemap>1, modemap<34], axis = 0)] = 2
        # modemap[modemap>=34] = 3
        # depthmap = self.tulist.getTuMaskFromIndex(2, self.pic.area.height, self.pic.area.width)
        # hortrans = self.tulist.getTuMaskFromIndex(3, self.pic.area.height, self.pic.area.width)
        # vertrans = self.tulist.getTuMaskFromIndex(4, self.pic.area.height, self.pic.area.width)
        # alfmap = self.ctulist.getTuMaskFromIndex(0, self.pic.area.height, self.pic.area.width)
        # qpmap = np.full(qpmap.shape, qpmap[100,100])
        data = np.stack([*self.pic.pelBuf[PictureFormat.RECONSTRUCTION], qpmap], axis = 0)
        orig = np.stack([*(np.array(self.pic.pelBuf[PictureFormat.ORIGINAL][0])[np.newaxis,:,:])])
        recon = copy.deepcopy(data[:self.output_channel_num])
        data = (data - self.mean) / self.std
        orig /= 1023.0
        recon /= 1023.0
        orig -= recon
        return self.cur_path, recon.astype('float32'), data.astype('float32'), orig.astype('float32')

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
    net = RDN(dataset.data_channel_num, 1)
    netmanage = NetTrainAndTest(net, train_loader, valid_loader, test_loader=None)
    netmanage.train()