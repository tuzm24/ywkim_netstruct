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

def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )

def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio, additional_data = None):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()
        self.additional_data = additional_data

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = torch.cat((y, *self.additional_data), dim = 0)
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




class _DataBatch(DataBatch):
    def __init__(self, istraining, batch_size):
        DataBatch.__init__(self, istraining, batch_size)
        self.mean = self.cfg.DATAMEAN
        self.std = self.cfg.DATASTD
        self.data_channel_num = 13
        self.output_channel_num = 1
        if not self.mean and not self.std:
            self.mean = 0
            self.std = 1
        else:
            self.mean = (np.array(list(self.mean)))
            self.std = (np.array(list(self.std)))
            self.std[self.std == 0] = 1
        self.qplist = [22,27,32,37]
        self.depthlist = [i for i in range(1,7)]

    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
    def unpackData(self, info):
        DataBatch.unpackData(self, info)
        qpmap = self.tulist.getTUMaskFromIndex_OneHot(0, info[2], info[1], self.qplist)
        tumap = self.tulist.getTUMaskFromIndex_OneHot(2, info[2], info[1], self.depthlist)
        data = np.stack([*self.reshapeRecon(), *qpmap, *tumap], axis=2)
        gt = self.dropPadding(np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis=0), 2)
        recon = self.TFdropPadding(data[:, :, :self.output_channel_num], 2, isDeepCopy=True).transpose((2, 0, 1))
        data[:,:,:3] = (data[:,:,:3] - self.mean) / self.std
        data = data.transpose((2, 0, 1))

        recon /= 1023.0
        gt /= 1023.0
        gt -= recon
        return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def ReverseNorm(self, x, idx):
        mean = torch.from_numpy(np.array(self.mean[idx]))
        std = torch.from_numpy(np.array(self.std[idx]))
        return x * std + mean


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
    valid_dataset = _DataBatch(LearningIndex.VALIDATION, _DataBatch.BATCH_SIZE)

    pt_dataset = myDataBatch(dataset=dataset)
    pt_valid_dataset = myDataBatch(dataset=valid_dataset)

    train_loader = DataLoader(dataset=pt_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=True,
                              num_workers=NetManager.NUM_WORKER)
    valid_loader = DataLoader(dataset=pt_valid_dataset, batch_size=dataset.batch_size, drop_last=True,
                              shuffle=False, num_workers=NetManager.NUM_WORKER)
    # net = DenseNet(dataset.data_channel_num, 1, growth_rate=12, block_config=(4,4,4,4), drop_rate=0.2)
    iter_training = cycle(train_loader)
    iter_valid = cycle(valid_loader)


    # net = MobileNetV2(input_dim=dataset.data_channel_num, output_dim=1)
    net = MixNet(input_channel=dataset.data_channel_num, output_channel=1)
    # net.to(device)
    summary(net, (dataset.data_channel_num,128,128), device='cpu')

    criterion = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    recon_MSE_loss = nn.MSELoss()
    if torch.cuda.is_available():
        recon_MSE_loss = recon_MSE_loss.cuda()
        if torch.cuda.device_count() > 1:
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
    for epoch_iter, epoch in enumerate(range(dataset.cfg.OBJECT_EPOCH), 1):
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
            tb.SetPSNR('CNN', myUtil.psnr(MSE.item()))
            tb.SetPSNR('Recon', myUtil.psnr(recon_MSE.item()))
            tb.plotScalars()

            if i % dataset.PRINT_PERIOD == dataset.PRINT_PERIOD - 1:
                logger.info('[Epoch : %d, %5d/%d] loss: %.7f' %
                                    (epoch_iter,i+1,
                                     dataset.batch_num, running_loss / dataset.PRINT_PERIOD))
                running_loss = 0.0

            tb.step += 1  # Must Used

        mean_loss_cnn = 0
        mean_psnr_cnn = 0
        mean_loss_recon = 0
        mean_psnr_recon = 0

        for i in range(valid_dataset.batch_num):
            with torch.no_grad():
                (recons, inputs, gts) = next(iter_valid)

                outputs = net(inputs)
                loss = criterion(outputs, gts)
                MSE = MSE_loss(outputs, gts)
                recon_MSE = torch.mean((gts) ** 2)
                mean_psnr_cnn += myUtil.psnr(MSE.item())
                mean_psnr_recon += myUtil.psnr(recon_MSE.item())
                mean_loss_cnn += MSE.item()
                mean_loss_recon += recon_MSE.item()
                if i == 0:
                    if torch.cuda.device_count() > 1:
                        outputs = torch.cat(outputs, dim=0)
                    tb.batchImageToTensorBoard(tb.Makegrid(recons), tb.Makegrid(outputs), 'CNN_Reconstruction')
                    tb.plotDifferent(tb.Makegrid(outputs), 'CNN_Residual')


                    logger.info("[epoch:%d] Finish Plot Image" % epoch_iter)
        logger.info('[epoch : %d] Recon_loss : %.7f, Recon_PSNR : %.7f' % (
        epoch_iter, mean_loss_recon / len(valid_loader), mean_psnr_recon / len(valid_loader)))
        logger.info('[epoch : %d] CNN_loss   : %.7f, CNN_PSNR :   %.7f' % (
        epoch_iter, mean_loss_cnn / len(valid_loader), mean_psnr_cnn / len(valid_loader)))
        torch.save({
            'epoch': epoch_iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, NetManager.MODEL_PATH + '/'+ os.path.splitext(os.path.basename(__file__))[0] +'_model.pth')
        lr_scheduler.step()

        logger.info('Epoch %d Finished' % epoch_iter)


