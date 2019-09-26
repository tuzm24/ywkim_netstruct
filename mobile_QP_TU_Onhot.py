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
logger = LoggingHelper.get_instance().logger
filename = os.path.basename(__file__)

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


    net = MobileNetV2(input_dim=dataset.data_channel_num, output_dim=1)
    # net.to(device)
    summary(net, (dataset.data_channel_num,132,132), device='cpu')

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
                dataset.logger.info('[Epoch : %d, %5d/%d] loss: %.7f' %
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


                    dataset.logger.info("[epoch:%d] Finish Plot Image" % epoch_iter)
        dataset.logger.info('[epoch : %d] Recon_loss : %.7f, Recon_PSNR : %.7f' % (
        epoch_iter, mean_loss_recon / len(valid_loader), mean_psnr_recon / len(valid_loader)))
        dataset.logger.info('[epoch : %d] CNN_loss   : %.7f, CNN_PSNR :   %.7f' % (
        epoch_iter, mean_loss_cnn / len(valid_loader), mean_psnr_cnn / len(valid_loader)))
        torch.save({
            'epoch': epoch_iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, NetManager.MODEL_PATH + '/'+ os.path.splitext(os.path.basename(__file__))[0] +'_model.pth')
        lr_scheduler.step()

        dataset.logger.info('Epoch %d Finished' % epoch_iter)


