import torch
from help_func.logging import LoggingHelper
from tqdm import tqdm
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CfgEnv.loadCfg import NetManager
from collections import OrderedDict
from help_func.help_torch_parallel import DataParallelModel, DataParallelCriterion
import os
import torch.optim as optim
from visual_tool.Tensorboard import Mytensorboard
from itertools import cycle
from help_func.help_python import myUtil
from help_func.__init__ import ExecFileName

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

class torchUtil:
    logger = LoggingHelper.get_instance().logger

    @staticmethod
    def online_mean_and_sd(loader):
        """Compute the mean and sd in an online fashion

            Var[x] = E[X^2] - E^2[X]
        """
        input_channel = loader.dataset.dataset.data_channel_num
        torchUtil.logger.info('Calculating data mean and std')
        cnt = 0
        fst_moment = torch.empty(input_channel).float().to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        snd_moment = torch.empty(input_channel).float().to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        for _, data, _ in tqdm(loader):
            b, c, h, w = data.shape
            data = data.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels
        torchUtil.logger.info('Finish calculate data mean and std')
        return fst_moment.cpu(), torch.sqrt(snd_moment - fst_moment ** 2).cpu()

    @staticmethod
    def Calc_Pearson_Correlation(loader, dataidx, opt='mean'):
        def maxCountValue(arr1d):
            brr, idxs = np.unique(arr1d, return_counts=True)
            return brr[np.argmax(idxs)]

        torchUtil.logger.info('Calculating data mean and std')
        dataidx +=3
        x = []
        y = []
        for _, data, gt in tqdm(loader):
            b, c, h, w = data.shape
            data = data.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            gt = gt.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            gtmse = torch.sum(gt[:,0,:,:]**2, dim = [1,2])
            y.append(gtmse.view(-1).numpy())
            if opt =='mean':
                datamean = torch.mean(data[:,dataidx,:,:], dim=[1,2])
                x.append(datamean.view(-1).numpy())
            if opt == 'max':
                data = data[:,dataidx,:,:].numpy().reshape((b,-1))
                x.append(np.apply_along_axis(maxCountValue, 1, data))

        y = np.array(y).reshape(-1)
        x = np.array(x).reshape(-1)
        return np.corrcoef(x,y)[0][1]

    @staticmethod
    def _RoundChannels(c, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
        if new_c < 0.9 * c:
            new_c += divisor
        return new_c

    @staticmethod
    def _SplitChannels(channels, num_groups):
        split_channels = [channels // num_groups for _ in range(num_groups)]
        split_channels[0] += channels - sum(split_channels)
        return split_channels

    @staticmethod
    def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU', padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def UnPixelShuffle(x, r):
        b, c, h, w = x.shape
        out_channel = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        return x.contiguous().view(b, c, out_h, r, out_w, r).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class NetTrainAndTest:
    logger = LoggingHelper.get_instance().logger
    def __init__(self, net,train_loader, valid_loader, test_loader,mainloss = 'l1', opt = 'adam'):
        self.net = net
        self.name = ExecFileName.filename
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.train_batch_num, self.train_iter = self.getBatchNumAndCycle(self.train_loader)
        self.valid_batch_num, self.valid_iter = self.getBatchNumAndCycle(self.valid_loader)
        self.test_batch_num, self.test_iter = self.getBatchNumAndCycle(self.test_loader)

        if torch.cuda.is_available():
            self.iscuda = True
            self.cuda_device_count = torch.cuda.device_count()
        else:
            self.iscuda = False
            self.cuda_device_count = 0
        self.criterion = self.setloss()
        self.ResultMSELoss = self.setloss('l2')
        self.GTMSELoss = self.setloss('l2')
        if self.iscuda:
            self.GTMSELoss = self.GTMSELoss.cuda()
            if self.cuda_device_count>1:
                self.net = DataParallelModel(net).cuda()
                self.criterion = DataParallelCriterion(self.criterion).cuda()
                self.ResultMSELoss = DataParallelCriterion(self.ResultMSELoss).cuda()
            else:
                self.net = self.net.cuda()
                self.criterion = self.criterion.cuda()
                self.ResultMSELoss = self.ResultMSELoss.cuda()
        self.optimizer = self.setopt(opt)(self.net.parameters(), lr = NetManager.cfg.INIT_LEARNING_RATE)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                      milestones=[int(NetManager.cfg.OBJECT_EPOCH * 0.5),
                                                                  int(NetManager.cfg.OBJECT_EPOCH * 0.75)],
                                                      gamma=0.1, last_epoch=-1)

        self.tb = Mytensorboard(self.name)
        self.highestScore = 0
        self.epoch = 0
        self.load_model()

    @staticmethod
    def setopt(opt = 'adam'):
        if opt == 'adam':
            return optim.Adam
        else:
            assert 0, 'The Optimizer is ambiguous'

    @staticmethod
    def getBatchNumAndCycle(dataloader):
        if dataloader is not None:
            return dataloader.dataset.batch_num, cycle(dataloader)
        else:
            return None, None


    @staticmethod
    def setloss(loss = 'l1'):
        loss = loss.lower()
        if loss == 'l1':
            return nn.L1Loss
        elif loss == 'l2':
            return nn.MSELoss()
        else:
            assert 0, 'The loss is ambiguous'


    def test(self):
        MSE_loss = nn.MSELoss()
        recon_MSE_loss = nn.MSELoss()
        if torch.cuda.is_available():
            MSE_loss.cuda()
            recon_MSE_loss.cuda()
        self.net.eval()
        test_psnr_mean = []
        for sequencedir, pocdics in self.test_loader.dataset.dataset.seqdic.items():
            total_psnr = []
            self.logger.info('%s' %sequencedir)
            for i, (pocpath, filelist) in enumerate(pocdics.items()):
                pocwidth = pocpath.split('_')[-1]
                pocheight = pocpath.split('_')[-2]
                gtbuf = np.full((pocheight, pocwidth), np.nan)
                resibuf = np.full((pocheight, pocwidth), np.nan)
                reconbuf = np.full((pocheight, pocwidth), np.nan)
                for filename in filelist:
                    ypos, xpos, h, w = filename.split('.')[0].split('_')
                    recons, inputs, gts = next(self.test_iter)
                    gtbuf[ypos:ypos+h, xpos:xpos+w] = gts[0].cpu().numpy()
                    reconbuf[ypos:ypos+h, xpos:xpos+w] = recons[0].cpu().numpy()
                    if self.iscuda:
                        recons.cuda()
                        inputs.cuda()
                        gts.cuda()
                    outputs = self.net(inputs)
                    resibuf[ypos:ypos+h, xpos:xpos+w] = outputs[0].cpu().numpy()
                if np.any(np.isnan(resibuf)):
                    self.logger.error('Nan is exists in array')
                    continue
                if i==0:
                    self.tb.SaveImageToTensorBoard('Recon/'+pocpath, gtbuf + resibuf)
                    self.tb.SaveImageToTensorBoard('Residual/' + pocpath, resibuf)

                poc_mse = (gtbuf - resibuf) ** 2
                poc_psnr = myUtil.psnr(poc_mse)




    def load_model(self):
        if NetManager.cfg.LOAD_SAVE_MODEL:
            PATH = './' + NetManager.MODEL_PATH + '/' + self.name + '_model.pth'
            if self.iscuda:
                checkpoint = torch.load(PATH)
            else:
                checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
            try:
                self.net.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                self.logger.info('Not equal GPU Number.. %s' % e)
                if self.cuda_device_count == 1:
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model_state_dict'].items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.net.load_state_dict(new_state_dict)
                else:
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model_state_dict'].items():
                        name = 'module.' + k  # remove `module.`
                        new_state_dict[name] = v
                    self.net.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if NetManager.cfg.LOAD_SAVE_MODEL == 1:
                self.epoch = checkpoint['epoch']
                self.tb.step = checkpoint['TensorBoardStep']
                if 'valid_psnr' in checkpoint:
                    self.highestScore = checkpoint['valid_psnr']
                self.logger.info('It is Transfer Learning...')
            self.logger.info('Load the saved checkpoint')
            # for g in optimizer.param_groups:
            #     g['lr'] = 0.0001

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'TensorBoardStep': self.tb.step,
            'valid_psnr': self.highestScore
        }, NetManager.MODEL_PATH + '/' + self.name + '_model.pth')


    def train(self, input_channel_list = None):
        dataset = self.train_loader.dataset.dataset
        for epoch_iter, epoch in enumerate(range(NetManager.OBJECT_EPOCH), self.epoch):
            running_loss = 0.0
            for i in range(dataset.batch_num):
                (recons, inputs, gts) = next(self.train_iter)

                if torch.cuda.is_available():
                    # recons = recons.cuda()
                    inputs = inputs.cuda()
                    gts = gts.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, gts)
                MSE = self.ResultMSELoss(outputs, gts)
                recon_MSE = torch.mean((gts) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += MSE.item()
                self.tb.SetLoss('CNN', MSE.item())
                self.tb.SetLoss('Recon', recon_MSE.item())
                self.tb.plotScalars()
                if i % NetManager.PRINT_PERIOD == NetManager.PRINT_PERIOD - 1:
                    self.logger.info('[Epoch : %d, %5d/%d] loss: %.7f' %
                                (epoch_iter, i + 1,
                                 dataset.batch_num, running_loss / dataset.PRINT_PERIOD))
                    running_loss = 0.0
                # del recons, inputs, gts
                self.tb.step += 1  # Must Used
            if self.valid_loader is not None:
                self.valid(epoch_iter, input_channel_list)
            self.lr_scheduler.step(epoch_iter)
            self.epoch += 1
            self.logger.info('Epoch %d Finished' % epoch_iter)

    def valid(self, epoch_iter, input_channel_list = None):
        mean_loss_cnn = 0
        mean_psnr_cnn = 0
        mean_loss_recon = 0
        mean_psnr_recon = 0
        valid_dataset = self.valid_loader.dataset.dataset
        cumsum_valid = torch.zeros(valid_dataset.getOutputDataShape()).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        if input_channel_list is None:
            input_channel_list = list()
            for i in range(valid_dataset.data_channel_num):
                input_channel_list.append(str(i) + '_channel')


        for i in range(valid_dataset.batch_num):
            with torch.no_grad():
                (recons, inputs, gts) = next(self.valid_iter)

                if torch.cuda.is_available():
                    recons = recons.cuda()
                    inputs = inputs.cuda()
                    gts = gts.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, gts)
                recon_loss = torch.mean(torch.abs(gts))
                MSE = self.ResultMSELoss(outputs, gts)
                recon_MSE = torch.mean((gts) ** 2)
                mean_psnr_cnn += myUtil.psnr(MSE.item())
                mean_psnr_recon += myUtil.psnr(recon_MSE.item())
                mean_loss_cnn += loss.item()
                mean_loss_recon += recon_loss.item()
                if self.cuda_device_count > 1:
                    outputs = torch.cat(outputs, dim=0)
                cumsum_valid += (outputs ** 2).sum(dim=0)
                if i == 0:
                    self.tb.batchImageToTensorBoard(self.tb.Makegrid(recons), self.tb.Makegrid(outputs), 'CNN_Reconstruction')
                    self.tb.plotDifferent(self.tb.Makegrid(outputs), 'CNN_Residual')
                    self.tb.plotDifferent(self.tb.Makegrid(outputs), 'CNN_Residual', percentile=100)
                    if epoch_iter == 1:
                        for channel, channel_name in enumerate(input_channel_list):
                            self.tb.plotMap(valid_dataset.ReverseNorm(
                                inputs.split(1, dim=1)[3], idx=channel).narrow(dim=2, start=0,                                                                                              length=128).narrow(
                                dim=3, start=0, length=128), channel_name)
                            # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[4], idx=4).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Mode_Map', [0, 3], 4)
                            # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[5], idx=5).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Depth_Map', [1, 6], 6)
                            # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[6], idx=6).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Hor_Trans', [0, 2], 2)
                            # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[7], idx=7).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Ver_Trans', [0, 2], 2)
                            # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[8], idx=8).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'ALF_IDX', [0, 16], 17)
                    self.logger.info("[epoch:%d] Finish Plot Image" % epoch_iter)
        cumsum_valid /= (valid_dataset.batch_num * valid_dataset.batch_size)
        self.tb.plotMSEImage(cumsum_valid, 'Error_MSE')
        self.tb.plotMAEImage(cumsum_valid, 'Error_MAE')
        self.tb.plotMAEImage(cumsum_valid, 'Error_MAE', percentile=80)
        self.tb.plotMAEImage(cumsum_valid, 'Error_MAE', percentile=100)
        if self.highestScore < (mean_psnr_cnn / len(self.valid_loader)):
            self.save_model()
            save_str = 'Save'
            self.highestScore = mean_psnr_cnn / len(self.valid_loader)
        else:
            save_str = 'No Save'
        self.logger.info('[epoch : %d] Recon_loss : %.7f, Recon_PSNR : %.7f' % (
            epoch_iter, mean_loss_recon / len(self.valid_loader), mean_psnr_recon / len(self.valid_loader)))
        self.logger.info('[epoch : %d] CNN_loss   : %.7f, CNN_PSNR :   %.7f   [%s]' % (
            epoch_iter, mean_loss_cnn / len(self.valid_loader), mean_psnr_cnn / len(self.valid_loader), save_str))
