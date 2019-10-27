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
    nDenselayer = 6
    nFeaturemaps = 32
    growthRate = 24

    def __init__(self, input_channels, output_channels):
        super(RDN, self).__init__()
        nChannel = input_channels
        nDenselayer = self.nDenselayer
        nFeat = self.nFeaturemaps
        growthRate = self.growthRate

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=0, bias=False)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=0, bias=False)
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
        self.firstlayer  = F.relu(self.conv1(x))
        self.secondlayer = F.relu(self.conv2(self.firstlayer))
        self.F_1 = self.RDB1(self.secondlayer)
        self.F_2 = self.RDB2(self.F_1)
        self.F_3 = self.RDB3(self.F_2)
        self.F_4 = self.RDB4(self.F_3)
        self.FF = torch.cat((self.F_1, self.F_2, self.F_3, self.F_4), 1)
        self.FdLF = F.relu(self.GFF_1x1(self.FF))
        self.FGF = F.relu(self.GFF_3x3(self.FdLF))
        self.FDF = self.FGF + self.secondlayer
        self.finallayer = self.conv_up(self.FDF)
        self.output = self.conv3(self.finallayer)
        return self.output

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
        self.data_padding = 2

    def getInputDataShape(self):
        return (self.batch_size, self.data_channel_num, self.batch[0][2], self.batch[0][1])


    def getOutputDataShape(self):
        return (self.output_channel_num, self.batch[0][2]- self.data_padding*2, self.batch[0][1] - self.data_padding*2)


    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
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
        gt = self.dropPadding(np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis=0), 2)
        recon = self.dropPadding(data[:self.output_channel_num], 2, isDeepCopy=True)
        data = (data - self.mean) / self.std
        recon /= 1023.0
        gt /= 1023.0
        gt -= recon
        return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def ReverseNorm(self, x, idx):
        mean = torch.from_numpy(np.array(self.mean[idx], dtype='float32')).cuda()
        std = torch.from_numpy(np.array(self.std[idx], dtype='float32')).cuda()
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
        orig = np.stack([*self.pic.dropPadding(np.array(self.pic.pelBuf[PictureFormat.ORIGINAL][0])[np.newaxis,:,:], 2, isDeepCopy=False)])
        recon = self.dropPadding(data[:self.output_channel_num], pad=2, isDeepCopy=True)
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
            net = net.cuda()
            criterion = criterion.cuda()
            MSE_loss = MSE_loss.cuda()
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
        tb.step = checkpoint['TensorBoardStep']
        net.eval()
        # for g in optimizer.param_groups:
        #     g['lr'] = 0.0001

    logger.info('Training Start')

    for epoch_iter, epoch in enumerate(range(NetManager.OBJECT_EPOCH), 1):
        running_loss = 0.0
        for i in range(dataset.batch_num):
            (recons, inputs, gts) = next(iter_training)

            if torch.cuda.is_available():
                recons = recons.cuda()
                inputs = inputs.cuda()
                gts = gts.cuda()
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
            # del recons, inputs, gts
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

                if torch.cuda.is_available():
                    recons = recons.cuda()
                    inputs = inputs.cuda()
                    gts = gts.cuda()
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
                        # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[4], idx=4).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Mode_Map', [0, 3], 4)
                        # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[5], idx=5).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Depth_Map', [1, 6], 6)
                        # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[6], idx=6).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Hor_Trans', [0, 2], 2)
                        # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[7], idx=7).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Ver_Trans', [0, 2], 2)
                        # tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[8], idx=8).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'ALF_IDX', [0, 16], 17)
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
    ctusize = 128
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



            MSE = MSE_loss(outputs, gts)
            recon_MSE = torch.mean((gts) ** 2)
            logger.info('ALL : %s(%s) : %s %s' %(path, i, myUtil.psnr(MSE.item()), myUtil.psnr(recon_MSE.item())))
            mean_test_psnr += myUtil.psnr(MSE.item())
            mean_testGT_psnr += myUtil.psnr(recon_MSE.item())

            for y in range(outputs.shape[2]):
                for x in range(outputs.shape[3]):
                    width = outputs.shape[2] - x if (x+ctusize)>outputs.shape[2] else ctusize
                    height = outputs.shzpe[3] - y if (y+ctusize)>outputs.shape[3] else ctusize
                    width += x
                    height +=y
                    if MSE_loss(outputs[:,:,y:height,x:width], gts[:,:, y:height, x:width]) > torch.mean(gts[:,:,y:height,x:width]**2):
                        outputs[:,:,y:height,x:width] = gts[:,:,y:height,x:width]


            logger.info('CTU %s : %s(%s) : %s %s' %(ctusize ,path, i, myUtil.psnr(MSE.item()), myUtil.psnr(recon_MSE.item())))



            outputs = outputs.cpu().numpy()[0,0]
            n_gts = gts.cpu().numpy()[0,0]


    logger.info("%s %s" % (mean_test_psnr / len(test_loader), mean_testGT_psnr / len(test_loader)))