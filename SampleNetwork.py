from help_func.__init__ import ExecFileName
import os
ExecFileName.filename = os.path.splitext(os.path.basename(__file__))[0]

import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as cp
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from help_func.logging import get_str_time



from CfgEnv.loadCfg import NetManager
from CfgEnv.loadData import DataBatch


from help_func.logging import LoggingHelper
from help_func.help_python import myUtil
from help_func.CompArea import LearningIndex

from collections import OrderedDict

from threading import Thread

from visual_tool.Tensorboard import Mytensorboard
from help_func.help_torch_parallel import DataParallelModel, DataParallelCriterion
import numpy as np

logger = LoggingHelper.get_instance().logger


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin*kernels_per_layer, kernel_size=3, padding = 1, groups=nin)
        self.pointwise = nn.Conv2d(nin*kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)






class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, num_input_features, num_output_features, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_final_conv = (32,32), memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_features, num_init_features, kernel_size=5, stride=1,
                                padding=2, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        for i, channel_num in enumerate(num_final_conv, 1):
            if i == len(num_final_conv):
                channel_num = num_output_features
            self.features.add_module('final_norm_%d' %i, nn.BatchNorm2d(num_features))
            self.features.add_module('final_RELU_%d' %i, nn.ReLU(inplace=True))
            self.features.add_module('final_conv_%d' %i, nn.Conv2d(num_features, channel_num,
                                                                   kernel_size=3, stride=1,
                                                                   bias=False))
            num_features = channel_num



        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        return out






class _DataBatch(DataBatch):
    def __init__(self, istraining, batch_size):
        DataBatch.__init__(self, istraining, batch_size)
        self.mean = self.cfg.DATAMEAN
        self.std = self.cfg.DATASTD
        self.data_channel_num = 6
        self.output_channel_num = 1
        if not self.mean and not self.std:
            self.mean = 0
            self.std = 1
        else:
            self.mean = (np.array(list(self.mean)))
            self.std = (np.array(list(self.std)))
            self.std[self.std==0] = 1

    def unpackData(self, info):
        DataBatch.unpackData(self, info)
        qpmap = self.tulist.getTuMaskFromIndex(0, info[2], info[1])
        modemap = self.tulist.getTuMaskFromIndex(1, info[2], info[1])
        depthmap = self.tulist.getTuMaskFromIndex(2, info[2], info[1])
        data = np.stack([*self.reshapeRecon(), qpmap, modemap, depthmap], axis = 2)
        gt = self.dropPadding(np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis = 0), 2)
        recon = self.TFdropPadding(data[:,:,:self.output_channel_num], 2, isDeepCopy=True).transpose((2,0,1))
        data = (data-self.mean)/self.std
        data = data.transpose((2,0,1))

        recon /= 1023.0
        gt /= 1023.0
        gt -= recon
        return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def ReverseNorm(self, x, idx):
        mean = torch.from_numpy(np.array(self.mean[idx]))
        std = torch.from_numpy(np.array(self.std[idx]))
        return x*std + mean


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
    logger.info("device %s" %device)
    dataset = _DataBatch(LearningIndex.TRAINING, _DataBatch.BATCH_SIZE)
    valid_dataset = _DataBatch(LearningIndex.VALIDATION, _DataBatch.BATCH_SIZE)
    # a = torch.randn(50, 80)  # tensor of size 50 x 80
    # b = torch.split(a, 40, dim=1)  # it returns a tuple
    # b = list(b)  # convert to list if you want
    # print(" ")
    pt_dataset = myDataBatch(dataset=dataset)
    pt_valid_dataset = myDataBatch(dataset=valid_dataset)
    if torch.cuda.is_available():
        train_loader = DataLoader(dataset=pt_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=True, num_workers=NetManager.NUM_WORKER, pin_memory=True)
        valid_loader = DataLoader(dataset=pt_valid_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=False, num_workers=NetManager.NUM_WORKER)
    else:
        train_loader = DataLoader(dataset=pt_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=True, num_workers=NetManager.NUM_WORKER)
        valid_loader = DataLoader(dataset=pt_valid_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=False, num_workers=NetManager.NUM_WORKER)
    # net = DenseNet(dataset.data_channel_num, 1, growth_rate=12, block_config=(4,4,4,4), drop_rate=0.2)
    iter_training = iter(train_loader)
    iter_valid = iter(valid_loader)

    net = DenseNet(num_input_features=dataset.data_channel_num, num_output_features=1,
                   growth_rate=24, block_config=(4,4,4), num_init_features=32, drop_rate=0.2,
                   num_final_conv=(32,32))
    # net.to(device)
    # summary(net, (dataset.data_channel_num,132,132))

    criterion = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    recon_MSE_loss = nn.MSELoss()
    if torch.cuda.is_available():
        recon_MSE_loss = recon_MSE_loss.cuda()
        if torch.cuda.device_count()>1:
            net = DataParallelModel(net).cuda()
            criterion = DataParallelCriterion(criterion).cuda()
            MSE_loss = DataParallelCriterion(nn.MSELoss()).cuda()
        else:
            net.cuda()
            criterion.cuda()
            MSE_loss.cuda()


    optimizer = optim.Adam(net.parameters(), lr = dataset.cfg.INIT_LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=[int(dataset.cfg.OBJECT_EPOCH*0.5), int(dataset.cfg.OBJECT_EPOCH*0.75)],
                                                  gamma=0.1, last_epoch=-1)




    object_step = dataset.batch_num * dataset.cfg.OBJECT_EPOCH
    tb = Mytensorboard()
    for epoch_iter, epoch in enumerate(range(dataset.cfg.OBJECT_EPOCH),1):
        running_loss = 0.0
        for i in range(dataset.batch_num):
            (recons, inputs, gts) = next(iter_training)
            # if torch.cuda.is_available():
            #     recons = recons.cuda().float()
            #     outputs = net(inputs.cuda().float())
            #     gts = gts.cuda().float()
            # else:
            #     recons = recons.float()
            #     outputs = net(inputs.float())
            #     gts = gts.float()
            outputs = net(inputs)
            loss = criterion(outputs, gts)
            MSE = MSE_loss(outputs, gts)
            recon_MSE = torch.mean((gts)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += MSE.item()

            tb.SetLoss('CNN', MSE.item())
            tb.SetLoss('Recon', recon_MSE.item())
            tb.SetPSNR('CNN', myUtil.psnr(MSE.item()))
            tb.SetPSNR('Recon', myUtil.psnr(recon_MSE.item()))
            tb.plotScalars()

            # dataset.logger.info('L2 loss %f' %MSE.item())
            if i%dataset.PRINT_PERIOD == dataset.PRINT_PERIOD - 1:
                dataset.logger.info('[Epoch : %d, %5d/%d] loss: %.7f' %
                  (epoch + 1, (i + 1 + (epoch_iter-1) * len(train_loader))*dataset.batch_size, object_step, running_loss / dataset.PRINT_PERIOD))
                running_loss = 0.0

            tb.step += 1 #Must Used

        mean_loss_cnn = 0
        mean_psnr_cnn = 0
        mean_loss_recon = 0
        mean_psnr_recon = 0

        for i in range(valid_dataset.batch_num):
            with torch.no_grad():
                (recons, inputs, gts) = next(iter_valid)

                # recons = recons.cuda().float()
                # outputs = net(inputs.cuda().float())
                # gts = gts.cuda().float()
                loss = criterion(outputs, gts)
                MSE = MSE_loss(outputs, gts)
                recon_MSE = torch.mean((gts)**2)
                mean_psnr_cnn += myUtil.psnr(MSE.item())
                mean_psnr_recon += myUtil.psnr(recon_MSE.item())
                mean_loss_cnn += MSE.item()
                mean_loss_recon += recon_MSE.item()
                if i==0:
                    if torch.cuda.device_count()>1:
                        outputs = torch.cat(outputs, dim = 0)
                    tb.batchImageToTensorBoard(tb.Makegrid(recons),tb.Makegrid(outputs), 'CNN_Reconstruction')
                    tb.plotDifferent(tb.Makegrid(outputs), 'CNN_Residual')
                    if epoch_iter==1:
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[3], idx=3).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'QP_Map', [22, 27], 4)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[4], idx=4).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Mode_Map', [0, 66], 67)
                        tb.plotMap(dataset.ReverseNorm(inputs.split(1, dim=1)[5], idx=5).narrow(dim =2, start=2, length=128).narrow(dim =3, start=2, length=128), 'Depth_Map', [1, 6], 6)

                    dataset.logger.info("[epoch:%d] Finish Plot Image" % epoch_iter)
        dataset.logger.info('[epoch : %d] Recon_loss : %.7f, Recon_PSNR : %.7f' %(epoch_iter, mean_loss_recon/len(valid_loader), mean_psnr_recon/len(valid_loader)))
        dataset.logger.info('[epoch : %d] CNN_loss   : %.7f, CNN_PSNR :   %.7f' %(epoch_iter, mean_loss_cnn/len(valid_loader), mean_psnr_cnn/len(valid_loader)))
        torch.save({
            'epoch': epoch_iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, NetManager.MODEL_PATH + '/model.pth')
        lr_scheduler.step()

        dataset.logger.info('Epoch %d Finished' %epoch_iter)


