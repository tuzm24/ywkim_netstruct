import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from CfgEnv.loadCfg import NetManager
from CfgEnv.loadData import DataBatch
import torch.nn.functional as F
from help_func.logging import LoggingHelper
from help_func.CompArea import LearningIndex
from help_func.help_torch import torchUtil

import os
import math

import torch.nn as nn
from bayes_opt import BayesianOptimization

logger = LoggingHelper.get_instance().logger
filename = os.path.basename(__file__)


class COMMONDATASETTING():
    DATA_CHANNEL_NUM = 4
    OUTPUT_CHANNEL_NUM = 1
    qplist = [22, 27, 32, 37]
    depthlist = [i for i in range(1, 7)]
    modelist = [i for i in range(0, 67)]
    translist = [0, 2]
    mean = NetManager.cfg.DATAMEAN[:DATA_CHANNEL_NUM]
    std = NetManager.cfg.DATASTD[:DATA_CHANNEL_NUM]
    alflist = [i for i in range(0, 17)]
    if not mean and not std:
        mean = 0
        std = 1
    else:
        mean = (np.array(list(mean))).reshape((len(mean), 1, 1))
        std = (np.array(list(std))).reshape((len(std), 1, 1))
        std[std == 0] = 1


import copy


class _DataBatch(DataBatch, COMMONDATASETTING):
    def __init__(self, istraining, batch_size):
        DataBatch.__init__(self, istraining, batch_size)
        self.data_channel_num = COMMONDATASETTING.DATA_CHANNEL_NUM
        self.output_channel_num = COMMONDATASETTING.OUTPUT_CHANNEL_NUM
        self.data_padding = 0

    def getInputDataShape(self):
        return (self.batch_size, self.data_channel_num, self.batch[0][2], self.batch[0][1])

    def getOutputDataShape(self):
        return (
        self.output_channel_num, self.batch[0][2] - self.data_padding * 2, self.batch[0][1] - self.data_padding * 2)

    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
    def unpackData(self, info):
        DataBatch.unpackData(self, info)
        qpmap = self.tulist.getTuMaskFromIndex(0, info[2], info[1])
        modemap = self.tulist.getTuMaskFromIndex(1, info[2], info[1])
        # modemap[np.all([modemap>1, modemap<34], axis = 0)] = 2
        # modemap[modemap>=34] = 3
        depthmap = self.tulist.getTuMaskFromIndex(2, info[2], info[1])
        hortrans = self.tulist.getTuMaskFromIndex(3, info[2], info[1])
        vertrans = self.tulist.getTuMaskFromIndex(4, info[2], info[1])
        alfmap = self.ctulist.getTuMaskFromIndex(0, info[2], info[1])
        data = np.stack([*self.reshapeRecon(), qpmap, modemap, depthmap, hortrans, vertrans, alfmap], axis=0)
        # data = np.stack([*self.reshapeRecon(),qpmap], axis=0)
        gt = (np.stack([self.orgY.reshape((self.info[2], self.info[1]))], axis=0))
        recon = copy.deepcopy(data[:self.output_channel_num])
        gt -= recon
        return recon.astype('float32'), data.astype('float32'), gt.astype('float32')

    def ReverseNorm(self, x, idx):
        mean = torch.from_numpy(np.array(self.mean[idx], dtype='float32')).cuda()
        std = torch.from_numpy(np.array(self.std[idx], dtype='float32')).cuda()
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
    dataset = _DataBatch(LearningIndex.TRAINING, _DataBatch.BATCH_SIZE)
    pt_dataset = myDataBatch(dataset=dataset)
    train_loader = DataLoader(dataset=pt_dataset, batch_size=dataset.batch_size, drop_last=True, shuffle=True,
                              num_workers=NetManager.NUM_WORKER)
    mse, qp, depth = torchUtil.Calc_Pearson_Correlation(train_loader, np.array([0,2]))
    # inputs = np.array(list(zip(qp, depth)))


    def getCorrelation(alpha):
        weight = qp * alpha + depth
        return np.corrcoef(mse, weight)[0][1]
    bayes_optimizer = BayesianOptimization(
        f = getCorrelation,
        pbounds = {
            'alpha' : (0,1),
        },
        random_state=42
    )
    bayes_optimizer.maximize(init_points=10, n_iter=1000, acq='ei', xi=0.01)
    for i, res in enumerate(bayes_optimizer.res):
        logger.info('Iteration %s: \n\t%s' %(i, res))
    logger.info('Final result: %s' %bayes_optimizer.max)

