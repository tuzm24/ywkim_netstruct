

import os

if '__main__' == __name__:

    os.chdir("../")
    print(os.getcwd())

from CfgEnv.loadCfg import NetManager
from help_func.CompArea import TuList
# from help_func.CompArea import UnitBuf
# from help_func.CompArea import Component
# from help_func.CompArea import ChromaFormat
from help_func.CompArea import PictureFormat
# from help_func.CompArea import UniBuf
from help_func.CompArea import Size
# from help_func.CompArea import Area
from help_func.help_torch import torchUtil
from help_func.CompArea import LearningIndex
import numpy as np
import struct
import pandas as pd
# import random
import sklearn
import threading
import copy
import torch
from CfgEnv.loadCfg import LoggingHelper

from help_func.help_python import myUtil

from multiprocessing import Queue

class DataBatch(NetManager):
    #istraining :: 0 : test, 1 : training, 2 : validation


    def __init__(self, istraining, batch_size = 1): #
        if istraining == LearningIndex.TRAINING:
            self.data_path = self.TRAINING_PATH
            self.csv_path = os.path.join(self.TRAINING_PATH, self.CSV_NAME)
        elif istraining == LearningIndex.VALIDATION:
            self.data_path = self.VALIDATION_PATH
            self.csv_path = os.path.join(self.VALIDATION_PATH, self.CSV_NAME)
        else:
            self.data_path = self.TEST_PATH
            self.csv_path = os.path.join(self.TEST_PATH, self.CSV_NAME)
        self.istraining = istraining
        self.batch_size = batch_size
        self.csv = pd.read_csv(self.csv_path)
        print("[%s DataNum : %d]" %(LearningIndex.INDEX_DIC[istraining], len(self.csv)))
        self.sizeDic = {}
        self.tulen = len(self.TU_ORDER)
        self.csv = self.csv.dropna(axis='columns')
        if istraining>0:
            self.csv = sklearn.utils.shuffle(self.csv, random_state=42)
        self.batch = self.MakeBatch()
        self.batch_num = len(self.batch)
        print("[%s NumberOfBatchs : %d]" %(LearningIndex.INDEX_DIC[istraining], self.batch_num))
        self.batch = np.array(self.batch).reshape(-1, len(self.batch[0][0]))
        self.iter = 0


        # self.data = Queue()
        # t = threading.Thread(target=self.setNextData, daemon = True)
        # t.start()

    def MakeBatch(self):
        sizedic = {}
        batch_list = []
        for index, row in self.csv.iterrows():
            sizetuple = (row['HEIGHT'], row['WIDTH'])
            if sizetuple not in sizedic:
                self.SetSizeDic(sizetuple)
                sizedic[sizetuple] = [[]]
            sizedic[sizetuple][-1].append(row.as_matrix())
            if len(sizedic[sizetuple][-1]) == self.batch_size:
                sizedic[sizetuple].append(list())
        for v in sizedic.values():
            if len(v[-1]) < self.batch_size:
                del(v[-1])
            batch_list += v
        return batch_list

    def SetSizeDic(self, sizetuple):
        size = Size(sizetuple[1], sizetuple[0])
        area = size.getArea()
        carea = size.getCArea()
        datanum = 0
        split_bin = []
        for i in range(PictureFormat.MAX_NUM_COMPONENT):
            if self.PEL_DATA[i]:
                split_bin.append(area)
                datanum += area
                if not self.IS_ONLY_LUMA:
                   split_bin.append(carea)
                   split_bin.append(carea)
                   datanum += carea*2
            else:
                split_bin.append(0)
                if not self.IS_ONLY_LUMA:
                    split_bin.append(0)
                    split_bin.append(0)
        del split_bin[-1]
        split_bin = np.cumsum(split_bin)
        self.sizeDic[sizetuple] = (split_bin, '<'  + str(datanum) + 'h', datanum*2)
        return


    def getNextData(self):
        data = self.data.get()
        return data

    def setNextData(self):
        while True:
            for i in range(self.iter, self.iter + self.batch_size):
                pred, gt = self.unpackData(self.batch[i])
                self.data.put((pred, gt))
            self.iter +=self.batch_size
            while self.data.qsize() != 0:
                pass

    def isReadySetData(self):
        if self.data.qsize() == self.batch_size:
            return True
        return False


    # self.info - 0 : filename, 1 : width, 2: height, 3: qp, 4: mode, 5: depth ...
    # self : info, orgY, orgCb, orgCr, predY, predCb, predCr, reconY, reconCb, reconCr, unfiltredY, unfiltredCb, unfiltredCr
    # self.tulist
    def unpackData(self, info):
        self.info = info
        filepath = os.path.join(self.data_path, info[0])
        split_bin, strdatanum, shortdatanum = self.sizeDic[(info[1], info[2])]
        with open(filepath, 'rb') as data:
            self.orgY, self.orgCb, self.orgCr,\
            self.predY, self.predCb, self.predCr,\
            self.reconY, self.reconCb, self.reconCr,\
            self.unfilteredY, self.unfilteredCb, self.unfilteredCr\
                = np.split(np.array(struct.unpack(strdatanum, data.read(shortdatanum)),
                                    dtype='float32'), split_bin, axis=0)
            self.cwidth = self.info[1] // 2
            self.cheight = self.info[2] // 2
            if not self.IS_CONST_TU_DATA:
                self.tulist = TuList(np.array([[*info[:2], 0, 0, *info[3:] ]]))
            else:
                self.tulist = TuList.loadTuList(data)

    def reshapeOrg(self):

        return self.orgY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.orgCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.orgCr.reshape((self.cheight, self.cwidth)))

    def reshapePred(self):

        return self.predY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.predCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.predCr.reshape((self.cheight, self.cwidth)))

    def reshapeRecon(self):
        return self.reconY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.reconCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.reconCr.reshape((self.cheight, self.cwidth)))

    def reshapeUnfiltered(self):

        return self.unfilteredY.reshape((self.info[2], self.info[1])), myUtil.UpSamplingChroma(
            self.unfilteredCb.reshape((self.cheight, self.cwidth))), myUtil.UpSamplingChroma(
            self.unfilteredCr.reshape((self.cheight, self.cwidth)))

    def dropPadding(self, x, pad, isDeepCopy = False):
        if isDeepCopy:
            return copy.deepcopy(x[:,pad:-pad,pad:-pad])
        else:
            return x[:,pad:-pad,pad:-pad]

    def TFdropPadding(self, x, pad, isDeepCopy = False):
        if isDeepCopy:
            return copy.deepcopy(x[pad:-pad,pad:-pad,:])
        else:
            return x[pad:-pad,pad:-pad,:]

    def loadMeanStd(self, loader, isGetNew = False):
        mean = 0
        std = 0
        if not isGetNew:
            if self.cfg.isExist('DATAMEAN'):
                mean = self.cfg.DATAMEAN
            if self.cfg.isExist('DATASTD'):
                std = self.cfg.DATASTD
            if not mean or not std:
                print('There is no mean or standard deviation data present.')
                mean, std = torchUtil.online_mean_and_sd(loader)
            elif len(mean)!=self.data_channel_num or len(std)!=self.data_channel_num:
                print('The mean and std already exist and the number\
                 of channels in the current data does not match.')
                mean, std = torchUtil.online_mean_and_sd(loader)
            else:
                for i, _ in enumerate(std):
                    if not std[i]:
                        std[i] = 1024.0
                return (np.array(list(mean))), (np.array(list(std)))
        else:
            mean, std = torchUtil.online_mean_and_sd(loader)
        print('Save mean and std')
        self.cfg.member['DATAMEAN'] = mean.numpy().tolist()
        self.cfg.member['DATASTD'] = std.numpy().tolist()
        self.cfg.write_yml()
        return (mean.numpy()), (std.numpy())





if '__main__' == __name__:
    df = DataBatch(1, 3)
