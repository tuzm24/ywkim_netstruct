from CfgEnv.loadCfg import NetManager
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

class Mytensorboard(NetManager):
    def __init__(self, comment=''):
        self.writer = SummaryWriter(comment='_'+comment)
        self.writerLayout = {'Loss': {},
                        'PSNR': {}}
        self.step = 0
    def plotToTensorboard(self, fig, name):
        self.writer.add_figure(name, fig, global_step=self.step, close=True)

    def imgToTensorboard(self, img, name):
        # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8 or use tensorflow
        self.writer.add_image(name, img, global_step=self.step)

    def batchImageToTensorBoard(self, recon, resi, name):
        img = (recon.cpu().detach().numpy() + resi.cpu().detach().numpy())*255.0
        img = np.clip(img, 0, 255).astype(int)
        self.writer.add_image(name, img, global_step=self.step)

    @staticmethod
    def Makegrid(imgs, nrow = 5):
        return vutils.make_grid(imgs, nrow=nrow)

    def setObjectStep(self, num_set):
        self.object_step = num_set * self.OBJECT_EPOCH

    def plotScalars(self):
        for key, values in self.writerLayout.items():
            self.writer.add_scalars(key, values, self.step)

    def plotDifferent(self, resi, name):
        img = (resi.cpu().detach().numpy())*1023.0
        img = np.clip(img, -1023.0, 1023.0)
        fig, ax= plt.subplots()
        if img.min()<0 and img.max()>0:
            mymax = max(abs(img.min()), img.max())
            mymin = -mymax
        else:
            mymin = img.min()
            mymax = img.max()
        imgs = ax.imshow((img[0]).astype(int), vmin=mymin, vmax=mymax, interpolation='bilinear',
                   cmap=plt.cm.get_cmap('seismic'))
        v1 = np.linspace(mymin, mymax, 10, endpoint=True)
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        self.plotToTensorboard(fig, name)
        return

    def plotMap(self, img, name, vminmax = None, color_num = None):
        img = self.Makegrid(img)
        fig, ax= plt.subplots()
        if vminmax is None:
            vminmax = (img.min(), img.max())
        imgs = ax.imshow((img.cpu().numpy()[0]).astype(int), vmin=vminmax[0], vmax=vminmax[1], interpolation='bilinear',
                   cmap=plt.cm.get_cmap('viridis', color_num))
        v1 = np.linspace(vminmax[0], vminmax[1], 10, endpoint=True)
        cb = fig.colorbar(imgs, ticks=v1)
        cb.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])
        self.plotToTensorboard(fig, name)
        return

    def SetLoss(self, name, value):
        self.writerLayout['Loss'][name] = value

    def SetPSNR(self, name, value):
        self.writerLayout['PSNR'][name] = value

