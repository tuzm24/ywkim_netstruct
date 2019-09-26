import torch
from help_func.logging import LoggingHelper
from tqdm import tqdm


class torchUtil:
    logger = LoggingHelper.get_instance().logger

    @staticmethod
    def online_mean_and_sd(loader):
        """Compute the mean and sd in an online fashion

            Var[x] = E[X^2] - E^2[X]
        """
        torchUtil.logger.info('Calculating data mean and std')
        cnt = 0
        fst_moment = torch.empty(6).double()
        snd_moment = torch.empty(6).double()

        for _, data,_ in tqdm(loader):
            b, c, h, w = data.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels
        torchUtil.logger.info('Finish calculate data mean and std')
        return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)