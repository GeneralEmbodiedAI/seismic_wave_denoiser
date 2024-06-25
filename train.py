import os
import random

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics.audio import SignalNoiseRatio
from tqdm import tqdm

# from model.simple_model import SimpleModel
from model.seismic_net import SeismicNet
from model.loss import MRSTFTLoss
from utils.get_yaml import const
from utils.train_dataset import EarthquakeDataset, EarthquakeDataset_hdf5
from utils.eval_dataset import EvaluationDataset_hdf5   
from utils.util import get_optimizer




class Training(object):
    def __init__(self):
        # cuda environment set
        os.environ["CUDA_VISIBLE_DEVICES"] = const['train']['cuda_devices']

        self.epoch_loops = int(const['train']['epoch'])
        self.batch_size = int(const['train']['batch_size'])
        self.num_workers = int(const['train']['num_workers'])
        self.log_path = str(const['train']['log_folder'])
        self.model_save_folder = const['train']['model_save_folder']
        self.min_samples_ratio = float(const['train']['min_samples'])
        self.max_samples_ratio = float(const['train']['max_samples'])
        self.log_path = const['train']['log_folder']
        self.acc_path = os.path.join(self.log_path, 'acc_loss.csv')
        self.result_path = os.path.join(self.log_path, 'res.csv')
        self.print_loss_freq = int(const['train']['print_loss_frequency'])
        self.save_freq = int(const['train']['save_freq'])
        self.spect_ratio = float(const['loss']['spect_ratio'])
        self.sample_rate = int(const['train']['sample_rate'])
        self.snr = SignalNoiseRatio()

        # running params
        self.epoch = 0
        self.optimizer = None
        self.model = None
        self.train_dir = None
        self.train_loader = None
        self.test_dir = None
        self.test_loader = None
        self.shed = None
        self.criterion = None
        self.device = 'cpu'
        self.spect_criterion = None
        self.best_pearson = 0

        self.initialize()

    def initialize(self,dataset='stead'):
        os.makedirs(self.log_path, exist_ok=True)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = SeismicNet().to(self.device)
        # self.model = SimpleModel().to(self.device)
        self.resume_model(os.path.join(self.model_save_folder, 'best_earth_model.pth'))
        self.model.train()
        self.criterion = self.get_criterion().to(self.device)
        self.spect_criterion = MRSTFTLoss(
            fft_sizes=[128, 256, 512],
            hop_sizes=[64, 128, 256],
            win_lengths=[128, 256, 512],
            scale=None,
            sample_rate=self.sample_rate,
        ).to(self.device)

        if dataset=="stead":
            self.train_dir = EarthquakeDataset_hdf5(train_mode=True)
            self.test_dir = EarthquakeDataset_hdf5(train_mode=False)
        else:
            self.train_dir = EarthquakeDataset(train_mode=True)
            self.test_dir = EarthquakeDataset(train_mode=False)
        self.train_loader = DataLoader(self.train_dir, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, drop_last=True)

        self.test_loader = DataLoader(self.test_dir, batch_size=1)
        self.optimizer = get_optimizer([self.model])
        self.shed = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.3)

        # self.shed  = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    def save_model(self, mode='epoch'):
        os.makedirs(self.model_save_folder, exist_ok=True)
        state = {'model': self.model.state_dict(),
                 'pearson': self.best_pearson,
                 'epoch': self.epoch}
        if mode == 'epoch':
            model_path = os.path.join(self.model_save_folder, f'earth_model_{self.epoch}.pth')
        elif mode == 'current':
            model_path = os.path.join(self.model_save_folder, f'current_earth_model.pth')
        elif mode == 'best':
            model_path = os.path.join(self.model_save_folder, f'best_earth_model.pth')
        else:
            raise ValueError('ERROR: save model mode not recognized!')
        torch.save(state, model_path)

    def resume_model(self, weights_path):
        self.epoch = 0
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(weights['model'])
            self.epoch = weights['epoch']
            self.best_pearson = weights['pearson']
        else:
            print(f'WARNING: model `{weights_path}` does not exist!')

    @staticmethod
    def get_criterion():
        loss_type = const['loss']['image_loss_type']
        if loss_type == 'ccc':
            from model.loss import ConcordanceCorrelationCoefficientLoss
            return ConcordanceCorrelationCoefficientLoss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError('ERROR: loss type error!')

    def augmentation(self, x, y):
        # feature size: [batch, frame]
        # cut feature
        # if random.random() < 0.5:
        #     length = x.size()[-1]
        #     if self.min_samples_ratio < self.max_samples_ratio:
        #         # import pdb
        #         # pdb.set_trace()
        #         frames = random.randint(int(length*self.min_samples_ratio), int(length*self.max_samples_ratio))
        #         start_frame = random.randint(0, x.size(1) - frames)
        #         x = x[..., start_frame:start_frame + frames]
        #         y = y[..., start_frame:start_frame + frames]
        # random flip
        if random.random() < 0.5:
            x = x * -1
            y = y * -1
        # magnitude variation
        if random.random() < 0.5:
            ratio = random.uniform(0.95, 1.05)
            x = x * ratio
            y = y * ratio
        return x, y

    def metrics(self, x, y):
        pearson = scipy.stats.pearsonr(x.numpy().flatten(), y.numpy().flatten()).statistic
        snr = self.snr(x, y).item()
        return pearson, snr

    def testing(self):
        pp = []
        ss = []
        self.model.eval()
        with tqdm(total=len(self.test_loader), dynamic_ncols=True) as p_bar:
            p_bar.set_description(f'Test')
            for i, data in enumerate(self.test_loader):
                try:
                    x, y = data
                except:
                    x, y, _ =data
                x = x.to(self.device)
                y = y.squeeze(0).cpu().detach()
                x = self.model(x).squeeze(0).cpu().detach()
                
                p, s = self.metrics(x, y)
                pp.append(p)
                ss.append(s)
                # print(p, s)
                p_bar.update()
        pp = np.mean(pp)
        ss = np.mean(ss)
        print(f'Pearson: {pp}, SNR: {ss}')
        # switch to train
        self.model.train()
        return pp, ss

    def training(self):
        self.model.train()
        while self.epoch < self.epoch_loops:
            self.epoch += 1
            running_loss = 0
            total_loss = []

            with tqdm(total=len(self.train_loader), dynamic_ncols=True) as p_bar:
                p_bar.set_description(f'Epoch {self.epoch}')
                for i, data in enumerate(self.train_loader):

                    # try:
                    if data is None:
                        continue
                    x, y = data
                    # raw_x = x.clone()
                    # import pdb
                    # pdb.set_trace()
                    x = x.to(self.device)
                    y = y.to(self.device)
                    x, y = self.augmentation(x, y)
                    
                    x = self.model(x)
                    # print(x.shape, y.shape)
                    # print(torch.mean(x))
                    # print(torch.mean(y))
                    # import matplotlib.pyplot as plt
                    
                    # print('111111111111111111', torch.min(x[0]), torch.max(x[0]))
                    # plt.plot(raw_x[0, :].detach().numpy())
                    # plt.plot(y[0, :].cpu().detach().numpy(), color='blue', alpha=0.6)
                    # plt.plot(x[0, :].cpu().detach().numpy(), color='black', alpha=0.6)
                    # plt.savefig('./temp.png')
                    # plt.show()
                    signal_loss = self.criterion(x, y)
                    spect_loss = self.spect_criterion(x, y)
                    # except:
                    #     import pdb
                    #     pdb.set_trace()
                    loss = signal_loss + self.spect_ratio * spect_loss
                    # print()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # self.shed.step()
                    # print(self.optimizer.param_groups[0]['lr'])
                    running_loss += loss.item()
                    print(loss.item(), signal_loss.item())
                    total_loss.append(loss.item())

                    progress_updates = {
                        'loss': str(round(np.mean(total_loss), 3))
                    }

                    # write log
                    if i % self.print_loss_freq == self.print_loss_freq - 1:
                        with open(self.acc_path, 'a+') as w:
                            w.write(
                                f'{self.epoch},{i + 1},'
                                f'{running_loss / self.print_loss_freq}\n')
                        running_loss = 0

                    p_bar.set_postfix(progress_updates)
                    p_bar.update()
                    # break
                self.shed.step()
                if self.epoch % self.save_freq == 0:
                    self.save_model()
                self.save_model(mode='current')
                pearson, snr = self.testing()
                if pearson > self.best_pearson:
                    self.best_pearson = pearson
                    self.save_model(mode='best')
                with open(self.result_path, 'a+') as w:
                    w.write(
                        f'{self.epoch},'
                        f'{pearson},'
                        f'{snr}\n')
                with open(self.acc_path, 'a+') as w:
                    w.write(
                        f'Total Epoch Loss,'
                        f'{np.mean(total_loss)}\n')


if __name__ == '__main__':
    tr = Training()
    tr.training()
