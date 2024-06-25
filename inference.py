import os
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchmetrics.audio import SignalNoiseRatio
# from model.simple_model import SimpleModel
from model.seismic_net import SeismicNet
from utils.eval_dataset import EvaluationDataset, EvaluationDataset_hdf5
from utils.train_dataset import EarthquakeDataset
from utils.get_yaml import const


class Evaluate(object):
    def __init__(self):
        # cuda environment set
        os.environ["CUDA_VISIBLE_DEVICES"] = const['train']['cuda_devices']

        self.model_save_folder = const['train']['model_save_folder']
        self.result_folder = const['io']['eval_result_folder']

        self.eval_dir = None
        self.eval_loader = None
        self.model = None
        self.device = None
        self.snr = SignalNoiseRatio()
        self.initialize()

    def initialize(self, dataset='stead'):
        os.makedirs(self.result_folder, exist_ok=True)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = SeismicNet().to(self.device)
        self.resume_model(os.path.join(self.model_save_folder, 'best_earth_model.pth'))
        print(os.path.join(self.model_save_folder, 'best_earth_model.pth'))


        # self.resume_model(os.path.join(self.model_save_folder, 'earth_model_320.pth'))
        # print(os.path.join(self.model_save_folder, 'earth_model_320.pth'))
        self.model.eval()

        if dataset=='stead':
            self.eval_dir = EvaluationDataset_hdf5()
        else:
            self.eval_dir = EvaluationDataset_hdf5(train_mode=False)

        self.eval_loader = DataLoader(self.eval_dir, shuffle=False, batch_size=1)
        print(len(self.eval_dir))
    def resume_model(self, weights_path):
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(weights['model'])
            print(weights_path)
        else:
            print(f'WARNING: model `{weights_path}` does not exist!')

    def export(self, x, name, type):
        name = name + type
        path = os.path.join(self.result_folder, name)
        np.savetxt(path, x.numpy(), fmt='%.6f')
        # with open(path, 'w') as w:
        #     import pdb
        #     pdb.set_trace()
        #     for d in x:
        #         w.write(str(d) + '\n')

    @staticmethod
    def norm(seq, v_range):
        seq[seq < v_range[0]] = v_range[0]
        seq[seq > v_range[1]] = v_range[1]
        seq = seq / v_range[1]
        return seq

    @staticmethod
    def denorm(seq, mag):
        seq = seq * mag
        return seq

    def metrics(self, x, y):
        # import pdb
        # pdb.set_trace()
        pearson = scipy.stats.pearsonr(x, y).statistic
        try:
            snr = self.snr(torch.from_numpy(x), torch.from_numpy(y))
        except:
            snr = self.snr(x, y)
        # print(snr)
        return pearson, snr.item()

    def testing(self):
        pp = []
        ss = []
        self.model.eval()
        with tqdm(total=len(self.eval_loader), dynamic_ncols=True) as p_bar:
            p_bar.set_description(f'Test')
            for i, data in enumerate(self.eval_loader):
                noised, y, name = data
                x = noised.clone().to(self.device)
                y = y.squeeze(0).cpu().detach()
                x = self.model(x).squeeze(0).cpu().detach()

                px, sx = self.metrics(x[0, ...], y[0, ...])
                py, sy = self.metrics(x[1, ...], y[1, ...])
                pz, sz = self.metrics(x[2, ...], y[2, ...])
                pall, sall = self.metrics(x.flatten(), y.flatten())
                


                self.export(x.squeeze(0), str(name.item()), type='_denoised.txt')
                self.export(y.squeeze(0), str(name.item()), type='_gt.txt')
                self.export(noised.squeeze(0), str(name.item()), type='_noised.txt')
                # print(pall, sall)
                pp.append([px,py,pz,pall])
                ss.append([sx,sy,sz,sall])
                p_bar.update()

                # import pdb
                # pdb.set_trace()
        np.savetxt(os.path.join(self.result_folder, 'pearson.txt'), np.array(pp))
        np.savetxt(os.path.join(self.result_folder, 'snr.txt'), np.array(ss))
        pp = np.mean(np.array(pp)[:, -1])
        ss = np.mean(np.array(ss)[:, -1])
        print(f'Pearson: {pp}, SNR: {ss}')

        # switch to train
        # self.model.train()
        return pp, ss
    def evaluate(self):
        self.model.eval()
        with (tqdm(total=len(self.eval_loader), dynamic_ncols=True) as p_bar):
            p_bar.set_description(f'Evaluate')
            for i, data in enumerate(self.eval_loader):
                x, y, name = data
                # x = self.norm(x, [-500, 500])
                x = x.to(self.device)
                x = self.model(x).squeeze(0).cpu().detach().numpy()

                # import pdb
                # pdb.set_trace()
                pred = x # self.denorm(x, 100000)  
                gt = y.squeeze(0).cpu().detach().numpy() # self.denorm(y.squeeze(0).cpu().detach().numpy(), 100000)  
                pearson, snr = self.metrics(pred, gt)
                print(pearson, snr)
                # self.export(x, name[0])
                p_bar.update()
    # def evaluate(self):
    #     self.model.eval()
    #     with tqdm(total=len(self.eval_loader), dynamic_ncols=True) as p_bar:
    #         p_bar.set_description(f'Evaluate')
    #         for i, data in enumerate(self.eval_loader):
    #             x, name = data
    #             # x = self.norm(x, [-500, 500])
    #             x = x.to(self.device)
    #             x = self.model(x).squeeze(0).cpu().detach().numpy()
    #             x = self.denorm(x, 100000)
    #             self.export(x, name[0])
    #             p_bar.update()


if __name__ == '__main__':
    eva = Evaluate()
    eva.testing()
