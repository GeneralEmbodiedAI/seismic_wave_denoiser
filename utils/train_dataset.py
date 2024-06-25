import os
import random
import h5py
import numpy as np
import torch
import pandas as pd
import tensorboardX 
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import pdb

from utils.get_yaml import const


class EarthquakeDataset(Dataset):
    def __init__(self, train_mode=True):
        self.input_folder = os.path.abspath(const['io']['input'])
        self.label_folder = os.path.abspath(const['io']['label'])
        self.test_ratio = float(const['io']['test_ratio'])
        self.dataset = []
        self.train_dataset = []
        self.test_dataset = []
        self.train_mode = train_mode

        self.initialize()
        print(len(self.dataset))
        print(len(self.test_dataset))
        self.setup_random_seed(123)

    @staticmethod
    def setup_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __getitem__(self, item):
        # import pdb
        # pdb.set_trace()
        if self.train_mode:
            x, y = self.train_dataset[item]
        else:
            x, y = self.test_dataset[item]
        x = self.read_file(x)
        x = self.norm(x, [-500, 500])
        y = self.read_file(y)
        y = self.norm(y, [-100000, 100000])
        if self.train_mode:
            return x, y
        else:
            return x, y, item


    @staticmethod
    def norm(seq, v_range):
        seq[seq < v_range[0]] = v_range[0]
        seq[seq > v_range[1]] = v_range[1]
        seq = seq / v_range[1]
        return seq

    def __len__(self):
        if self.train_mode:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)

    def initialize(self):
        self.load_dataset()
        self.separate_dataset()

    def load_dataset(self):
        # print(os.listdir(self.input_folder))
        for name_qz in os.listdir(self.input_folder):
            name = name_qz.split('_')[0]
            name_cz = name + '_C.txt'
            path_qz = os.path.join(self.input_folder, name_qz)
            path_cz = os.path.join(self.label_folder, name_cz)
            if os.path.exists(path_qz) and os.path.exists(path_cz):
                self.dataset.append((path_qz, path_cz))

    @staticmethod
    def read_file(path):
        numbers = np.loadtxt(path)
        numbers = torch.from_numpy(numbers).float()
        return numbers

    def separate_dataset(self):
        random.shuffle(self.dataset)
        test_samples = int(len(self.dataset) * self.test_ratio)
        self.test_dataset = self.dataset[:test_samples]
        self.train_dataset = self.dataset[test_samples:]


class EarthquakeDataset_hdf5(Dataset):
    def __init__(self, train_mode=True):
        self.setup_random_seed(123)
        self.train_mode = train_mode
        self.input_folder = os.path.abspath(const['io']['input'])
        self.label_folder = os.path.abspath(const['io']['label'])
        self.val_ratio = float(const['io']['test_ratio'])

        self.train_dataset = []
        self.test_dataset = []
        self.dataset = self.load_h5py_dataset(self.input_folder, self.label_folder)

        self.separate_dataset()
        # self.initialize()


    def load_h5py_dataset(self, file, label_f):
        # import pdb
        # pdb.set_trace()


        with h5py.File(file, "r") as f:
            data = f['traces'][:]
            # data = (data[:,:,0]**2 + data[:,:,1]**2+data[:,:,2]**2)**0.5
            idx_i = f['metadata'][:]
        with h5py.File(label_f, "r") as f:
            label = f['traces'][:]
            # label = (label[:,:,0]**2 + label[:,:,1]**2+label[:,:,2]**2)**0.5
            idx_l = f['metadata'][:]

        dataset = []
        for i in range(len(data)):
            x, y = data[i, ...], label[i, ...]
            # y = self.norm2(y, [-30000, 30000])
            # x = self.norm2(x, [-500, 500])
            x,y = self.norm(x,y)
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            dataset.append([x , y])
        # import pdb
        # pdb.set_trace()
        return np.array(dataset).transpose((0, 3, 2, 1))
        # pass

    @staticmethod
    def setup_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def __getitem__(self, item):
        # import pdb
        # pdb.set_trace()
        if self.train_mode:
            x, y = self.train_dataset[item][..., 0], self.train_dataset[item][..., 1]
        else:
            x, y = self.test_dataset[item][..., 0], self.test_dataset[item][..., 1]
        # x = self.read_file(x)
            
        # y = self.read_file(y)

        x = y + x*random.randint(30, 80) * 0.01
        # y = self.read_file(y)
        # y = self.norm(y, None)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
        # y = self.norm(y, None)
        # return torch.from_numpy(y).float(), torch.from_numpy(x).float()


    def norm(self, x, y):
        return x/np.max(np.abs(x)), y/np.max(np.abs(y))

    def __len__(self):
        if self.train_mode:
            return len(self.train_dataset)
        else:
            return len(self.test_dataset)

    def initialize(self):
        self.load_dataset()
        self.separate_dataset()

    def load_dataset(self):
        for name_qz in os.listdir(self.input_folder):
            name = name_qz.split('_')[0]
            name_cz = name + '_C.txt'
            path_qz = os.path.join(self.input_folder, name_qz)
            path_cz = os.path.join(self.label_folder, name_cz)
            if os.path.exists(path_qz) and os.path.exists(path_cz):
                self.dataset.append((path_qz, path_cz))

    @staticmethod
    def read_file(path):
        numbers = np.loadtxt(path)
        numbers = torch.from_numpy(numbers).float()
        return numbers

    def separate_dataset(self):
        random.shuffle(self.dataset)
        test_samples = int(len(self.dataset) * self.val_ratio)
        self.test_dataset = self.dataset[:test_samples]
        self.train_dataset = self.dataset[test_samples:]


# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# import pdb

# def create_dataloader(df, feature_columns = ['Z_channel'] , target_columns = ['p_arrival_sample','s_arrival_sample'], trace_name_column = ['trace_name'], batch_size = 32, shuffle=True, normalize=True, is_noise = False, train_frac=0.7, val_frac=0.15, test_frac=0.15):
#     # Step 1: Normalization of columns
#     def normalize_multidimensional_row(row):
#             Z = np.array(row['Z_channel'])
#             E = np.array(row['E_channel'])
#             N = np.array(row['N_channel'])
            
#             # Combina i canali in un unico array per trattare l'evento come un'entità unica
#             event = np.stack([Z, E, N], axis=0)  # Shape: (3, N)

#             # Converti in un tensore PyTorch per facilitare la normalizzazione
#             src = torch.from_numpy(event)

#             # Trova il valore massimo assoluto nell'intero evento per normalizzare tutto l'evento come un'entità unica
#             src_max = src.abs().max()

#             # Normalizza l'evento rispetto al suo valore massimo assoluto
#             src_norm = src / src_max

#             # Converti il tensore normalizzato di nuovo in un array numpy
#             event_normalized = src_norm.numpy()

#             # Restituisci i canali come array separati per mantenerli distinguibili nel DataFrame
#             Z_normalized, E_normalized, N_normalized = event_normalized[0], event_normalized[1], event_normalized[2]

#             # Restituisce i canali normalizzati come una serie
#             return pd.Series({'Z_channel': Z_normalized, 'E_channel': E_normalized, 'N_channel': N_normalized})

#     df_normalized = df.apply(normalize_multidimensional_row, axis=1)
#     df_normalized['p_arrival_sample'] = df['p_arrival_sample'].values
#     df_normalized['s_arrival_sample'] = df['s_arrival_sample'].values
#     df_normalized['trace_name'] = df['trace_name'].values

#     # Step 2: Splitting data into training, validation, and test sets
#     def split_data(df_normalized, train_frac, val_frac):
#         assert train_frac + val_frac == 1.0, "Le frazioni devono sommare a 1"
#         train_df, val_df = train_test_split(df_normalized, test_size=1.0 - train_frac, random_state=42)
#         return train_df, val_df
    
#     train_df, val_df = split_data(df_normalized, train_frac, val_frac)
    
#     # Step 3: Converting columns to arrays
#     def convert_column_to_tensor(column):
#         stacked_array = np.stack(column.values)
#         return torch.tensor(stacked_array, dtype=torch.float32)
    
#     def prepare_tensors(df_normalized):
#         feature_tensors = [convert_column_to_tensor(df_normalized[col]) for col in feature_columns]
#         # Stack lungo la nuova dimensione (1) per mantenere i canali separati
#         features = torch.stack(feature_tensors,dim = 1)
#         #pdb.set_trace()
#         length = features.size(0)

#         if is_noise == True:
#             p_sample_tensor = torch.zeros(length)
#             s_sample_tensor = torch.zeros(length)
        
#         else:
#             s_sample_array = np.array(df_normalized['s_arrival_sample'].tolist(), dtype=float).round(1)
#             s_sample_array = np.round(s_sample_array, 1) 

#             p_sample_array = np.array(df_normalized['p_arrival_sample'].tolist(), dtype=float).round(1)
#             p_sample_array = np.round(p_sample_array, 1)  

#             p_sample_tensor = torch.tensor(p_sample_array, dtype=torch.float32)
#             s_sample_tensor = torch.tensor(s_sample_array, dtype=torch.float32)
           
            
#         targets = torch.cat([p_sample_tensor.unsqueeze(1), s_sample_tensor.unsqueeze(1)], dim=1)
        
#         df_normalized = df_normalized.reset_index(drop=True)
#         indices = torch.arange(len(df_normalized))
        
#         index_to_trace_name = {index: name for index, name in enumerate(df_normalized[trace_name_column])}
        
#         return TensorDataset(indices, features, targets), index_to_trace_name
    
#     # Step 4: Creating DataLoader instances
#     train_dataset, index_to_trace_name = prepare_tensors(train_df)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    
#     val_dataset, _ = prepare_tensors(val_df)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
#     return train_loader, val_loader, index_to_trace_name


if __name__ == "__main__":
    from tqdm import tqdm
    dataset_path = "/data/lost+found/share/datasets/earch_dataset/"
    df = pd.read_pickle(dataset_path + "df_train.csv")
    df_noise = pd.read_pickle(dataset_path + "df_noise_train.csv")

    print('Uploading data')
    # args = cp.configure_args()    
    tr_dl, val_dl, index_train = create_dataloader(df, batch_size = 12, is_noise = False, train_frac=0.9, val_frac=0.1, 
                                                              test_frac=0.1) # type: ignore
    tr_dl_noise, val_dl_noise, index_noise = create_dataloader(df, batch_size = 12, is_noise = True, 
                                                                                train_frac=0.9, val_frac=0.1, 
                                                                                test_frac=0.1)
    
    for i in tr_dl:
        print(i)
        import pdb
        pdb.set_trace

    # for eq_in, noise_in in zip(tr_dl, tr_dl_noise):
    #     # print(eq_in.size())




