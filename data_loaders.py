import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from scipy import signal
from scipy.signal import resample
import yaml

def get_dataloaders(data_name, train_val_test, resampling_rate, 
                    signal_transform=None):
    with open("./dataset_configs.yaml", "r") as ymlfile:
        data_config = yaml.safe_load(ymlfile)
    
    if data_name == 'WORKLOAD':
        t_dataset = Workloadloader(data_dir= os.path.expanduser(data_config['WORKLOAD']['data_dir']),
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)

    elif data_name == 'EarEEG':
        t_dataset = EarEEGloader(data_dir= data_config['EarEEG']['data_dir'],
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)

    elif data_name == 'ISRUC':
        t_dataset = ISRUCloader(data_dir= data_config['ISRUC']['data_dir'],
                                train_val_test=train_val_test,
                                resampling_rate=resampling_rate,
                                signal_transform=signal_transform)
    
    return t_dataset

class Workloadloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 200
        self.signal_len = 4
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform

        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings from Workload: ", len(self.data_files))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx): 
        with open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb') as f:
            signal_data = pickle.load(f)
        X = np.array(signal_data["X"])

        # filter first 16 channels
        X = X[:16,:]
        labels = signal_data["y"]
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        if self.signal_transform == 'stft':
            window_len = self.resampling_rate # 1s
            hop_len = self.resampling_rate//2 #0.5s
            X_fft = []
            for i in range(X.shape[0]):
                f, _, Zxx = signal.stft(X[i], 
                            fs=self.resampling_rate, 
                            nperseg=window_len, 
                            noverlap=window_len-hop_len,
                            return_onesided=True,boundary=None,
                            padded = False, scaling='spectrum')
                X_fft.append(np.abs(Zxx))
            X = np.stack(X_fft, axis=0) # (batch, channels, freqs, timesamples)
            X = X[:,:-1,:]
        
        X = torch.FloatTensor(X)
        
        return X, labels

class EarEEGloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 256, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 250
        self.signal_len = 30
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        self.data_files = os.listdir(os.path.join(data_dir,train_val_test))
        print("Number of recordings from EarEEG: ", len(self.data_files))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx): 
        with open(os.path.join(self.data_dir,self.train_val_test,self.data_files[idx]), 'rb') as f:
            signal_data = pickle.load(f)
        X = np.array(signal_data["X"])
        X = X[:-1,:]
        labels = int(signal_data["label"])
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        X = torch.FloatTensor(X)
        
        return X, labels
    
    
class ISRUCloader(Dataset):
    def __init__(self, data_dir, train_val_test, resampling_rate = 200, signal_transform=None):
        
        self.data_dir = data_dir
        self.train_val_test = train_val_test
        self.default_sampling_rate = 200
        self.signal_len = 30
        self.resampling_rate = resampling_rate
        self.signal_transform = signal_transform
        
        from pathlib import Path
        self.pf_path = Path(data_dir) / f"{train_val_test}.mdb"
        self.tables = pd.read_parquet(Path(data_dir) / f"ISRUC.{train_val_test}.index.parquet")
        
        print("Number of recordings from ISRUC: ", self.tables.shape)
        # print("Number of recordings from ISRUC labels: ", self.labels.shape)
            
        
    def __len__(self):
        return self.tables.shape[0]
    
    def __getitem__(self, idx): 
        DATA_DTYPE = np.dtype([('features', 'f8', (6, 6000)), ('label', 'i8')])

        import lmdb
        import lz4.frame
        import struct

        cache = getattr(self, 'cache', None)
        if cache is None:
            cache = lmdb.open(str(self.pf_path), 
                readonly=True, 
                lock=False, 
                readahead=True)

            self.cache = cache
        
        with cache.begin() as txn:
            compressed_val = txn.get(struct.pack('!Q', idx))
            
            if compressed_val:
                raw_bytes = lz4.frame.decompress(compressed_val)
                buffer = np.frombuffer(raw_bytes, dtype=DATA_DTYPE)[0]

                X, labels = buffer['features'], buffer['label']
        
        # resample
        if self.default_sampling_rate != self.resampling_rate:
            X = resample(X, self.resampling_rate*self.signal_len , axis=-1)
        
        # Normalize
        X = X/(np.quantile(np.abs(X), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
        
        
        X = torch.FloatTensor(X)
        
        return X, labels