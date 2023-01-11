import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
import sys
sys.path.append('.')


class DvsTFDataset(Dataset):
    def __init__(self, path) -> None:
        super(DvsTFDataset, self).__init__()
        self.data, self.label = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


class DvsDataset(Dataset):
    def __init__(self, DATADIR, train, num_points=1024, use_raw=True):
        super(DvsDataset, self).__init__()
        self.num_points = num_points
        self.use_raw = use_raw

        if self.use_raw:
            self.dataset_dir = os.path.join(
                DATADIR, "train") if train else os.path.join(DATADIR, "test")
            files = os.listdir(self.dataset_dir)
            print("processing dataset:{} ".format(self.dataset_dir))
        else:
            files = getDataFiles(os.path.join(DATADIR, 'train_files.txt')) if train else getDataFiles(
                os.path.join(DATADIR, 'test_files.txt'))
            print("processing dataset:{} ".format(DATADIR))

        self.data, self.label = [], []
        if self.use_raw:
            for f in files:
                with open(os.path.join(self.dataset_dir, f), 'rb') as f:
                    dataset = pickle.load(f)
                self.data += dataset['data']
                self.label += dataset['label'].tolist()
        else:
            for f in files:
                d, l = loadDataFile(os.path.join(DATADIR, f))
                self.data.append(d)
                self.label.append(l)
            self.data = np.concatenate(self.data, axis=0).squeeze()
            self.label = np.concatenate(self.label, axis=0).squeeze()

    def __getitem__(self, index):
        if self.use_raw:
            label = int(self.label[index])
            events = self.data[index]
            nr_events = events.shape[0]
            idx = np.arange(nr_events)
            np.random.shuffle(idx)
            idx = idx[0: self.num_points]
            events = events[idx, ...]

            events_normed = np.zeros_like(events, dtype=np.float32)
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            events_normed[:, 1] = x / 127
            events_normed[:, 2] = y / 127
            t = t - np.min(t)
            t = t / np.max(t)
            events_normed[:, 0] = t

            return events_normed, label
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    DATADIR = 'data/DVS_C10_TS1_1024'
    tr = DvsDataset(DATADIR, train=True)
    length = len(tr)
    print(length)
