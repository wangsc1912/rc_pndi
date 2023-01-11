import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST, FashionMNIST, EMNIST
import matplotlib.pyplot as plt
import cv2
import os
from utility.utils import binarize_dataset, single_fig_show, reshape
import copy


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 num_pulse,
                 crop=False,
                 transform=None,
                 sampling=0,
                 ori_img=False,
                 choose_func='train_ete'):
        super(SimpleDataset, self).__init__()

        if choose_func == 'train_ete' or choose_func == 'save_feat':
            self.image_proc = True
        else:
            self.image_proc = False

        self.get_ori_img = ori_img

        if type(path) is str:
            self.data, self.targets = torch.load(path)
        elif type(path) is tuple:
            self.data, self.targets = path[0], path[1]
        else:
            print('wrong path type')
        self.ori_img = self.data

        if crop and self.image_proc:
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0 and self.image_proc:
            self.data = self.data.unsqueeze(dim=1)
            self.data = F.interpolate(self.data, size=(sampling, sampling))
            self.data = self.data.squeeze()

        if len(self.data[0].shape) > 1:
            plt.figure()
            plt.imshow(self.data[0])
            plt.savefig('downsampled_img')

        num_data = self.data.shape[0]
        if type(path) is str and self.image_proc:
            self.bin_data = binarize_dataset(self.data, threshold=0.25)

            self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
            self.reshaped_data = reshape(self.bin_data, num_pulse)
            self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)
            self.data = self.reshaped_data

        else:
            self.reshaped_data = torch.squeeze(self.data)
            self.targets = torch.squeeze(self.targets)
        self.transform = transform

    def __getitem__(self, index: int):

        target = self.targets[index]
        img = self.reshaped_data[index]

        # Normalize
        if self.transform:
            img = self.transform(img)

        if self.get_ori_img:
            return img, self.ori_img[index], target
        else:
            return img, target

    def __len__(self):
        return self.data.shape[0]

    def get_new_width(self):
        return self.data.shape[-1]

    @property
    def num_class(self):
        return len(set(self.targets.squeeze().tolist()))

    def visualize_sample(self, save_dir_path, idx=0, cls=0, grid=False):
        '''
        For a certain sample (given by 'idx'), outputs the original img, binarized img,
        and the corresponding pulse_sequences.
        '''
        if cls:
            idx = torch.nonzero(self.targets == cls)[idx][0]
        ori_sample = self.ori_img[idx]
        bin_sample = self.bin_data[idx]
        pulse_sequences = self.reshaped_data[idx]
        single_fig_show(ori_sample, f'ori_sample_cls{cls}_{idx}', save_dir_path, grid, format='pdf')
        single_fig_show(bin_sample, f'bin_sample_cls{cls}_{idx}', save_dir_path, grid, format='pdf')
        single_fig_show(pulse_sequences, f'pulse_sequences_cls{cls}_{idx}', save_dir_path, grid, grid_width=0.5, format='pdf')

    def visualize_reshaping(self, save_dir_path, idx=0, cls=0, grid=False):
        if cls:
            idx = torch.nonzero(self.targets == cls)[idx][0]
        pulse_sequences = self.reshaped_data[idx]

        len_seg = self.img_h
        num_segment = int(pulse_sequences.shape[1] / len_seg)

        for i in range(num_segment):
            sample = pulse_sequences[:, i * len_seg: (i + 1) * len_seg]
            single_fig_show(sample, f'sample_cls{cls}_{idx}_seg{i}', save_dir_path, grid, format='pdf')

    def visualize_classes(self, save_dir_path, format='pdf'):
        '''
        save original images for each class.
        '''
        sample_dict = {}
        for img, target in zip(self.ori_img, self.targets):
            target = target.item()
            if target not in sample_dict.keys():
                sample_dict[target] = img
            if len(sample_dict.keys()) == self.num_class:
                break
        for target, img in sample_dict.items():
            filename = f'class_{target}.jpg'
            filename = os.path.join(save_dir_path, filename)
            cv2.imwrite(filename, img.squeeze().numpy())


class MnistDataset(MNIST, SimpleDataset):
    def __init__(self,
                 root: str,
                 num_pulse: int,
                 crop=False,
                 sampling=0,
                 mode='sim',
                 ori_img=False,
                 split='letters',
                 **kwargs) -> None:
        super(MnistDataset, self).__init__(root, **kwargs)
        self.get_ori_img = ori_img
        self.ori_img = self.data

        if crop:
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0:
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
        num_data = self.data.shape[0]
        img_h, img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = img_h * img_w

        # binarize
        self.bin_data = binarize_dataset(self.data, threshold=0.25)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)
        if mode == 'real':
            self.reshaped_data = torch.squeeze(self.reshaped_data.reshape(num_data, -1)).to(torch.float)

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = self.reshaped_data[index]

        # Normalize
        if self.transform:
            img = self.transform(img)

        if self.get_ori_img:
            return img, self.ori_img[index], target
        else:
            return img, target

    def get_new_width(self):
        return self.reshaped_data.shape[-1]

    @property
    def num_class(self):
        return len(set(self.targets.tolist()))


class EmnistDataset(EMNIST, SimpleDataset):
    def __init__(self,
                 root: str,
                 num_pulse: int,
                 crop_class=[],
                 crop=False,
                 sampling=0,
                 mode='sim',
                 ori_img=False,
                 split='letters',
                 **kwargs) -> None:
        super(EmnistDataset, self).__init__(root, split, **kwargs)

        self.get_ori_img = ori_img

        if crop_class:
            list_data_less_cls = []
            list_target_less_cls = []
            for x, y in zip(self.data, self.targets):
                if y.cpu().numpy() not in crop_class:
                    list_data_less_cls.append(x)
                    list_target_less_cls.append(y)
            data_less_cls = torch.stack(list_data_less_cls, dim=0)
            target_less_cls = torch.tensor(list_target_less_cls)
            self.data = data_less_cls
            self.targets = target_less_cls
            crop_class_temp = copy.deepcopy(crop_class)
            while crop_class_temp:
                t = crop_class_temp.pop()
                self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)

        if self.transform:
            self.data = self.transform(self.data)
        self.ori_img = self.data

        if crop:
            self.data = self.data[:, 5: 25, 5: 25]
        if sampling != 0:
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        num_data = self.data.shape[0]
        self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = self.img_h * self.img_w

        # binarize
        self.bin_data = binarize_dataset(self.data, threshold=0.25)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)

        if mode == 'real':
            self.reshaped_data = torch.squeeze(self.reshaped_data.reshape(num_data, -1)).to(torch.float)

    def __getitem__(self, index: int):
        target = self.targets[index]
        if self.split == 'letters':
            target = target - 1
        img = self.reshaped_data[index]
        if self.get_ori_img:
            return img, self.ori_img[index], target
        else:
            return img, target

    def get_new_width(self):
        return self.reshaped_data.shape[-1]

    @property
    def num_class(self):
        return len(set(self.targets.tolist()))
    
    @property
    def class_set(self):
        return set(self.targets.tolist())


class FmnistDataset(FashionMNIST, SimpleDataset):
    def __init__(self,
                 root,
                 num_pulse,
                 crop_class: list,
                 crop=False,
                 sampling=0,
                 mode='sim',
                 ori_img=False,
                 bin_thres=0.25,
                 **kwargs):
        super(FmnistDataset, self).__init__(root, **kwargs)

        self.get_ori_img = ori_img
        self.ori_img = self.data

        if crop_class:
            list_data_less_cls = []
            list_target_less_cls = []
            for x, y in zip(self.data, self.targets):
                if y.cpu().numpy() not in crop_class:
                    list_data_less_cls.append(x)
                    list_target_less_cls.append(y)
            data_less_cls = torch.stack(list_data_less_cls, dim=0)
            target_less_cls = torch.tensor(list_target_less_cls)
            self.data = data_less_cls
            self.targets = target_less_cls
            while crop_class:
                t = crop_class.pop()
                self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)

        if crop:
            # self.data = self.data[:, 5: 25, 5: 25]
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0:
            # self.data = self.data.ups
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        num_data = self.data.shape[0]
        self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = self.img_h * self.img_w

        # binarize
        self.bin_data = binarize_dataset(self.data, threshold=bin_thres)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)

        # if transform is not None:
        #     self.transform = transform
        if mode == 'real':
            self.reshaped_data = torch.squeeze(self.reshaped_data.reshape(num_data, -1)).to(torch.float)

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = self.reshaped_data[index]
        if self.get_ori_img:
            return img, self.ori_img[index], target
        else:
            return img, target

    def get_new_width(self):
        return self.reshaped_data.shape[-1]
        # return self.reshaped_data.shape[1]


class FashionWithSize(FashionMNIST, SimpleDataset):
    def __init__(self,
                 root,
                 num_pulse,
                 crop_class=[2, 4, 6],
                 crop=False,
                 sampling=0,
                 ori_img=False,
                 bin_thres=0.25,
                 **kwargs):
        super(FashionWithSize, self).__init__(root, **kwargs)

        self.get_ori_img = ori_img
        # self.data, self.targets = k
        self.ori_img = self.data
        digit_letter = np.load(open('data_generate/digit_letter.npz', 'rb'))
        self.digits = digit_letter['digits']
        self.letters = digit_letter['letters']

        if crop_class:
            list_data_less_cls = []
            list_target_less_cls = []
            for x, y in zip(self.data, self.targets):
                if y.cpu().numpy() not in crop_class:
                    list_data_less_cls.append(x)
                    list_target_less_cls.append(y)
            data_less_cls = torch.stack(list_data_less_cls, dim=0)
            target_less_cls = torch.tensor(list_target_less_cls)
            self.data = data_less_cls
            self.targets = target_less_cls
            while crop_class:
                t = crop_class.pop()
                self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)

        if crop:
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0:
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        num_data = self.data.shape[0]
        img_h, img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = img_h * img_w

        # binarize
        self.bin_data = binarize_dataset(self.data, threshold=bin_thres)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = self.reshaped_data[index]
        if target in [0, 1, 2, 5]:
            target2 = np.random.randint(0,3)
            img2 = self.letters[target2]
            target2 = target2 + 10
        elif target in [3, 4, 6]:
            target2 = np.random.randint(0, 10)
            img2 = self.digits[target2].T
        
        if self.get_ori_img:
            return img, img2, self.ori_img[index], target, target2
        else:
            return img, img2, target, target2

    def get_new_width(self):
        return self.reshaped_data.shape[-1]


class FashionWithMnist(torch.utils.data.Dataset):
    def __init__(self, roots_dict={}, soft=False):
        super(FashionWithMnist, self).__init__()

        if roots_dict:
            fashion = torch.load(roots_dict['FMNIST'])
            digit = torch.load(roots_dict['MNIST'])
            letter = torch.load(roots_dict['EMNIST'])
        else:
            fashion = torch.load('data/huang_FMNIST_5cls_oldtarget_0211_te.pt')
            digit = torch.load('data/huang_MNIST_10251557_te.pt')
            letter = torch.load('data/huang_EMNIST_letters_02102137_te.pt')

        self.data = fashion[0]
        self.targets = fashion[1]
        self.ddata = digit[0]
        self.dtargets = digit[1]
        self.ldata = letter[0]
        self.ltargets = letter[1]

        self.ddata_len = self.dtargets.shape[0]
        self.ldata_len = self.ltargets.shape[0]
        self.data_len = self.targets.shape[0]

    def __getitem__(self, index:int):
        target = self.targets[index]
        data = self.data[index]

        d_idx = np.random.randint(0, self.ddata_len)
        dtarget = self.dtargets[d_idx]
        ddata = self.ddata[d_idx]
        l_idx = np.random.randint(0, self.ldata_len)
        ltarget = self.ltargets[l_idx]
        ldata = self.ldata[l_idx]

        return data, ddata, ldata, target, dtarget, ltarget

    def __len__(self):
        return len(self.targets)
