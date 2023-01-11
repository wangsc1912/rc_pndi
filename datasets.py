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
        # self.choose_func = choose_func

        if type(path) is str:
            self.data, self.targets = torch.load(path)
        elif type(path) is tuple:
            self.data, self.targets = path[0], path[1]
        else:
            print('wrong path type')
        # except:
        #     self.data, self.label = path['']
        self.ori_img = self.data

        if crop and self.image_proc:
            self.data = self.data[:, 4: 26, 5: 25]
            # self.data = self.data[:, 5: 25, 5: 25]
        if sampling != 0 and self.image_proc:
            self.data = self.data.unsqueeze(dim=1)
            self.data = F.interpolate(self.data, size=(sampling, sampling))
            self.data = self.data.squeeze()

        if len(self.data[0].shape) > 1:
            plt.figure()
            plt.imshow(self.data[0])
            plt.savefig('downsampled_img')

        num_data = self.data.shape[0]
        # self.data = img_h
        if type(path) is str and self.image_proc:
            self.bin_data = binarize_dataset(self.data, threshold=0.25)

            self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
            # self.ori_img = self.data

            # if num_pixel % num_pulse == 0:
            #     self.data = self.data.view((num_data, num_pulse, -1))
            # else:
            #     self.data = self.data
            self.reshaped_data = reshape(self.bin_data, num_pulse)
            self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)
            self.data = self.reshaped_data

        else:
            self.reshaped_data = torch.squeeze(self.data)
            self.targets = torch.squeeze(self.targets)
        self.transform = transform

    def __getitem__(self, index: int):

        # label = self.label_dict[img_name]
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
        # for cls in self.num_class:
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
            # self.data = self.data[:, 5: 25, 5: 25]
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0:
            # self.data = self.data.ups
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
        # if self.split == 'letters':
        #     target = target - 1
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
            # remap the target numbers
            # for t in crop_class:
                # self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)
            crop_class_temp = copy.deepcopy(crop_class)
            while crop_class_temp:
                t = crop_class_temp.pop()
                self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)

        if self.transform:
            self.data = self.transform(self.data)
        # self.data, self.targets = k
        self.ori_img = self.data

        if crop:
            self.data = self.data[:, 5: 25, 5: 25]
        if sampling != 0:
            # self.data = self.data.ups
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        num_data = self.data.shape[0]
        self.img_h, self.img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = self.img_h * self.img_w

        # binarize
        self.bin_data = binarize_dataset(self.data, threshold=0.25)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)

        # if transform is not None:
        #     self.transform = transform
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

    # def visualize_classes(self, save_dir_path, format='pdf'):
    #     return super().visualize_classes(save_dir_path, format)

    # def visualize_sample(self, save_dir_path, idx=0, cls=0, grid=False):
    #     return super().visualize_sample(save_dir_path, idx, cls, grid)


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
        # self.data, self.targets = k
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
            # remap the target numbers
            # for t in crop_class:
                # self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)
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
            # remap the target numbers
            # for t in crop_class:
                # self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)
            while crop_class:
                t = crop_class.pop()
                self.targets = torch.where(self.targets > t, self.targets - 1, self.targets)

        if crop:
            #TODO: how should we crop the image for fmnist?

            # self.data = self.data[:, 5: 25, 5: 25]
            self.data = self.data[:, 4: 26, 5: 25]
        if sampling != 0:
            # self.data = self.data.ups
            self.data = F.interpolate(self.data, size=(sampling, sampling))

        num_data = self.data.shape[0]
        img_h, img_w = self.data.shape[1], self.data.shape[2]
        num_pixel = img_h * img_w

        # binarize
        self.bin_data = binarize_dataset(self.data, threshold=bin_thres)
        self.reshaped_data = reshape(self.bin_data, num_pulse)
        self.reshaped_data = torch.transpose(self.reshaped_data, dim0=1, dim1=2)

        # if transform is not None:
        #     self.transform = transform

    def __getitem__(self, index: int):
        target = self.targets[index]
        img = self.reshaped_data[index]
        if target in [0, 1, 2, 5]:
            target2 = np.random.randint(0,3)
            img2 = self.letters[target2]
            # transpose the image 
            # img2 = img2.T
            target2 = target2 + 10
        elif target in [3, 4, 6]:
            target2 = np.random.randint(0, 10)
            img2 = self.digits[target2].T
            # transpose the image
            # img2 = img2.T
        
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
            if not soft:
                fashion = torch.load('data/huang_FMNIST_5cls_oldtarget_0211_te.pt')
                digit = torch.load('data/huang_MNIST_10251557_te.pt')
                letter = torch.load('data/huang_EMNIST_letters_02102137_te.pt')
            else:
                fashion = torch.load('data/huang_FMNIST_letters_05082209_te.pt')
                digit = torch.load('data/huang_MNIST_letters_05082157_te.pt')
                letter = torch.load('data/huang_EMNIST_letters_05082209_te.pt')

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


class SizeDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super(SizeDataset, self).__init__()
        digit_letter = np.load(open('data_generate/digit_letter.npz', 'rb'))
        self.digits = digit_letter['digits']
        self.letters = digit_letter['letters']
        #TODO: check this
        self.digits_letters_raw = np.concatenate((self.digits, self.letters))
        self.digits_letters = np.repeat(self.digits_letters_raw, 1000, axis=0)
        self.targets = np.repeat(list(range(13)), 1000)

    def __getitem__(self, index: int):
        return self.digits_letters[index].T, self.targets[index]

    def __len__(self) -> int:
        return len(self.digits_letters)

    def get_new_width(self):
        return self.digits_letters.shape[-1]


class FashionSizeMaterial(torch.utils.data.Dataset):
    def __init__(self, root):
        super(FashionWithMnist, self).__init__()
        fashion = torch.load('fmnist_5cls.pt')
        digit = torch.load('mnist.pt')
        letter = torch.load('emnist_11cls.pt')

        self.data = fashion['reshaped_data']
        self.targets = fashion['target']
        self.ori_img = fashion['data']

        self.digit_data = digit['reshaped_data']
        self.digit_targets = digit['target']
        self.ori_digit_img = digit['data']

        self.letter_data = letter['reshaped_data']
        self.letter_targets = letter['target']
        self.ori_letter_img = letter['data']

        self.ddata_len = self.digit_targets.shape[0]
        self.ldata_len = self.letter_targets.shape[0]
        self.data_len = self.targets.shape[0]

    def __getitem__(self, index:int):
        target = self.targets[index]
        data = self.data[index]

        d_idx = np.random.randint(0, self.ddata_len)
        dtarget = self.digit_targets[d_idx]
        ddata = self.digit_data[d_idx]
        l_idx = np.random.randint(0, self.ldata_len)
        ltarget = self.letter_targets[l_idx]
        ldata = self.letter_data[l_idx]

        return data, ddata, ldata, target, dtarget, ltarget

    def __len__(self):
        return len(self.targets)
