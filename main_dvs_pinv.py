import os
from click import Argument
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import utility.utils as utils
from utility.utils import oect_data_proc_std
from dvs_dataset import DvsTFDataset
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from datetime import datetime

'''OPTION'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-9)
parser.add_argument('--device_name', type=str, default='p_NDI_05s')
parser.add_argument('--device_cnt', type=int, default=1)

# parser.add_argument('--feat_path', type=str, default='10271027_')
# parser.add_argument('--feat_path', type=str, default='_10271250') # 3 class version
# parser.add_argument('--feat_path', type=str, default='10271506_') # 5 class version
parser.add_argument('--feat_path', type=str, default='11222014_final_012cls') # 4 class version
parser.add_argument('--log_dir', type=str, default='')
options = parser.parse_args()

num_cls = 3
img_width = 28

'''PATH'''
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
# TR_FILEPATH = os.path.join(DATA_DIR, 'dvs_processed_tr_12cls.pt')
# TE_FILEPATH = os.path.join(DATA_DIR, 'dvs_processed_te_12cls.pt')
TR_FILEPATH = os.path.join(DATA_DIR, 'dvs_proced_tr_012cls_w28_32frame_s6i5.pt')
TE_FILEPATH = os.path.join(DATA_DIR, 'dvs_proced_te_012cls_w28_32frame_s6i5.pt')
DEVICE_DIR = os.path.join(ROOT_DIR, 'data')

SAVE_PATH = os.path.join(ROOT_DIR, 'log/dvs')
time_str = datetime.now().strftime('%m%d%H%M')
savepath = os.path.join(SAVE_PATH, f'{options.log_dir}{time_str}')

for path in [SAVE_PATH, savepath]:
    if not os.path.exists(path):
        os.mkdir(path)

'''load dataset'''
tr_dataset = DvsTFDataset(TR_FILEPATH)
te_dataset = DvsTFDataset(TE_FILEPATH)

# for dataset in [tr_dataset, te_dataset]:
#     dataset.data = dataset.data[dataset.label != 3]
#     dataset.label = dataset.label[dataset.label != 3]
#     dataset.label = torch.where(dataset.label == 4, torch.tensor(3).to(dataset.label.dtype), dataset.label)
# num tr data
num_tr_data = len(tr_dataset)
tr_loader = DataLoader(tr_dataset, batch_size=num_tr_data, shuffle=False, num_workers=0)
te_loader = DataLoader(te_dataset, batch_size=options.batch, shuffle=False)

'''load device data'''
device_path = os.path.join(DEVICE_DIR, f'{options.device_name}.xlsx')
device_output = oect_data_proc_std(path=device_path,
                                   device_test_cnt=options.device_cnt)
device_output = device_output.to_numpy().astype(np.float32)

'''define model'''
model = nn.Sequential(nn.Linear(in_features=img_width ** 2, out_features=num_cls))

optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
cross_entropy = nn.CrossEntropyLoss()


def feat_extract(savepath=''):
    '''feature extraction'''
    print('Extracting feature')
    tr_feat = []
    for i, (data, label) in enumerate(tr_loader):
        this_batch_size = data.shape[0]
        data = data.view(this_batch_size, -1, img_width ** 2)
        oect_output = utils.batch_rc_feat_extract_in_construction(data, device_output, options.device_cnt,
                                                    1, 5, this_batch_size)
        tr_feat.append(oect_output)
        del oect_output
    # tr_feat = torch.cat(tr_feat, dim=0)

    # test
    te_feat = []
    for i, (data, label) in enumerate(te_loader):
        this_batch_size = data.shape[0]
        data = data.view(this_batch_size, -1, img_width ** 2)
        oect_output = utils.batch_rc_feat_extract(data, device_output, options.device_cnt,
                                                    5, this_batch_size)
        te_feat.append(oect_output)
    # te_feat = torch.cat(te_feat, dim=0)
    if savepath:
        torch.save((tr_feat, te_feat), os.path.join(savepath, 'feat.pt'))
    return tr_feat, te_feat


# first try load feature
try:
    tr_feat, te_feat = torch.load(f'log/dvs/{options.feat_path}/feat.pt')
    print('Use extracted feature')
except:
    print('No existing feature, extract feature from dataset')
    tr_feat, te_feat = feat_extract(savepath=savepath)

# # draw picture for debug
# for i in range(5):
#     plt.figure()
#     plt.imshow(torch.reshape(tr_feat[0][0].detach(), (28, 28)).numpy())
#     plt.colorbar()
#     plt.show()
#     plt.close()

'''training'''
print('start training')
for epoch in range(1):
    # train
    correct_cnt, loss_epoch = 0, 0

    for i, (data, label) in enumerate(tr_loader):
        this_batch_size = data.shape[0]
        oect_output = torch.cat(tr_feat, dim=0)

        # for 2 class
        # label = label - 1
        # multi class
        label_onehot = torch.zeros(label.shape[0], num_cls)
        label_onehot = label_onehot.scatter_(1, label.view(-1, 1).to(torch.long), 1)

        # oect_output = oect_output.view(-1, 1, img_width, img_width)

        # # draw picture for debug
        # for j in range(5):
        #     plt.figure()
        #     plt.imshow(torch.reshape(oect_output[j].detach(), (img_width, img_width)).numpy())
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        # pseudo inverse
        # 2 class
        # readout = torch.linalg.lstsq(oect_output, label.to(oect_output.dtype)).solution
        # multi class
        readout = torch.linalg.lstsq(oect_output, label_onehot.to(oect_output.dtype)).solution

    # test
    correct_cnt = 0
    logics, outputs, labels = [], [], []
    for i, (data, label) in enumerate(te_loader):
        this_batch_size = data.shape[0]
        oect_output = te_feat[i]

        # for bi class
        # label -= 1
        # for multi class
        # label_onehot = torch.zeros(label.shape[0], num_cls)
        # label_onehot = label_onehot.scatter_(1, label.view(1, -1), 1)
        logic = torch.mm(oect_output, readout)

        # for 2 class
        # logic = torch.where(logic > 0.5, torch.tensor(1).to(logic.dtype), torch.tensor(0).to(logic.dtype))
        # logic = logic.squeeze()
        # for multi class
        logic = torch.round(logic)

        # for biclass
        # correct_cnt += torch.sum(logic == label)
        # for multi class
        correct_cnt += torch.sum(torch.argmax(logic, dim=1) == label)

        logics.append(logic)
        outputs.append(torch.argmax(logic, dim=1))
        labels.append(label)

    # outputs
    outputs, labels = torch.cat(outputs), torch.cat(labels)
    # accuracy
    te_acc = correct_cnt / len(te_dataset)
    # log
    print(f'epoch: {epoch}, tr loss: {loss_epoch}, te acc: {te_acc}')


# pca
color = ['coral', 'dodgerblue', 'tan', 'orange', 'green', 'silver', 'chocolate', 'lightblue', 'violet', 'crimson']
color_list = [color[i] for i in labels]
pca = PCA(n_components=2)
outputs_pca = pca.fit_transform(torch.cat(te_feat, 0))
plt.figure()
plt.scatter(outputs_pca[:, 0], outputs_pca[:, 1], c=color_list)
plt.savefig(f'{savepath}/dvs_2cls_pca.pdf')
plt.close()


# conf mat
# text_label = ['arm roll', 'hand clap', 'arm circle', ]
text_label = list(range(num_cls))
conf_mat = confusion_matrix(labels, outputs)
confusion_matrix_df = pd.DataFrame(conf_mat, index=text_label, columns=text_label)
plt.figure(figsize=(num_cls, num_cls))
sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.savefig(f'{savepath}/conf_mat_dvs.pdf', format='pdf')
plt.close()
normed_conf_mat = conf_mat / np.expand_dims(conf_mat.sum(1), -1)
normed_confusion_matrix_df = pd.DataFrame(normed_conf_mat, index=text_label, columns=text_label)
plt.figure()
sns.heatmap(normed_confusion_matrix_df, annot=True, fmt='.2f', cmap=plt.cm.Blues)
plt.savefig(f'{savepath}/normed_conf_mat_dvs.pdf', format='pdf')
plt.close()
print('Confusion matrix saved')


if __name__ == '__main__':
    pass
