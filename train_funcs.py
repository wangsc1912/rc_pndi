import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import utility.utils as utils
import time


def get_dataset_statics(num_data,
                        data_loader):
    '''
    for extracted feature.
    '''
    mean_list = []
    std_list = []
    data_max, data_min = 0, 0
    for data, target in tqdm(data_loader):
        data = data.to(torch.float)
        mean = data.to(torch.float).mean(dim=-1).sum(0)
        std = data.to(torch.float).std(dim=-1).sum(0)
        max = data.max()
        min = data.min()
        mean_list.append(mean)
        std_list.append(std)
        data_max = max if max > data_max else data_max
        data_min = min if min < data_min else data_min
    mean = sum(mean_list) / num_data
    std = sum(std_list) / num_data

    print(f'mean: {mean}, std: {std}, max: {data_max}, min: {data_min}')


def test(num_data,
         num_class,
         batchsize,
         test_loader,
         model,
         criterion,
         save_dir_name,
         ):
    # test
    te_accs = []
    te_losses = []
    te_outputs = []
    targets = []
    with torch.no_grad():
        for i, (data, img, target) in enumerate(test_loader):

            this_batch_size = len(data)

            data = data.to(torch.float)

            output = F.softmax(model(data.squeeze()), dim=-1)
            loss = criterion(output.unsqueeze(0), target)
            te_outputs.append(output)
            acc = torch.sum(output.argmax(dim=-1) == target) / this_batch_size
            te_accs.append(acc)
            te_losses += loss.cpu().numpy()
            targets.append(target)
        te_acc = (sum(te_accs) * batchsize / num_data).numpy()
        te_loss = te_losses / num_data

        # log infos
        log = "test acc: %.6f" % te_acc
        print(log)

        if batchsize == 1:
            te_outputs = torch.stack(te_outputs, dim=0)
        else:
            te_outputs = torch.cat(te_outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        # confusion matrix
        conf_mat = confusion_matrix(targets, torch.argmax(te_outputs, dim=-1))

        conf_mat_dataframe = pd.DataFrame(conf_mat,
                                        index=list(range(num_class)),
                                        columns=list(range(num_class)))

        conf_mat_normalized = conf_mat_dataframe.divide(conf_mat_dataframe.sum(axis=1), axis=0)

        return te_acc, te_loss, conf_mat, conf_mat_normalized


def train_with_feature(num_data, num_te_data,
                       num_class, num_epoch,
                       batchsize, te_batchsize,
                       train_loader, test_loader,
                       model, optimizer,
                       scheduler, criterion,
                       dataset, save_dir_name):

    start_time = time.time()
    acc_list = []
    loss_list = []
    log_list = []
    test_acc_list = [] 
    test_loss_list = []
    conf_mat_list = []
    for epoch in range(num_epoch):

        acc = []
        loss = 0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(torch.float).squeeze()
            # readout layer
            logic = F.softmax(model(data), dim=-1)

            batch_loss = criterion(logic, target)
            loss += batch_loss
            batch_acc = torch.sum(logic.argmax(dim=-1) == target) / batchsize
            acc.append(batch_acc)
            batch_loss.backward()
            optimizer.step()
        te_acc, te_loss, conf_mat, conf_mat_normalized = test(num_te_data,
                                                               num_class,
                                                               te_batchsize,
                                                               test_loader,
                                                               model,
                                                               criterion,
                                                               save_dir_name)
        test_acc_list.append(te_acc)
        test_loss_list.append(te_loss)
        if te_acc == max(test_acc_list):
            # save readout layer
            torch.save(model, os.path.join(save_dir_name, f'{dataset}_{te_acc*1e5:.0f}.pt'))
        scheduler.step()
        acc_epoch = (sum(acc) * batchsize / num_data).numpy()
        acc_list.append(acc_epoch)
        loss_list.append(loss)

        epoch_end_time = time.time()
        if epoch == 0:
            epoch_time = epoch_end_time - start_time
        else:
            epoch_time = epoch_end_time - epoch_start_time
        epoch_start_time = epoch_end_time

        # log info
        log = "epoch: %d, loss: %.4f, acc: %.6f, test acc: %.6f, time: %.2f" % (epoch, loss, acc_epoch, te_acc, epoch_time)
        print(log)
        log_list.append(log + '\n')
    utils.write_log(save_dir_name, log_list)
    # save results
    np.savez(os.path.join(save_dir_name, f'{dataset}_train_results.npz'), acc_list=acc_list, te_acc_list=test_acc_list, loss_list=loss_list, te_loss_list=test_loss_list,
             conf_mats=conf_mat_list)
    return os.path.join(save_dir_name, f'{dataset}_{max(test_acc_list)*1e5:.0f}.pt')


def save_rc_feature(train_loader,
                    test_loader,
                    num_pulse,
                    device_output,
                    device_tested_number,
                    filename):
    device_features = []
    tr_targets = []

    for i, (data, target) in enumerate(train_loader):
        this_batch_size = len(data)
        # reservoir output
        oect_output = utils.batch_rc_feat_extract(data,
                                                device_output,
                                                device_tested_number,
                                                num_pulse,
                                                this_batch_size)


        device_features.append(oect_output)
        tr_targets.append(target)
    tr_features = torch.cat(device_features, dim=0)
    tr_targets = torch.cat(tr_targets).squeeze()

    te_oect_outputs = []
    te_targets = []
    for i, (data, im, target) in enumerate(test_loader):
        this_batch_size = len(data)
        # reservoir output
        oect_output = utils.batch_rc_feat_extract(data,
                                                device_output,
                                                device_tested_number,
                                                num_pulse,
                                                this_batch_size)

        te_oect_outputs.append(oect_output)
        te_targets.append(target)
    te_features = torch.cat(te_oect_outputs, dim=0)
    te_targets = torch.cat(te_targets).squeeze()

    tr_filename = filename + f'_tr.pt'
    te_filename = filename + f'_te.pt'
    torch.save((tr_features, tr_targets), tr_filename)
    torch.save((te_features, te_targets), te_filename)
    print('data_saved')


def test_fashion_size(# device_output,
                      num_data,
                      batchsize,
                      epoch,
                      test_loader,
                      fmnist_model, mnist_model, emnist_model):

    te_acc, te_acc2, te_acc_total = [], [], []
    te_targets, te_outputs, te_targets2, te_outputs2 = [], [], [], []
    feat = []
    with torch.no_grad():
        for e in range(epoch):
            for (data, ddata, ldata, target, dtarget, ltarget) in test_loader:
                te_logic = F.softmax(fmnist_model(data), dim=-1)

                te_batch_acc = torch.sum(te_logic.argmax(dim=-1) == target) / batchsize
                te_acc.append(te_batch_acc)

                te_targets.append(target)
                te_outputs.append(te_logic.argmax(dim=-1))

                # use letters
                if target in [0, 1, 2, 3]:
                    te_logic2 = F.softmax(emnist_model(ldata), dim=-1)

                    te_logic2_non_onehot = te_logic2.argmax(dim=-1) + 10 # old targets
                    te_targets2.append(ltarget.squeeze() + 10) # old targets

                    # for ann
                    feat2 = ldata.squeeze()

                # use digits
                elif target in [4,]:
                    te_logic2 = F.softmax(mnist_model(ddata), dim=-1)
                    te_logic2_non_onehot = te_logic2.argmax(dim=-1)
                    te_targets2.append(dtarget.squeeze())
                    feat2 = ddata.squeeze()

                feat.append(torch.cat((data.squeeze(), feat2), dim=0))

                te_outputs2.append(te_logic2_non_onehot.squeeze())

                te_batch_acc_total = torch.sum((te_logic.argmax(dim=-1) == target) * (te_logic2.argmax(dim=-1) == dtarget)) / batchsize
                te_acc_total.append(te_batch_acc_total)

        te_outputs2 = torch.stack(te_outputs2, dim=0)
        te_targets2 = torch.stack(te_targets2, dim=0)
        te_targets = torch.stack(te_targets, dim=0)
        te_outputs = torch.stack(te_outputs, dim=0)
        # calcutate the overall targets/logic/acc/conf_mat
        te_target_rearange = torch.zeros((num_data * epoch))
        for i, (target, target2) in enumerate(zip(te_targets, te_targets2)):
            te_target_rearange[i] = target * 13 + target2

        te_output_rearange = torch.zeros((num_data * epoch))
        # outputs are alreated to be non-one-hot
        for i, (output, output2) in enumerate(zip(te_outputs, te_outputs2)):
            te_output_rearange[i] = output * 13 + output2

        total_acc = (te_target_rearange == te_output_rearange).sum() / te_target_rearange.shape[0]
        fashion_acc = (te_targets.squeeze() == te_outputs.squeeze()).sum() / te_targets.shape[0]
        print(f'total acc: {total_acc}, fashion acc: {fashion_acc}')
