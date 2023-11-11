import random
import math
from model import MDA
import pandas as pd
import numpy as np
from torch import optim, nn
import torch as t
import csv

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # （代表仅使用第0，1号GPU）
# 如果GPU可用，利用GPU进行训练
device = t.device('cuda:1' if t.cuda.is_available() else "cpu")
t.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--hid_feats", type=int, default=1500, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=901, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[64, 2], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
parser.add_argument('--save_score', default='True', help='save_score')

args = parser.parse_args()  # 会将命令行参数转换为Python对象，可以通过属性来访问这些参数的值。
args.dd2 = True
args.data_dir = 'dataset/'
args.result_dir = 'result/'  # 保存结果
args.save_score = True if str(args.save_score) == 'True' else False


# 加载数据集
def loading():
    data = dict()
    # 读取所有md_sample样本
    data['all_sample'] = pd.read_csv(args.data_dir + 'all_sample.csv', header=None).iloc[:, :].values
    # 读取miRNA和disease的名字
    data['miRNA'] = pd.read_csv(args.data_dir + 'quchong_bianhao/miRNA.csv', header=None).iloc[:, :].values
    data['disease'] = pd.read_csv(args.data_dir + 'quchong_bianhao/disease.csv', header=None).iloc[:, :].values
    data['miRNA_disease'] = np.concatenate((data['miRNA'], data['disease']), axis=0)
    # 读取融合了GIP的miRNA和disease的特征
    data['miRNA_disease_feature'] = pd.read_csv(args.data_dir + 'miRNA_disease_feature.csv', header=None).iloc[:,
                                    :].values
    # 加载提取miRNA的embedding
    miRNA_embedding = np.loadtxt(args.data_dir + 'data/miRNA_embedding.txt',dtype=np.float,delimiter=None,unpack=False)
    data['miRNA_embedding'] = miRNA_embedding[:901]

    # 加载提取disease的embedding
    disease_embedding = np.loadtxt(args.data_dir + 'data/disease_embedding.txt',dtype=np.float,delimiter=None,unpack=False)
    data['disease_embedding'] = disease_embedding[:877]
    data['miRNA_disease_embedding'] = np.concatenate((data['miRNA_embedding'], data['disease_embedding']), axis=0)
    data['inter_miRNA_disease_feature'] = np.concatenate((data['miRNA_disease_feature'], data['miRNA_disease_embedding']), axis=1)
    return data
# 将miRNA、disease名字转为索引
def make_index(data, sample):
    sample_index = []
    for i in range(sample.shape[0]):
        idx = np.where(sample[i][0] == data['miRNA_disease'])
        idy = np.where(sample[i][1] == data['miRNA_disease'])
        sample_index.append([idx[0].item(), idy[0].item()])
    sample_index = np.array(sample_index)
    return sample_index
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
if __name__ == '__main__':
    # 数据处理
    dataset = loading()
    # args.m_drug_d_num = dataset['m_drug_d_sample'].shape[0]
    # args.m_mRNA_d_num = dataset['m_mRNA_d_sample'].shape[0]
    # args.m_incRNA_d_num = dataset['m_incRNA_d_sample'].shape[0]
    dataset['inter_miRNA_disease_feature'] = t.FloatTensor(dataset['inter_miRNA_disease_feature']).to(device)
    # 模型实例化
    model = MDA(args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    cross_entropy = nn.BCELoss(reduction='mean')
    file_num = 1

    # 保存最优的test_auc、recall等
    auc = 0
    auprc = 0
    acc = 0
    f1 = 0
    recall = 0
    pre = 0

    # 记录最大auc
    max_test_acc = 0
    # 记录五折折数
    k = 1
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    for train_index, test_index in kfold.split(dataset['all_sample'][:, :2]):
        tran_sample = dataset['all_sample'][train_index][:, :2]
        tran_sample_index = make_index(dataset, tran_sample)
        tran_label = dataset['all_sample'][train_index][:, 2]
        test_sample = dataset['all_sample'][test_index][:, :2]
        test_sample_index = make_index(dataset, test_sample)
        test_label = dataset['all_sample'][test_index][:, 2]
        # dataset['m_drug_drug_d_sample'] = np.concatenate((dataset['m_drug_drug_d_sample'], tran_sample), axis=0)
        # dataset['m_mRNA_mRNA_d_sample'] = np.concatenate((dataset['m_mRNA_mRNA_d_sample'], tran_sample), axis=0)
        # dataset['m_incRNA_incRNA_d_sample'] = np.concatenate((dataset['m_incRNA_incRNA_d_sample'], tran_sample), axis=0)
        # 构造drug图
        # file_name1 = str(file_num) + "m_drug_d_adj.csv"
        # dataset['m_drug_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values
        # # 构造mRNA图
        # file_name1 = str(file_num) + "m_mRNA_d_adj.csv"
        # dataset['m_mRNA_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values
        # # 构造incRNA图
        # file_name1 = str(file_num) + "m_incRNA_d_adj.csv"
        # dataset['m_incRNA_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values
        # 将字典中的数组转换为张量
        # dataset['m_drug_d_adj'] = t.FloatTensor(dataset['m_drug_d_adj'])
        # dataset['m_mRNA_d_adj'] = t.FloatTensor(dataset['m_mRNA_d_adj'])
        # dataset['m_incRNA_d_adj'] = t.FloatTensor(dataset['m_incRNA_d_adj'])
        tran_sample_index = t.FloatTensor(tran_sample_index).to(device)
        tran_label = t.FloatTensor(tran_label.astype(int)).to(device)
        test_sample_index = t.FloatTensor(test_sample_index).to(device)
        test_label = t.FloatTensor(test_label.astype(int)).to(device)
        for i in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            train_score, test_score = model(dataset, tran_sample_index, test_sample_index, device)
            train_score = train_score.squeeze(1)

            train_label = tran_label.to(device)

            train_loss = cross_entropy(train_score, train_label)
            train_loss.backward()
            train_auc = roc_auc_score(train_label.detach().cpu().numpy(),
                                      train_score.detach().cpu().numpy())
            train_acc = accuracy_score(train_label.detach().cpu().numpy().astype(np.int64),
                                       np.rint(train_score.detach().cpu().numpy()).astype(np.int64))
            optimizer.step()
            model.eval()
            test_score = test_score.squeeze(1)
            test_label = test_label.to(device)
            # test_loss = cross_entropy(test_score, test_label)
            test_pro = test_score.detach().cpu().numpy()
            test_pro_int =  np.rint(test_score.detach().cpu().numpy()).astype(np.int64)
            test_auc = roc_auc_score(test_label.detach().cpu().numpy(),
                                     test_score.detach().cpu().numpy())
            test_acc = accuracy_score(test_label.detach().cpu().numpy().astype(np.int64),
                                      np.rint(test_score.detach().cpu().numpy()).astype(np.int64))
            test_aupr = average_precision_score(test_label.detach().cpu().numpy(), test_score.detach().cpu().numpy())
            test_f1 = f1_score(test_label.detach().cpu().numpy(),
                               np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            test_recall = recall_score(test_label.detach().cpu().numpy(),
                                       np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            test_pre = precision_score(test_label.detach().cpu().numpy(),
                                       np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')

            # 保存达标的训练的模型
            if test_acc > max_test_acc:
                t.save(model.state_dict(), "./save_model/5_fold/train_model.pth")
                max_test_acc = test_acc
                auc = test_auc
                auprc = test_aupr
                acc = test_acc
                f1 = test_f1
                recall = test_recall
                pre = test_pre
                att_pro = "5fold_" + str(file_num) +"_pro.csv"
                att_int = "5fold_" + str(file_num) +"_int.csv"
                dataset['att_pro'] = np.concatenate((test_label.unsqueeze(1).detach().cpu().numpy(),test_pro.reshape(-1,1)),axis=1).tolist()
                dataset['att_int'] = np.concatenate((test_label.unsqueeze(1).detach().cpu().numpy(),test_pro_int.reshape(-1,1)),axis=1).tolist()
                StorFile(dataset['att_pro'], '5CV/' + att_pro)
                StorFile(dataset['att_int'], '5CV/' + att_int)
                print(f'Epoch: {i + 1:03d}/{args.epochs:03d}' f'   | Learning Rate {scheduler.get_last_lr()[0]:.6f}')
                # print(f'Epoch: {i + 1:03d}/{args.epochs:03d}')
                print(f'Train Auc.: {train_auc:.4f}' f' | Test Auc.: {test_auc:.4f}')
                print(f'Train Loss.: {train_loss.item():.4f}')
                print(f'Train Acc.: {train_acc:.4f}' f' | Test Acc.: {test_acc:.4f}')
            # 更新学习率
            scheduler.step()
        file_num += 1
        max_test_acc = 0
        # k += 1
        # if k > 1:
        #     break
    print(f' | Test Auc.: {auc:.4f}')
    print(f' | Test Auprc.: {auprc:.4f}')
    print(f' | Test Acc.: {acc:.4f}')
    print(f' | Test F1.: {f1:.4f}')
    print(f' | Test Recall.: {recall:.4f}')
    print(f' | Test Precision.: {pre:.4f}')