import torch as t
import torch.utils.data as Data
import pandas as pd
import csv
import random
import numpy as np
import sys
import argparse
device = t.device("cuda:2" if t.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
args = parser.parse_args()  # 会将命令行参数转换为Python对象，可以通过属性来访问这些参数的值。
args.dd2 = True
args.data_dir = 'data/data/'
args.m_drug_d_num = 2060
args.m_mRNA_d_num = 3929
args.m_incRNA_d_num = 2459
def setup_seed(seed):
     t.manual_seed(seed)
     t.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     t.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(123)

# 将miRNA、disease名字转为索引
def make_index(data,sample):
    sample_index = []
    for i in range(sample.shape[0]):
        idx = np.where(sample[i][0] == data['miRNA_disease'])
        idy = np.where(sample[i][1] == data['miRNA_disease'])
        sample_index.append([idx[0].item(), idy[0].item()])
    sample_index = np.array(sample_index)
    return sample_index
# 保存成csv文件    
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
# 加载数据集
def loading():
    data = dict()
    # 读取所有md_sample样本
    data['all_sample'] = pd.read_csv(args.data_dir + 'all_sample.csv', header=None).iloc[:,:].values
    # 读取miRNA和disease的名字
    data['miRNA'] = pd.read_csv(args.data_dir + 'miRNA.csv', header=None).iloc[:, :].values
    data['disease'] = pd.read_csv(args.data_dir + 'disease.csv', header=None).iloc[:, :].values
    data['miRNA_disease'] = np.concatenate((data['miRNA'], data['disease']), axis=0)
    # 读取融合了GIP的miRNA和disease的特征
    data['miRNA_disease_feature'] = pd.read_csv(args.data_dir + 'miRNA_disease_feature.csv', header=None).iloc[:,:].values
    # 加载构建drug的图adj的数据
    data['m_drug_d_sample'] = pd.read_csv(args.data_dir + 'm_drug_d_sample.csv', header=None).iloc[:,:].values
    data['m_drug_drug_d_sample'] = pd.read_csv(args.data_dir + 'm_drug_drug_d_sample.csv', header=None).iloc[:, :].values
    # 加载构建mRNA的图adj的数据
    data['m_mRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_mRNA_d_sample.csv', header=None).iloc[:, :].values
    data['m_mRNA_mRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_mRNA_mRNA_d_sample.csv', header=None).iloc[:, :].values
    # 加载构建incRNA的图adj的数据
    data['m_incRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_incRNA_d_sample.csv', header=None).iloc[:, :].values
    data['m_incRNA_incRNA_d_sample'] = pd.read_csv(args.data_dir + 'm_incRNA_incRNA_d_sample.csv', header=None).iloc[:,:].values
    return data





if __name__ == '__main__':
    dataset = loading()
    association = pd.read_csv('data/data/association.csv', header=None).iloc[:,:].values  # MiRNA-disease关联矩阵
    # 将未知的关系中找miRNA和疾病之间有关联的miRNA下标
    test_sample_index = []
    for i in range(association.shape[0]):
        for j in range(association.shape[1]):
            if j == 11 and association[i][j] == 0:                    #将所有与疾病相关的miRNA坐标取出
                test_sample_index.append([i, j])
    test_sample_index = np.array(test_sample_index)
    tran_sample = dataset['all_sample'][:,:2]
    tran_sample_index = make_index(dataset, tran_sample)
    tran_label = dataset['all_sample'][:,2]
    dataset['m_drug_drug_d_sample'] = np.concatenate((dataset['m_drug_drug_d_sample'], tran_sample), axis=0)
    dataset['m_mRNA_mRNA_d_sample'] = np.concatenate((dataset['m_mRNA_mRNA_d_sample'], tran_sample), axis=0)
    dataset['m_incRNA_incRNA_d_sample'] = np.concatenate((dataset['m_incRNA_incRNA_d_sample'], tran_sample), axis=0)
    # 构造drug图
    file_name1 = "all_m_drug_d_adj.csv"
    dataset['m_drug_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:,:].values
    # 构造mRNA图
    file_name1 = "all_m_mRNA_d_adj.csv"
    dataset['m_mRNA_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values
    # 构造incRNA图
    file_name1 = "all_m_incRNA_d_adj.csv"
    dataset['m_incRNA_d_adj'] = pd.read_csv(args.data_dir + file_name1, header=None).iloc[:, :].values
    # 将字典中的数组转换为张量
    dataset['m_drug_d_adj'] = t.FloatTensor(dataset['m_drug_d_adj'])
    dataset['m_mRNA_d_adj'] = t.FloatTensor(dataset['m_mRNA_d_adj'])
    dataset['m_incRNA_d_adj'] = t.FloatTensor(dataset['m_incRNA_d_adj'])
    tran_sample_index = t.FloatTensor(tran_sample_index).to(device)
    tran_label = t.FloatTensor(tran_label.astype(int)).to(device)
    test_sample_index = t.FloatTensor(test_sample_index).to(device)

    # map_location:指定设备，cpu或者GPU
    model = t.load("./save_model/no5fold/no5fold_train_model.pth", map_location={'2':'GPU'})
    model.eval()
    with t.no_grad():
        train_score, test_score = model(dataset, tran_sample_index, test_sample_index, device)
        predictions = np.rint(test_score.detach().cpu().numpy()).astype(np.int64).tolist()
    predictions = [item for sublist in predictions for item in sublist]
    np.set_printoptions(threshold=sys.maxsize)
    print(predictions)
    np.savetxt('./data/lymphoma_carcinoma_predictions.txt', predictions, fmt="%d", comments='')
    np.savetxt('./data/lymphoma_index.txt', test_sample_index.detach().cpu().numpy(), fmt="%d", comments='')

