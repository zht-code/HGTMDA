from torch import nn
import torch as torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
from torch_geometric.nn import GATConv
import scipy.sparse as sp
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv


class MS_CAM(nn.Module):
 

    def __init__(self, channels=3, r=2):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class MultiGraphConvolution_Layer(nn.Module):

    def __init__(self, in_features, out_features):
        super(MultiGraphConvolution_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # self.view_conv1 = GATConv(in_features, out_features, heads=3, dropout=0.05, concat=False)
        # self.view_conv2 = GATConv(out_features, out_features, heads=3, dropout=0.05, concat=False)
        self.view_conv1 = GATConv(in_features, out_features)
        # self.view_conv2 = GATConv(out_features, out_features)

    def forward(self, input_x, adj, device):
        sum_x = torch.zeros((0, input_x.shape[0], self.out_features)).to(device)
        adj_temp = adj.numpy()

        adj_temp = sp.coo_matrix(adj_temp)
        # 转换为 PyTorch Geometric 中的数据对象
        edge_index, edge_weight = utils.from_scipy_sparse_matrix(adj_temp)
        input_x = input_x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        input_x_view_conv1 = F.relu(self.view_conv1(input_x, edge_index, edge_weight))
        input_x_view_conv2 = torch.unsqueeze(input_x_view_conv1, dim=0)
        sum_x = torch.cat((sum_x, input_x_view_conv2), 0)
        return sum_x


class MGC_Model(nn.Module):

    def __init__(self, feature_num, hidden_num, out_num):
        super(MGC_Model, self).__init__()
        self.feature_num = feature_num
        self.hidden_num = hidden_num  # hidden_num = 1 or feature_num
        self.mgc = MultiGraphConvolution_Layer(in_features=feature_num, out_features=out_num)

    def forward(self, input_x, adj, device):
        x = self.mgc(input_x, adj, device)
        x = F.relu(x)

        # x_ce = self.linear_ce(x)
        # x_b = self.linear_b(input_x)
        # x = x_b + x_ce

        # x = torch.sigmoid(x)
        return x


class MDA(nn.Module):
    def __init__(self, args):
        super(MDA, self).__init__()
        self.args = args
        # self.gat_m_drug_m = MGC_Model(args.m_drug_d_num, args.hid_feats, args.out_feats)
        # self.gat_m_mRNA_d = MGC_Model(args.m_mRNA_d_num, args.hid_feats, args.out_feats)
        # self.gat_m_incRNA_d = MGC_Model(args.m_incRNA_d_num, args.hid_feats, args.out_feats)
        # self.mlp = nn.Sequential(
        #         nn.Linear(901*2, 1024),
        #         nn.Dropout(0.1),
        # nn.Linear(1024, 512),
        # nn.Dropout(0.1),
        # nn.Linear(512, 64),
        # nn.Dropout(0.1),
        # nn.Linear(64, 1),nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(901 * 2, 1024),
                                 nn.Linear(1024, 512),
                                 nn.Linear(512, 64),
                                 nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, data, train_sample, test_sample, device):
      
        # m_drug_d_feature = torch.randn((self.args.m_drug_d_num, self.args.m_drug_d_num))
        # m_mRNA_d_feature = torch.randn((self.args.m_mRNA_d_num, self.args.m_mRNA_d_num))
        # m_incRNA_d_feature = torch.randn((self.args.m_incRNA_d_num, self.args.m_incRNA_d_num))
      
        # m_drug_d_ass = self.gat_m_drug_m(m_drug_d_feature, data['m_drug_d_adj'], device)  # 1x2060x901
        # m_incRNA_d_ass = self.gat_m_incRNA_d(m_incRNA_d_feature, data['m_incRNA_d_adj'], device)  # 1x2459x901
        # m_mRNA_d_ass = self.gat_m_mRNA_d(m_mRNA_d_feature, data['m_mRNA_d_adj'], device)  # 1x3929x901
     
        # m_drug_d_ass = torch.cat((m_drug_d_ass[0][:901], m_drug_d_ass[0][1183:]), dim=0).unsqueeze(0)  # 1x1778x901
        # m_incRNA_d_ass = torch.cat((m_incRNA_d_ass[0][:901], m_incRNA_d_ass[0][1582:]), dim=0).unsqueeze(
        #     0)  # 1x1778x901
        # m_mRNA_d_ass = torch.cat((m_mRNA_d_ass[0][:901], m_mRNA_d_ass[0][3052:]), dim=0).unsqueeze(0)  # 1x1778x901
        
        # result1 = torch.cat((m_drug_d_ass, m_incRNA_d_ass), 0)
        # sum_x = torch.cat((result1, m_mRNA_d_ass), 0)
        # sum_x = sum_x.unsqueeze(0)  # 1x3x1778x901
        # sum_x = sum_x.permute(2, 1, 0, 3)
        # result1 = torch.cat((m_drug_d_ass, m_incRNA_d_ass), 2)
        # result2 = torch.cat((result1, m_mRNA_d_ass), 2)
        # sum_x = torch.cat((result2, data['miRNA_disease_feature'].unsqueeze(0)), 2)                 #1x1778x3604

        # sum_x = torch.unsqueeze(sum_x, dim=0)                         #1,1,1778x3604
        # fusion_mode = MS_CAM().to(device)
        # sum_x = fusion_mode(sum_x)  # 1778 3 1 901
        # sum_x = sum_x.permute(2, 1, 0, 3)  # 1x3x1778x901
        # sum_x = torch.squeeze(sum_x, dim=0)  # 3x1778x901

       
        # sum_x = (sum_x[0] + sum_x[1] + sum_x[2]) / 3  # 1778x901
        # sum_x = torch.cat((sum_x, data['miRNA_disease_feature']), 1)  # 1778x1802

        train_sample = train_sample.int()
        train_emb = torch.empty(0).to(device)
        for i in range(len(train_sample)):
            a = torch.cat((data['miRNA_disease_feature'][train_sample[i][0]], data['miRNA_disease_feature'][train_sample[i][1]]), dim=0).unsqueeze(0)
            train_emb = torch.cat((train_emb, a), dim=0)
        train_score = self.mlp(train_emb)
        test_sample = test_sample.int()
        test_emb = torch.empty(0).to(device)
        for i in range(len(test_sample)):
            a = torch.cat((data['miRNA_disease_feature'][test_sample[i][0]], data['miRNA_disease_feature'][test_sample[i][1]]), dim=0).unsqueeze(0)
            test_emb = torch.cat((test_emb, a), dim=0)
        test_score = self.mlp(test_emb)

        return train_score, test_score
