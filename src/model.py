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



class MDA(nn.Module):
    def __init__(self, args):
        super(MDA, self).__init__()
        self.args = args
        self.mlp = nn.Sequential(nn.Linear(1802 * 2, 1024),
                                 nn.Linear(1024, 512),
                                 nn.Linear(512, 64),
                                 nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, data, train_sample, test_sample, device):
        train_sample = train_sample.int()
        train_emb = torch.empty(0).to(device)
        for i in range(len(train_sample)):
            a = torch.cat((data['inter_miRNA_disease_feature'][train_sample[i][0]], data['inter_miRNA_disease_feature'][train_sample[i][1]]), dim=0).unsqueeze(0)
            train_emb = torch.cat((train_emb, a), dim=0)
        train_score = self.mlp(train_emb)
        test_sample = test_sample.int()
        test_emb = torch.empty(0).to(device)
        for i in range(len(test_sample)):
            a = torch.cat((data['inter_miRNA_disease_feature'][test_sample[i][0]], data['inter_miRNA_disease_feature'][test_sample[i][1]]), dim=0).unsqueeze(0)
            test_emb = torch.cat((test_emb, a), dim=0)
        test_score = self.mlp(test_emb)

        return train_score, test_score
