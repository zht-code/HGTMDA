import dgl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv
import numpy as np
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data



miRNA_node = pd.read_csv('dataset/quchong_bianhao/miRNA_node.csv', header=None)
disease_node = pd.read_csv('dataset/quchong_bianhao/disease_node.csv', header=None)
drug_node = pd.read_csv('dataset/quchong_bianhao/drug_node.csv', header=None)
mRNA_node = pd.read_csv('dataset/quchong_bianhao/mRNA_node.csv', header=None)
protein_node = pd.read_csv('dataset/quchong_bianhao/protein_node.csv', header=None)
lncRNA_node = pd.read_csv('dataset/quchong_bianhao/lncRNA_node.csv', header=None)
microbe_node = pd.read_csv('dataset/quchong_bianhao/microbe_node.csv', header=None)
circRNA_node = pd.read_csv('dataset/quchong_bianhao/circRNA_node.csv', header=None)


edge_list_circRNA_disease = pd.read_csv('dataset/all_association/CircDiseaseMergeAssociation.csv', header=None)
edge_list_circRNA_miRNA = pd.read_csv('dataset/all_association/CircMiSomamiRAssociation.csv', header=None)
edge_list_disease_mRNA = pd.read_csv('dataset/all_association/DiseaseMDisGeNETAssociation.csv', header=None)
edge_list_disease_microbe = pd.read_csv('dataset/all_association/DiseaseMicrobeHMDADAssociation.csv', header=None)
edge_list_drug_disease = pd.read_csv('dataset/all_association/DrugDiseaseSCMFDDAssociation.csv', header=None)
edge_list_drug_microbe = pd.read_csv('dataset/all_association/DrugMicrobeAssociation.csv', header=None)
edge_list_drug_mRNA = pd.read_csv('dataset/all_association/DrugMPharmGKBAssociation.csv', header=None)
edge_list_drug_protein = pd.read_csv('dataset/all_association/DrugProteinDrugBankAssociationThreshold5.csv', header=None)
edge_list_lncRNA_disease = pd.read_csv('dataset/all_association/LncDiseaseMergeAssociation.csv', header=None)
edge_list_lncRNA_miRNA = pd.read_csv('dataset/all_association/LncMiSNPAssociation.csv', header=None)
edge_list_lncRNA_mRNA = pd.read_csv('dataset/all_association/LncMLncRNA2TargetAssociation.csv', header=None)
edge_list_lncRNA_protein = pd.read_csv('dataset/all_association/LncProteinNPInterAssociation.csv', header=None)
edge_list_miRNA_drug = pd.read_csv('dataset/all_association/MiDrugSM2Association.csv', header=None)
edge_list_miRNA_mRNA = pd.read_csv('dataset/all_association/MiMNMiTarbaseAssociation.csv', header=None)
edge_list_miRNA_protein = pd.read_csv('dataset/all_association/MiProteinMergeAssociation.csv', header=None)
edge_list_mRNA_protein = pd.read_csv('dataset/all_association/MProteinNCBIAssociation.csv', header=None)


id_mapping = {}


for node_type in [miRNA_node, disease_node, drug_node, mRNA_node, protein_node, lncRNA_node, microbe_node, circRNA_node]:
    for node_id in range(len(node_type)):
        if node_id not in id_mapping:
            id_mapping[node_type[0][node_id]] = node_type[1][node_id]


for edge_list in [edge_list_circRNA_disease, edge_list_circRNA_miRNA, edge_list_disease_mRNA, edge_list_disease_microbe, edge_list_drug_disease, edge_list_drug_microbe, edge_list_drug_mRNA, edge_list_drug_protein, edge_list_lncRNA_disease, edge_list_lncRNA_miRNA, edge_list_lncRNA_mRNA, edge_list_lncRNA_protein, edge_list_miRNA_drug, edge_list_miRNA_mRNA, edge_list_miRNA_protein, edge_list_mRNA_protein]:
    edge_list[0] = edge_list[0].map(id_mapping)
    edge_list[1] = edge_list[1].map(id_mapping)
   


x_dict = {"miRNA": torch.randn(len(miRNA_node), 901), "disease": torch.randn(len(disease_node), 901), 
          "drug": torch.randn(len(drug_node), 901), "mRNA": torch.randn(len(mRNA_node), 901), "protein": torch.randn(len(protein_node), 901), 
          "lncRNA": torch.randn(len(lncRNA_node), 901), "microbe": torch.randn(len(microbe_node), 901), "circRNA": torch.randn(len(circRNA_node), 901)}


edge_index_dict = {
   ('circRNA', 'to', 'disease'): torch.tensor([edge_list_circRNA_disease[0].apply(lambda x: int(x[1:])).tolist(), edge_list_circRNA_disease[1].apply(lambda x: int(x[1:])).tolist()]),
   ('disease', 'to', 'circRNA'): torch.tensor([edge_list_circRNA_disease[1].apply(lambda x: int(x[1:])).tolist(), edge_list_circRNA_disease[0].apply(lambda x: int(x[1:])).tolist()]),
    ('circRNA', 'to', 'miRNA'): torch.tensor([edge_list_circRNA_miRNA[0].apply(lambda x: int(x[1:])).tolist(), edge_list_circRNA_miRNA[1].apply(lambda x: int(x[1:])).tolist()]),
 
    ('disease', 'to', 'mRNA'): torch.tensor([edge_list_disease_mRNA[0].apply(lambda x: int(x[1:])).tolist(), edge_list_disease_mRNA[1].apply(lambda x: int(x[1:])).tolist()]),
    ('disease', 'to', 'microbe'): torch.tensor([edge_list_disease_microbe[0].apply(lambda x: int(x[1:])).tolist(), edge_list_disease_microbe[1].apply(lambda x: int(x[1:])).tolist()]),
    ('drug', 'to', 'disease'): torch.tensor([edge_list_drug_disease[0].apply(lambda x: int(x[1:])).tolist(), edge_list_drug_disease[1].apply(lambda x: int(x[1:])).tolist()]),
    ('drug', 'to', 'microbe'): torch.tensor([edge_list_drug_microbe[0].apply(lambda x: int(x[1:])).tolist(), edge_list_drug_microbe[1].apply(lambda x: int(x[1:])).tolist()]),
    ('drug', 'to', 'mRNA'): torch.tensor([edge_list_drug_mRNA[0].apply(lambda x: int(x[1:])).tolist(), edge_list_drug_mRNA[1].apply(lambda x: int(x[1:])).tolist()]),
    ('drug', 'to', 'protein'): torch.tensor([edge_list_drug_protein[0].apply(lambda x: int(x[1:])).tolist(), edge_list_drug_protein[1].apply(lambda x: int(x[1:])).tolist()]),
    ('lncRNA', 'to', 'disease'): torch.tensor([edge_list_lncRNA_disease[0].apply(lambda x: int(x[1:])).tolist(), edge_list_lncRNA_disease[1].apply(lambda x: int(x[1:])).tolist()]),
    ('lncRNA', 'to', 'miRNA'): torch.tensor([edge_list_lncRNA_miRNA[0].apply(lambda x: int(x[1:])).tolist(), edge_list_lncRNA_miRNA[1].apply(lambda x: int(x[1:])).tolist()]),
    ('miRNA', 'to', 'lncRNA'): torch.tensor([edge_list_lncRNA_miRNA[1].apply(lambda x: int(x[1:])).tolist(), edge_list_lncRNA_miRNA[0].apply(lambda x: int(x[1:])).tolist()]),
    ('lncRNA', 'to', 'mRNA'): torch.tensor([edge_list_lncRNA_mRNA[0].apply(lambda x: int(x[1:])).tolist(), edge_list_lncRNA_mRNA[1].apply(lambda x: int(x[1:])).tolist()]),
    ('lncRNA', 'to', 'protein'): torch.tensor([edge_list_lncRNA_protein[0].apply(lambda x: int(x[1:])).tolist(), edge_list_lncRNA_protein[1].apply(lambda x: int(x[1:])).tolist()]),
    ('miRNA', 'to', 'drug'): torch.tensor([edge_list_miRNA_drug[0].apply(lambda x: int(x[1:])).tolist(), edge_list_miRNA_drug[1].apply(lambda x: int(x[1:])).tolist()]),
    ('miRNA', 'to', 'mRNA'): torch.tensor([edge_list_miRNA_mRNA[0].apply(lambda x: int(x[1:])).tolist(), edge_list_miRNA_mRNA[1].apply(lambda x: int(x[1:])).tolist()]),
    ('miRNA', 'to', 'protein'): torch.tensor([edge_list_miRNA_protein[0].apply(lambda x: int(x[1:])).tolist(), edge_list_miRNA_protein[1].apply(lambda x: int(x[1:])).tolist()]),
    ('mRNA', 'to', 'protein'): torch.tensor([edge_list_mRNA_protein[0].apply(lambda x: int(x[1:])).tolist(), edge_list_mRNA_protein[1].apply(lambda x: int(x[1:])).tolist()])
}
node_types = {"miRNA", "disease", "drug", "mRNA", "protein", "lncRNA", "microbe", "circRNA"}
metadata = (["miRNA", "disease", "drug", "mRNA", "protein", "lncRNA", "microbe", "circRNA"],
            [('circRNA', 'to', 'disease'),
             ('disease', 'to', 'circRNA'),
            ('circRNA', 'to', 'miRNA'),
            ('disease', 'to', 'mRNA'),
            ('disease', 'to', 'microbe'),
            ('drug', 'to', 'disease'),
            ('drug', 'to', 'microbe'),
            ('drug', 'to', 'mRNA'),
            ('drug', 'to', 'protein'),
            ('lncRNA', 'to', 'disease'),
            ('lncRNA', 'to', 'miRNA'),
            ('miRNA', 'to', 'lncRNA'),
            ('lncRNA', 'to', 'mRNA'),
            ('lncRNA', 'to', 'protein'),
            ('miRNA', 'to', 'drug'),
            ('miRNA', 'to', 'mRNA'),
            ('miRNA', 'to', 'protein'),
            ('mRNA', 'to', 'protein')])



miRNA_labels = miRNA_node[2]
disease_labels = disease_node[2]
drug_labels = drug_node[2]
mRNA_labels = mRNA_node[2]
protein_labels = protein_node[2]
lncRNA_labels = lncRNA_node[2]
microbe_labels = microbe_node[2]
circRNA_labels = circRNA_node[2]

node_type_labels = {"miRNA":torch.tensor(miRNA_labels), "disease":torch.tensor(disease_labels),
                     "drug":torch.tensor(drug_labels), "mRNA":torch.tensor(mRNA_labels),
                     "lncRNA":torch.tensor(lncRNA_labels), "circRNA":torch.tensor(circRNA_labels),
                       "protein":torch.tensor(protein_labels), "microbe":torch.tensor(microbe_labels)}

# data = HeteroData()
data = Data(x_dict=x_dict, edge_index_dict=edge_index_dict, node_types = node_types, metadata = metadata, labels = node_type_labels)
pass

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # for ntype in data.node_types:
        #     x_dict[ntype] = self.lin(x_dict[ntype])
        return torch.sigmoid(self.lin(x_dict['miRNA'])), torch.sigmoid(self.lin(x_dict['disease']))


model = HGT(hidden_channels=1024, out_channels=901, num_heads=2, num_layers=2)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)

for epoch in range(50):
    model.train()  
    optimizer.zero_grad()
   
    logits_miRNA_dict, logits_disease_dict = model(data.x_dict, data.edge_index_dict)

    miRNA_labels = data.labels['miRNA']
    miRNA_loss = criterion(logits_miRNA_dict, miRNA_labels)
    miRNA_loss.backward(retain_graph = True)
    disease_labels = data.labels['disease']
    disease_loss = criterion(logits_disease_dict, disease_labels)
    disease_loss.backward()
    optimizer.step()
    print(f"Epoch {epoch},  miRNA Loss: {miRNA_loss.item()}, disease Loss: {disease_loss.item()}")

model.eval() 
with torch.no_grad():
    miRNA_embedding, disease_embedding = model(data.x_dict, data.edge_index_dict)
print(f"miRNA embedding: {miRNA_embedding}, disease embedding: {disease_embedding}")
np.savetxt('./dataset/data/miRNA_embedding.txt', miRNA_embedding.tolist(), fmt="%f", comments='')
np.savetxt('./dataset/data/disease_embedding.txt', disease_embedding.tolist(), fmt="%f", comments='')


