# HGTMDA: Heterogeneous graph transformer via biological entity graph for miRNA-disease associations prediction
 To accurately identify MDAs, we proposed a heterogeneous graph transformer via biological entity graph for the miRNA-disease association's prediction model (called HGTMDA). HGTMDA collects and collates 8 types of biological entity relationships from 8 types of small biological molecules to construct one of the most complete heterogeneous biological entity graphs. For complex heterogeneous graphs, HGTMDA introduces a powerful heterogeneous graph transformer to extract graph structure features of miRNAs and diseases and combine them with attribute features of both to identify potential associations.
![Image text](https://github.com/zht-code/HGTMDA/blob/main/HGTMDA.pdf)

Overall architecture of HGTMDA. A. Data sources for HGTMDA. B. The integrated miRNA sequence and similarity network and the integrated disease similarity network were constructed respectively to extract the inherent attribute features of both. C. The sub-module constructed Biological entity graphs, including MiRNA, Disease, Microbe, LncRNA, CircRNA, MRNA, Protein, and Drug. D. The sub-module mainly extracts the embedding feature of miRNA and disease in Biological entity graphs. E. The multimodal embedding representations of miRNAs and diseases were concatenated and fed into the MLP for training and prediction.
## Table of Contents
- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)

# Data description

| File name  | Description |
| ------------- | ------------- |
| miRNA.csv    | microRNA name file  |
| disease.csv  | disease name file   |
| all_association  | all RNA association file   |
| quchong_bianhao  | all RNA name file   |
| all_sample.csv  | all miRNA-disease sample  |
| AllMiKmer.csv  | all miRNA sequences feature  |
| AllMiKmer.csv  | all miRNA sequences feature  |
| miRNA_disease_feature.csv | feature of miRNAs and diseases fused with GIP |


# Installation
HGTMDA is tested to work under:

Python == 3.7

pytorch == 1.10.2+cu113

scipy == 1.5.4

numpy == 1.19.5

sklearn == 0.24.2

pandas == 1.1.5

matplotlib == 3.3.4

networkx == 2.5.1

# Quick start
To reproduce our results:

1, Download the environment required by HGTMDA
```
pip install pytorch == 1.10.2+cu113

```
2, Run embedding.py to generate miRNA and disease embedding feature, the options are:
```
python ./embedding.py

```
3, Run train.py to generate train_model and performance score, the options are:
```
python ./train.py

```
4, Ablation experiment：Run attributes_feature/train.py，network_feature/train.py to generate performance score for everyone, the options are:
```
python ./ablation/attributes_feature/train.py

python ./ablation/network_feature/train.py

```
5, Run 5CV/train.py to generate 5-CV scores, the options are:
```
python ./5_Fold.py

```
6, case_study: Run casestudies.py to generate three diseases prediction, the options are:
```
python  ./casestudies.py

```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.



# Contributing

Zouhaitao Jiboya..

# Cite



# Contacts
If you have any questions or comments, please feel free to email: zht@glut.edu.cn.
