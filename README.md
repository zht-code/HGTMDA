# MHGTMDA: molecular heterogeneous graph transformer based on biological entity graph for miRNA-disease associations prediction
MicroRNAs (miRNAs) play a crucial role in the prevention, prognosis, diagnosis, and treatment of complex diseases. However, the current research on miRNA and Disease Association (MDA) is limited. Existing computational methods primarily focus on biologically relevant molecules directly associated with miRNA or disease, overlooking the fact that the human body is a highly complex system where miRNA or disease may indirectly correlate with various types of biomolecules. To address this, we propose a novel prediction model named MHGTMDA (miRNA and disease association prediction using Heterogeneous Graph Transformer based on Molecular Heterogeneous Graph). MHGTMDA integrates biological entity relationships of eight biomolecules, constructing a relatively comprehensive heterogeneous biological entity graph. Serving as a powerful molecular heterogeneous graph transformer, MHGTMDA extracts graph structural elements of miRNA and disease, combining their attribute information to detect potential correlations. In a 5-fold cross-validation study, MHGTMDA achieved an Area Under the Receiver Operating Characteristic Curve (AUC) of 0.9569, surpassing state-of-the-art methods by at least 3\%. Feature ablation experiments suggest that considering features among multiple biomolecules is more effective in uncovering miRNA-disease correlations. Furthermore, we conducted differential expression analyses on breast cancer and lung cancer, using MHGTMDA to further validate differentially expressed miRNAs. The results demonstrate MHGTMDA's capability to identify novel MDAs.

![Image text](https://github.com/zht-code/HGTMDA/blob/main/IMG/MHGTMDA.svg)

Overall architecture of MHGTMDA. A. Data sources for MHGTMDA. B. The integrated miRNA sequence and similarity network and the integrated disease similarity network were constructed respectively to extract the inherent attribute features of both. C. The sub-module constructed Biological entity graphs, including MiRNA, Disease, Microbe, LncRNA, CircRNA, MRNA, Protein, and Drug. D. The sub-module mainly extracts the embedding feature of miRNA and disease in Biological entity graphs. E. The multimodal embedding representations of miRNAs and diseases were concatenated and fed into the MLP for training and prediction.
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
MHGTMDA is tested to work under:

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

1, Download the environment required by MHGTMDA
```
pip install pytorch == 1.10.2+cu113
```
2, Run embedding.py to generate miRNA and disease embedding features, Specifically, we constructed the graph utilizing the torch_geometric tool. First, We enter the collected biological entities into HeteroData as nodes (HeteroData is a PyG built-in data structure for representing heterogeneous graphs). Next, we constructed node mappings by different node types to construct edge indexes in HeteroData. Finally, we construct node type labels to represent the type of each node in HeteroData..the options are:
```
python ./src/embedding.py
```
3, The specific code is run by referring to the following train.py to generate train_model and performance score, the options are:
```
python ./src/train.py

```
4, Ablation experiment： To further demonstrate the effectiveness of MHGTMDA, we conducted two sets of ablation experiments, removing attribute features and structural features respectively, to compare the effects with MHGTMDA under 5-fold cross-validation experiment. The specific code is run by referring to the following ../attributes_feature/train.py，../network_feature/train.py to generate performance score for everyone, the options are:
```
python ./ablation/attributes_feature/train.py

python ./ablation/network_feature/train.py
```
5, We use a 5-fold cross-validation strategy to evaluate the generalization ability of our model (MHGTMDA). In the results, we plot the receiver operating characteristic curves(ROCs) and precision-recall curves (PRCs). Furthermore, the area under the ROCs (AUC) was also used to measure the ability of MHGTMDA. The specific code is run by referring to the following ./5CV/train.py to generate 5-CV scores, the options are:
```
python ./5CV/train.py
```
6, we conducted differential expression analyses on breast cancer and lung cancer, using MHGTMDA to further validate differentially expressed miRNAs. The specific code is run by referring to the following ./src/casestudies.py to generate two disease predictions, the options are:
```
python  ./src/casestudies.py
```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.



# Contributing

All authors were involved in the conceptualization of the proposed method. BYJ and SLP conceived and supervised the project. HTZ and BYJ designed the study and developed the approach. BYJ and HTZ implemented and applied the method to microbial data. BYJ and HTZ analyzed the results. BYJ and SLP contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.

# Cite



# Contacts
If you have any questions or comments, please feel free to email: zht@glut.edu.cn.
