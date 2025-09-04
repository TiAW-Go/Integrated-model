import gzip
import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch.utils import data

from gnn.graph_features import atom_features
from collections import defaultdict


class MolGraphDataset(data.Dataset):#①由文件获得smiles list  逐个获取所有分子的邻接矩阵、原子特征、边特征
    r"""For datasets consisting of SMILES strings and target values.

    Expects a csv file formatted as:
    comment,smiles,targetName1,targetName2
    Some Comment,CN=C=O,0,1
    ,CC(=O)NCCC1=CNc2c1cc(OC)cc2,1,1

    Args:
        path
        prediction: set to True if dataset contains no target values
    """

    def __init__(self, path, prediction=False):
        print(path)
        with gzip.open(path, 'r') as file:#处理表头
            self.header_cols = file.readline().decode('utf-8')[:-2].split(' ')
        n_cols = len(self.header_cols)
        print(n_cols)

        self.target_names = self.header_cols[1:]#获得每一个target的名字
        #self.comments = np.genfromtxt(path, delimiter=' ', skip_header=1, usecols=[0], dtype=np.str, comments=None)
        # comments=None because default is "#", that some smiles contain
        #self.smiles = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=[1], dtype=np.str, comments=None)
        self.smiles = np.genfromtxt(path, delimiter=' ', skip_header=1, usecols=[0], dtype=np.str, comments=None)
        #print('smiles:',self.smiles)

        if prediction:
            print(n_cols)
            #self.targets = np.empty((len(self.smiles), n_cols))  #创建一个数组关于n_cols个性质
            self.targets = np.empty((len(self.smiles), n_cols-1))####原本使用的预测语句


            # may be used to figure out number of targets etc
        else:
            #self.targets = np.genfromtxt(path, delimiter=' ', skip_header=1, usecols=range(1, n_cols), comments=None).reshape(-1, n_cols - 1)
            self.targets = np.genfromtxt(path, delimiter=' ', skip_header=1, usecols=[1],
                                         comments=None).reshape(-1, n_cols-1)



    def __getitem__(self, index):
        adjacency, nodes, edges = smile_to_graph(self.smiles[index])#邻接矩阵 原子特征集合 边特征集
        targets = self.targets[index, :]
        return (adjacency, nodes, edges), targets

    def __len__(self):
        return len(self.smiles)

rdLogger = rdkit.RDLogger.logger()
rdLogger.setLevel(rdkit.RDLogger.ERROR)

def smile_to_graph(smile):#获取分子的原子特征的函数
    molecule = Chem.MolFromSmiles(smile)
    #print("yes")
    n_atoms = molecule.GetNumAtoms()#节点数
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]#获得一个分子的所有原子

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)#获得这个分子的邻接矩阵
    node_features = np.array([atom_features(atom) for atom in atoms])#获得每个节点的特征
    #print(n_atoms)
#获取边的特征
    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])#起始节点，终止节点，边特征
    for bond in molecule.GetBonds():#获取每条边的特征
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return adjacency, node_features, edge_features#返回邻接矩阵，每个原子的特征，每条边的特征

# rdkit GetBondType() result -> int
BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


class MolGraphDatasetSubset(MolGraphDataset):
    r"""Takes a subset of MolGraphDataset.

    The "Subset" class of pytorch does not allow column selection
    """

    def __init__(self, path, indices=None, columns=None):
        super(MolGraphDatasetSubset, self).__init__(path)
        if indices:
            self.smiles = self.smiles[indices]
            self.targets = self.targets[indices]
        if columns:
            self.target_names = [self.target_names[col] for col in columns]
            self.targets = self.targets[:, columns]


# data is list of ((g,h,e), [targets])
# to be passable to DataLoader it needs to have this signature,
# where the outer tuple is that which is returned by Dataset's __getitem__
def molgraph_collate_fn(data):
    n_samples = len(data)
    (adjacency_0, node_features_0, edge_features_0), targets_0 = data[0]
    n_nodes_largest_graph = max(map(lambda sample: sample[0][0].shape[0], data))
    n_node_features = node_features_0.shape[1]
    n_edge_features = edge_features_0.shape[2]
    n_targets = len(targets_0)

    adjacency_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph)
    node_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_node_features)
    edge_tensor = torch.zeros(n_samples, n_nodes_largest_graph, n_nodes_largest_graph, n_edge_features)
    target_tensor = torch.zeros(n_samples, n_targets)

    for i in range(n_samples):
        (adjacency, node_features, edge_features), target = data[i]
        n_nodes = adjacency.shape[0]

        adjacency_tensor[i, :n_nodes, :n_nodes] = torch.Tensor(adjacency)
        node_tensor[i, :n_nodes, :] = torch.Tensor(node_features)
        edge_tensor[i, :n_nodes, :n_nodes, :] = torch.Tensor(edge_features)

        target_tensor[i] = torch.Tensor(target)

    return adjacency_tensor, node_tensor, edge_tensor, target_tensor
