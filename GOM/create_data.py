import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    # 将不允许设置的输入映射到最后一个元素。
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


compound_iso_smiles = []

"""
修改文件
"""
df = pd.read_excel("data/Summary.xlsx", sheet_name=0, engine='openpyxl')
compound_data = df.loc[:, ["SMILES"]].values.reshape(-1,)
compound_iso_smiles = compound_data.tolist()
compound_iso_smiles = set(compound_iso_smiles)  # 去重 字典
# print(compound_iso_smiles)

smile_graph = {}
for i, smile in enumerate(compound_iso_smiles):
    # print(i,smile)
    g = smile_to_graph(smile)
    smile_graph[smile] = g

systems = ["SYSTEM1", "SYSTEM2", "SYSTEM3",
           "SYSTEM4", "SYSTEM5", "SYSTEM6", "SYSTEM7"]

for system in systems:
    processed_data_file_train = 'data/processed/' + system + '_train.pt'
    processed_data_file_test = 'data/processed/' + system + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        # 要读取的文件
        df = pd.read_excel("data/Summary.xlsx", sheet_name=0, engine='openpyxl')
        x_data = df.loc[:, ["SMILES"]].values.reshape(-1,)
        y_data = df.loc[:, [system]].values.reshape(-1,)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

        # make data PyTorch Geometric ready
        print('preparing ', system + '_train.pt in pytorch format!')
        train_data = TestbedDataset(
            root='data', dataset=system+'_train', xd=x_train, y=y_train, smile_graph=smile_graph)

        print('preparing ', system + '_test.pt in pytorch format!')
        test_data = TestbedDataset(
            root='data', dataset=system+'_test', xd=x_test, y=y_test, smile_graph=smile_graph)
        print(processed_data_file_train, ' and ',
              processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ',
              processed_data_file_test, ' are already created')
