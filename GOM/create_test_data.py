'''
Author: your name
Date: 2021-03-19 21:17:29
LastEditTime: 2021-09-01 21:37:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \SEA\create_test_data.py
'''
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
    print(smile, mol)
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

df = pd.read_csv('./data/Test2.csv')
# print(df.head())
compound_iso_smiles += list(df['smiles'])
compound_iso_smiles = set(compound_iso_smiles)  # 去重

smile_graph = {}
for i, smile in enumerate(compound_iso_smiles):
    g = smile_to_graph(smile)
    smile_graph[smile] = g

processed_data_file = 'data/processed/test2.pt'
if ((not os.path.isfile(processed_data_file))):
    # 要读取的文件
    df = pd.read_csv('data/Test2.csv')
    drugs, Y = list(
        df['smiles']), list(df['label'])
    data = TestbedDataset(
        root='data', dataset='test2', xd=drugs, y=Y, smile_graph=smile_graph)
    print(processed_data_file, 'has been created')
else:
    print(processed_data_file, 'is already created')
