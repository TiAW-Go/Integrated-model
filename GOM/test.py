'''
Author: your name
Date: 2021-01-29 15:07:25
LastEditTime: 2021-09-01 21:29:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \SEA\test.py
'''
from rdkit import Chem
from rdkit.Chem import Draw
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# df = pd.read_excel("data/Summary.xlsx", sheet_name=0, engine='openpyxl')
#
# data = df.loc[:, ["SYSTEM1", "SYSTEM2", "SYSTEM3",
#   "SYSTEM4", "SYSTEM5", "SYSTEM6", "SYSTEM7"]]
# x_data = df.loc[:, ["SMILES"]].values.reshape(-1,)
# y_data = df.loc[:, ["SYSTEM1"]].values.reshape(-1,)
# x_train, x_test, y_train, y_test = train_test_split(
    # x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

# for x, y in zip(x_train, y_train):
    # print(x, y)

mol = Chem.MolFromSmiles("CC1(C2CC(C34C(C2(C=CC1=O)C)CCC(C3O)C(=C)C4=O)O)C")
print(mol)

# hitmapData = data.corr()
# f, ax = plt.subplots(figsize=(12,12))
# sns.heatmap(hitmapData, vmax=1, square=True,annot=True,cmap="BuGn_r")
# plt.savefig('Correlation-Matrix.png')

# plt.figure(figsize = (12, 6))
# sns.regplot(x = 'SYSTEM2', y = 'SYSTEM4', data = data)
# plt.show()
# count = pd.cut(data['SYSTEM1'], 8, include_lowest=True, duplicates='raise')
# count = pd.value_counts(count)


# plt.figure(figsize = (12, 6))
# sns.distplot(count, kde=False, rug=True);
# plt.show()

# hitmap_dict = hitmapData['SYSTEM3'].to_dict()
# del hitmap_dict['SYSTEM1']
# print("List the numerical features decendingly by their correlation with Sale Price:\n")
# for ele in sorted(hitmap_dict.items(), key = lambda x: -abs(x[1])):
# print(ele)


# df = pd.read_csv('data/davis_test.csv')
# train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
# train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
# print(train_drugs)
a = '123123123'

# print(' '.join(a))
