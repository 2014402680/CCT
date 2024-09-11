from bondedgeconstruction import smiles_to_data
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.data import Data
root='data'#data_rlm
list1=pd.read_csv(root+'/data_nav_dev.csv')
listtest=pd.read_csv(root+'/eval_set_nav.csv')
import os

if not os.path.exists(root+'/train_nav.pth'):

    datasettrain = []
    for idx, row in tqdm(list1.iterrows()):
        data1 = smiles_to_data(row['SMILES'])


        label_value = row['Label']


        data1.y = torch.tensor(1 if label_value >= 5 else 0, dtype=torch.long)


        data_listtrain = data1
        datasettrain.append(data_listtrain)
    torch.save(datasettrain, root+'/train_nav.pth')
if not os.path.exists(root+'/test_nav60.pth'):
    datasettest=[]
    for idx, row in tqdm(listtest.iterrows()):
        data2=smiles_to_data(row['SMILES'])

        label_value = row['Label']


        data2.y = torch.tensor(1 if label_value >= 5 else 0, dtype=torch.long)

        #data2.y = torch.tensor(row['Label'], dtype=torch.long)
        data_listtest = data2
        datasettest.append(data_listtest)
    torch.save(datasettest, root+'/test_nav60.pth')


