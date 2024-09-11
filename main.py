import sys
from model import *
# from utils import *
from evalution import *
import torch.nn.functional as F
from nt_xent import NT_Xent
from model import set_seed
from sklearn.model_selection import KFold, GridSearchCV

import torch

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    lossz = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        batch1 = data.batch1.detach()
        edge = data.edge_index1.detach()
        xd = data.x1.detach()
        n = data.y.shape[0]  # batch
        optimizer.zero_grad()
        output, x_g, x_g1, output1 = model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
        loss_1 = criterion(output, data.y.view(-1, 1).float())
        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
        loss_2 = criterion(output1, data.y.view(-1, 1).float())

        cl_loss = criterion1(x_g, x_g1)
        loss = loss_1 + 0.3 * cl_loss + loss_2
        # loss = loss_1
        loss.backward()
        optimizer.step()
        lossz = loss + lossz




    print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, lossz))

DATASET_TYPE = "single"
TARGET = 1
TRAINING_SET_SPLIT = "FULL" #None, FULL, 0, 1, 2
MODEL_NUM = 0
TARGET_CPDS = "P14416_P42336"
DATASET_NAME = "chembl29_predicting_target_" + TARGET_CPDS + "_target_"+ str(TARGET) +"_vs_random_cpds"
CSV_DATA_PATH = "../data/"+ DATASET_NAME + ".csv"
GNNEXPLAINER_USAGE = True
SEED = 42
SAVE = True
M = 100
M = 100

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch1 = data.batch1.detach()
            edge = data.edge_index1.detach()
            xd = data.x1.detach()
            output, x_g, x_g1, output1 = model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)


    return total_labels, total_preds





    return total_labels, total_preds


if __name__ == "__main__":
    set_seed(42)
    cuda_name = "cuda:0"
    print('cuda_name:', cuda_name)

    NUM_EPOCHS = 500
    LR = 0.0005

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    root = 'C:/Users/Administrator/Desktop/CCT-master (1)/CCT-master/data'
    processed_train = root + '/train_herg.pth'
    processed_val =root + '/val_herg.pth'
    data_listtrain = torch.load(processed_train)
    processed_val = torch.load(processed_val)


    def custom_batching(data_list, batch_size):
        for i in range(0, len(data_list), batch_size):
            yield data_list[i:i + batch_size]


    best_results = []

    batchestrain = list(custom_batching(data_listtrain, 1024))
    batchestrain1 = list()
    for batch_idx, data in enumerate(batchestrain):
        data = collate_with_circle_index(data)
        data.edge_attr = None

        batchestrain1.append(data)
    processed_val = list(custom_batching(processed_val, 30))
    processed_val1 = list()
    for batch_idx, data in enumerate(processed_val):
        data = collate_with_circle_index(data)
        data.edge_attr = None
        processed_val1.append(data)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = CCT().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    max_auc = 0

    model_file_name = 'model' + '.pt'
    result_file_name = 'result' + '.csv'
    for epoch in range(NUM_EPOCHS):
        train(model, device, batchestrain1, optimizer, epoch + 1)
        G, P = predicting(model, device, processed_val1)

        auc, acc, precision, f1_scroe, recall, specificity, ccr, mcc = metric(G, P)
        ret = [auc, acc, precision, f1_scroe, recall, specificity, ccr, mcc]
        if acc > max_auc:
            max_auc = acc
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            ret1 = [auc, acc, precision, f1_scroe, recall, specificity, ccr, mcc]



        print(
            'test---------------------------  auc:%.4f\t acc:%.4f\t precision:%.4f\t  f1_scroe:%.4f\t recall:%.4f\t  specificity:%.4f\t  ccr:%.4f\t  mcc:%.4f' % (
                auc, acc, precision, f1_scroe, recall, specificity, ccr, mcc))



    print('Maximum acc found. Model saved.')
    best_results.append({
        'auc, acc, precision,f1_scroe,  recall, specificity,ccr,mcc': (
            ret1)

    })


for i, result in enumerate(best_results, 1):
    print(f"Fold {i}:")
    print(f"Best auc, acc, precision,f1_scroe, recall, specificity,ccr, mcc: {result}")


