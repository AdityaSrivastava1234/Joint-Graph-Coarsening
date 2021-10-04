import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull

import utilis_sr

# %%
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

# %%
def splits(data, num_classes, exp):
    if exp!='fixed_SR':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random_SR':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data

# %%
def load_data(configs):
    ds_nm_sr = configs['dataset'].split('_')[0]
    if configs['dataset'] == 'dblp_SR':
        dataset = CitationFull(root=configs['data_path'][:-1], name=ds_nm_sr)
    elif configs['dataset'] == 'Physics_SR':
        dataset = Coauthor(root=configs['data_path']+configs['dataset'], name=ds_nm_sr)
    else:
        dataset = Planetoid(root=configs['data_path'][:-1], name=ds_nm_sr)
    
    n_classes = len(set(np.array(dataset[0].y)))
    data = splits(data=dataset[0], num_classes=n_classes, exp=configs['experiment'])
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(configs['candidate']):
        H = configs['candidate'][number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = utilis_sr.one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = utilis_sr.one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])

            C = configs['C_list'][number]
            Gc = configs['Gc_list'][number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    print('Size of coarsen graph features:', coarsen_features.shape)
    print("")

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge