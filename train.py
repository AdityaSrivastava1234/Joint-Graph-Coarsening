import argparse
import torch
import torch.nn.functional as F
from networks import Net
from torch import tensor
from torch.optim import Adam
from utils import load_data, coarsening
import numpy as np
import os
import random

torch.backends.cudnn.deterministic = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--experiment', type=str, default=None) #'fixed', 'random', 'few'
parser.add_argument('--runs', type=int, default=21)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--epochs', type=int, default=201)
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--coarsening_ratio', type=float, default=0.5)
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
parser.add_argument('--jobid',type=str,default="abc_sr")

# -------------------------------------------------- Feature Learning Related Arguments SitaRam _/\_ ------------------------------------------------------ 
parser.add_argument('--features_mode',type=str,default=None) # 'RBF','CosSim','Smooth_Sigz'
parser.add_argument('--mix_threshold',type=float,default=0.5)
parser.add_argument('--mix_alpha',type=float,default=0.5)
parser.add_argument('--alpha_ss',type=float,default=0.6,help="alpha param used for Smooth Sigz")
parser.add_argument('--beta_ss',type=float,default=None,help="beta param used for Smooth Sigz")
parser.add_argument('--maxiter_ss',type=int,default=1001,help="Max iterations param used for Smooth Sigz")
parser.add_argument('--on_Wmix',type=int,default=1)

args = parser.parse_args()
print(vars(args))
print("")
path = "params/"
if not os.path.isdir(path):
    os.mkdir(path)

args.on_Wmix = args.on_Wmix==1
if args.beta_ss is None:
    args.beta_ss = 1-args.alpha_ss
if args.experiment is None:
    if args.dataset in ["cora","citeseer","pubmed"]:
        args.experiment = "fixed"
    elif args.dataset in ["dblp","Physics"]:
        args.experiment = "random"
    else:
        raise NotImplementedError("Argument --dataset undecipherable :/")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method, args)
model = Net(args).to(device)
all_acc = []

for _ in range(args.runs):

    data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
        args.dataset, candidate, C_list, Gc_list, args.experiment)

    data = data.to(device)
    coarsen_features = coarsen_features.to(device)
    coarsen_train_labels = coarsen_train_labels.to(device)
    coarsen_train_mask = coarsen_train_mask.to(device)
    coarsen_val_labels = coarsen_val_labels.to(device)
    coarsen_val_mask = coarsen_val_mask.to(device)
    coarsen_edge = coarsen_edge.to(device)

    if args.normalize_features:
        coarsen_features = F.normalize(coarsen_features, p=1)
        data.x = F.normalize(data.x, p=1)

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    val_loss_history = []

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()
        out = model(coarsen_features, coarsen_edge)
        loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(coarsen_features, coarsen_edge)
        val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(path,'checkpoint-best-acc-'+args.jobid+'.pkl'))

        val_loss_history.append(val_loss)
        if args.early_stopping > 0 and epoch > args.epochs // 2:
            tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                break

    model.load_state_dict(torch.load(os.path.join(path,'checkpoint-best-acc-'+args.jobid+'.pkl')))
    model.eval()
    pred = model(data.x, data.edge_index).max(1)[1]
    test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
    print(test_acc*100)
    all_acc.append(test_acc*100)

print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

