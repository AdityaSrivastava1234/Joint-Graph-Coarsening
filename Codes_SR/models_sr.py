import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.nn import GCNConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
class APPNP_SR(torch.nn.Module):
    def __init__(self, configs):
        super(APPNP_SR, self).__init__()
        self.K = configs['K']
        self.beta = configs['beta']
        self.num_features = configs['num_features']
        self.num_classes = configs['num_classes']
        self.hidden = configs['hidden']
        self.lin1 = Linear(self.num_features, self.hidden)
        self.lin2 = Linear(self.hidden, self.num_classes)
        self.prop1 = APPNP(self.K, self.beta)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)

        return F.log_softmax(x, dim=1)

# %%
class GCN_SR(torch.nn.Module):
    def __init__(self, configs):
        super(GCN_SR, self).__init__()
        self.num_features = configs['num_features']
        self.num_classes = configs['num_classes']
        self.hidden = configs['hidden']
        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.conv2 = GCNConv(self.hidden, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)