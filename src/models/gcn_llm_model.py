import networkx as nx
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import torch.nn as nn
import pandas as pd

class GCN_LLM(torch.nn.Module):
    def __init__(self,num_node_features,num_classes, hidden_channels,smile_llm_dim):
        super(GCN_LLM, self).__init__()
        torch.manual_seed(12345)
        self.smiles_proj = nn.Linear(smile_llm_dim, 10)

        self.conv1 = GCNConv(num_node_features + 10, hidden_channels)
        self.conv2 = GCNConv(hidden_channels +10 , hidden_channels)
        self.conv3 = GCNConv(hidden_channels +10, hidden_channels)
        self.conv4 = GCNConv(hidden_channels +10, hidden_channels)
        self.conv5 = GCNConv(hidden_channels +10, hidden_channels)
        self.conv6 = GCNConv(hidden_channels +10, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels , hidden_channels)  # Update input size
        self.lin2 = nn.Linear(hidden_channels, num_classes)


        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.bn6 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, batch_indice, smile_llm):
        # 1. Obtain node embeddings
        if not isinstance(smile_llm, torch.Tensor):
            smile_llm = torch.tensor(smile_llm, dtype=torch.float32)
        smile_llm = smile_llm.to(device) # Move graph_features to GPU before applying operations

        # Project SMILES vector to reduced dimensionality
        smile_llm = self.smiles_proj(smile_llm).relu()
        smile_llm = F.dropout(smile_llm, p=0.3, training=self.training).squeeze(1)


        # Repeat SMILES vector for each node in batch
        smile_llm_repeated = smile_llm[batch_indice]

        # Concatenate with node features
        x = torch.cat([x, smile_llm_repeated], dim=1)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.bn1(x)


        x = torch.cat([x, smile_llm_repeated], dim=1)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.bn2(x)

        x = torch.cat([x, smile_llm_repeated], dim=1)


        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.bn3(x)

        x = torch.cat([x, smile_llm_repeated], dim=1)


        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.bn4(x)

        x = torch.cat([x, smile_llm_repeated], dim=1)

        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.bn5(x)

        x = torch.cat([x, smile_llm_repeated], dim=1)


        x = self.conv6(x, edge_index)
        x = x.relu()
        x = self.bn6(x)
   # Concatenate graph features with node embeddings
       # graph_features = graph_features.unsqueeze(1).expand(-1, x.size(1), -1)  # Expand graph features to match node count
       # x = torch.cat((x, graph_features), dim=-1)
        x = global_mean_pool(x, batch_indice)

        # Classifier
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


 
