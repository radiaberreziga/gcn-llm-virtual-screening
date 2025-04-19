class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(num_node_features , hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)

        self.lin1 = nn.Linear(hidden_channels , hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.bn5 = nn.BatchNorm1d(hidden_channels)
        self.bn6 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, batch_indice):
    

        # GCN layers
        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)

        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)

        x = self.conv3(x, edge_index).relu()
        x = self.bn3(x)

        x = self.conv4(x, edge_index).relu()
        x = self.bn4(x)

        x = self.conv5(x, edge_index).relu()
        x = self.bn5(x)

        x = self.conv6(x, edge_index).relu()
        x = self.bn6(x)
        # Readout layer
        x = global_mean_pool(x, batch_indice)


        # Classifier
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x