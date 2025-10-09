import dgl.nn as dglnn
import torch.nn as nn

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate, layer_type='GraphConv', aggregator_type='mean'):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()  # Add ReLU activation
        self.layer_type = layer_type  # Store the layer type

        if layer_type == 'GraphConv':
            self.layer = dglnn.GraphConv(in_dim, out_dim)
        elif layer_type == 'GATConv':
            self.layer = dglnn.GATConv(in_dim, out_dim, num_heads=1)
        elif layer_type == 'SAGEConv':
            self.layer = dglnn.SAGEConv(in_dim, out_dim, aggregator_type)
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(self, g, feature):
        h = self.dropout(feature)
        # print("h shape before layer:", h.shape)
        h = self.layer(g, h)
        if self.layer_type == 'GATConv':
            h = h.squeeze(1)  # Remove the head dimension for GATConv
        # print("h shape after layer:", h.shape)
        return self.activation(h)

class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout_rate, layer_type='GraphConv', aggregator_type='mean'):
        super(GNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(in_dim, hidden_dim, dropout_rate, layer_type, aggregator_type))
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, dropout_rate, layer_type, aggregator_type))
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, feature):
        h = feature
        for layer in self.layers:
            h = layer(g, h)
            # print("h shape after layer:", h.shape)
        return self.linear(h)