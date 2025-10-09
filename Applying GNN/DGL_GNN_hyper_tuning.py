import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import gc
import torch.nn.functional as F
import dgl.function as fn
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_intermediate_values, plot_slice
import plotly.io as pio
from sklearn.metrics import f1_score

# Set the default renderer for plotly
pio.renderers.default='browser'

distance_metric = 'cosine'
pruning_threshold = 0.4

def load_graphs():
    # Load the graphs from binary files
    train_graphs, _ = dgl.load_graphs(f'train_graph_{distance_metric}_scaled_node_features_pruned_{pruning_threshold}_robust_scaler.bin')
    validation_graphs, _ = dgl.load_graphs(f'validation_graph_{distance_metric}_scaled_node_features_pruned_{pruning_threshold}_robust_scaler.bin')
    test_graphs, _ = dgl.load_graphs(f'test_graph_{distance_metric}_scaled_node_features_pruned_{pruning_threshold}_robust_scaler.bin')

    train_graph = train_graphs[0]
    validation_graph = validation_graphs[0]
    test_graph = test_graphs[0]

    return train_graph, validation_graph, test_graph

def create_batches(nodes, batch_size):
    # Shuffle the node indices
    nodes = nodes[torch.randperm(nodes.shape[0])]

    # Split the nodes into mini-batches
    batches = torch.split(nodes, batch_size)

    # Check the size of the last batch
    if batches[-1].size(0) < batch_size:
        batches = batches[:-1]  # Drop the last batch

    return batches

def compute_class_weights(labels):
    class_counts = labels.bincount()
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    return class_weights

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

train_graph, validation_graph, test_graph = load_graphs()
print(train_graph)
print(validation_graph)
print(test_graph)

train_nodes = torch.arange(train_graph.number_of_nodes())
validation_nodes = torch.arange(validation_graph.number_of_nodes())

# Define the dimensionality of the input features
in_dim = 20

num_epochs = 2000  # You can tune this hyperparameter
# ### Define the Objective Function for Optuna
def objective(trial):
    # Access suggested hyperparameters from the trial
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 0.1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 6)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 1)
    batch_size = trial.suggest_int('batch_size', 32, 1024)
    patience = trial.suggest_int('patience', 20, 200)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 0.1, log=True)  # Add L1 regularization coefficient as a hyperparameter
    aggregator_type = trial.suggest_categorical('aggregator_type', ['mean', 'gcn', 'pool', 'lstm'])

    # layer_type = trial.suggest_categorical('layer_type', ['GraphConv', 'GATConv', 'SAGEConv'])
    layer_type = 'SAGEConv'  # Set a fixed layer type

    train_batches = create_batches(train_nodes, batch_size=batch_size)
    validation_batches = create_batches(validation_nodes, batch_size=batch_size)

    # Create and initialize the GNN model
    model = GNNModel(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=3, num_layers=num_layers, 
                     dropout_rate=dropout_rate, layer_type=layer_type, aggregator_type=aggregator_type)
    
    # Compute class weights
    train_labels = train_graph.ndata['label']
    # class_weights = compute_class_weights(train_labels)
    class_weights = torch.tensor([0.55, 0.35, 0.1]) # real class weights tensor([0.6373, 0.3002, 0.0625])
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(train_graph.device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the best validation score and the patience counter
    best_val_score = 0
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_batches:
            # Get the subgraph for the current batch of nodes
            subgraph = train_graph.subgraph(batch)

            # The features of the input nodes
            batch_inputs = subgraph.ndata['feat']
            # The labels of the output nodes
            batch_labels = subgraph.ndata['label']

            # Forward pass
            batch_pred = model(subgraph, batch_inputs)
            loss = criterion(batch_pred, batch_labels)

            # Add L1 regularization
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.norm(param, 1)
            loss += l1_lambda * l1_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clear memory
            del batch_pred, loss
            gc.collect()


        # Validation loop
        with torch.no_grad():
            total_score = 0
            for batch in validation_batches:
                # Get the subgraph for the current batch of nodes
                subgraph = validation_graph.subgraph(batch)

                # The features of the input nodes
                batch_inputs = subgraph.ndata['feat']
                # The labels of the output nodes
                batch_labels = subgraph.ndata['label']

                # Forward pass
                batch_pred = model(subgraph, batch_inputs)
                batch_pred_class = torch.argmax(batch_pred, dim=1)  # Get the predicted classes

                # Calculate the metrics
                f1_macro = f1_score(batch_labels.cpu(), batch_pred_class.cpu(), average='macro')
                f1_weighted = f1_score(batch_labels.cpu(), batch_pred_class.cpu(), average='weighted')

                # Calculate accuracy
                accuracy = (batch_pred_class == batch_labels).float().mean()

                # Combine the metrics into a single score
                # You can adjust the weights depending on how important each metric is
                score = 0.5 * f1_macro + 0.3 * f1_weighted + 0.2 * accuracy

                total_score += score

                # Clear memory
                del batch_pred, batch_pred_class, f1_macro, f1_weighted, score, accuracy
                gc.collect()

            # Calculate the average score
            avg_score = total_score / len(validation_batches)

            # Check if the validation score has improved
            if avg_score > best_val_score:
                best_val_score = avg_score
                patience_counter = 0
            else:
                patience_counter += 1

            # Check if the trial should be pruned
            if patience_counter >= patience:
                break
                # raise optuna.exceptions.TrialPruned()
        
            # Report the average F1 score to the trial
        trial.report(avg_score, epoch)

    # Clear memory
    gc.collect()

    # Return the average macro-averaged F1 score
    return best_val_score


# Create an Optuna study
pruner = optuna.pruners.HyperbandPruner(max_resource='auto', reduction_factor=3)

# Create a storage
storage = optuna.storages.RDBStorage("mysql+mysqlconnector://root:mojsql@localhost/hyper_tuning_db")

# Check if the study exists before deleting
if "distributed-tuning" in [study.study_name for study in storage.get_all_studies()]:
    # Try to delete the study, and handle the exception if it does not exist
    try:
        optuna.delete_study(study_name="distributed-tuning", storage=storage)
    except KeyError:
        pass

# Create a new study that uses the storage
# study = optuna.create_study(storage=storage, study_name="distributed-tuning", direction="maximize", 
#                             load_if_exists=False, sampler=optuna.samplers.TPESampler(), pruner=pruner)

study = optuna.create_study(storage=storage, study_name="distributed-tuning", direction="maximize", 
                            load_if_exists=False, sampler=optuna.samplers.TPESampler())

# Optimize the study
study.optimize(objective, n_jobs=3, n_trials=21)

# Print the best hyperparameters and their corresponding objective value
print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Plot the optimization history
plot_optimization_history(study).show()

# Plot the parameter importance
plot_param_importances(study).show()

# Plot the intermediate values
plot_intermediate_values(study).show()

# Plot the slice
plot_slice(study).show()