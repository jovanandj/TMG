import dgl
from config import DISTANCE_METRIC, pruning_threshold, DATASET_NAME, SCALER

def load_graphs():    
    train_graphs, _ = dgl.load_graphs(f'{DATASET_NAME}_train_val_graph_{DISTANCE_METRIC}_scaled_svmsmote_node_features_pruned_{pruning_threshold}_{SCALER}.bin')
    test_graphs, _ = dgl.load_graphs(f'{DATASET_NAME}_test_graph_{DISTANCE_METRIC}_scaled_node_features_pruned_{pruning_threshold}_{SCALER}.bin')

    train_graph = train_graphs[0]
    test_graph = test_graphs[0]

    return train_graph, test_graph
