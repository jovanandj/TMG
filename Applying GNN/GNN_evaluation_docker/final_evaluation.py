import logging
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from data_loader import load_graphs
from model import GNNModel
from train import train_model, test_model
from utils import create_batches, compute_class_weights
import optuna
from optuna import load_study
import csv
from tqdm import tqdm
from config import IN_DIM, LOAD_FROM_OPTUNA, BEST_PARAMS, BEST_EPOCH, DATASET_NAME

# Configure logging at the top of the module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('{}_app.log'.format(DATASET_NAME))
    ]
)

def main():
    logging.info('Starting script...')
    logging.info('Loading graphs')
    train_graph, test_graph = load_graphs()  # Assuming load_graphs now also returns the test graph
    logging.info(f'Train graph: {train_graph}')
    logging.info(f'Test graph: {test_graph}')

    train_nodes = torch.arange(train_graph.number_of_nodes())
    test_nodes = torch.arange(test_graph.number_of_nodes())

    if LOAD_FROM_OPTUNA:
        # Load the best hyperparameters from the Optuna study
        storage = optuna.storages.RDBStorage("mysql+mysqlconnector://root:mojsql@localhost/hyper_tuning_db")
        study = load_study(study_name="distributed-tuning", storage=storage)
        best_params = study.best_params
        best_epoch = study.best_trial.user_attrs['best_epoch']
    else:
        # Load the best hyperparameters from the config file
        best_params = BEST_PARAMS
        best_epoch = BEST_EPOCH

    logging.info('Creating and initializing the GNN model with the best hyperparameters')

    # Create and initialize the GNN model with the best hyperparameters
    model = GNNModel(in_dim=IN_DIM, hidden_dim=best_params['hidden_dim'], out_dim=3, num_layers=best_params['num_layers'], 
                     dropout_rate=best_params['dropout_rate'], layer_type='SAGEConv', aggregator_type=best_params['aggregator_type'])

    # Compute class weights
    train_labels = train_graph.ndata['label']
    class_weights = compute_class_weights(train_labels)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(train_graph.device))
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    logging.info('Creating batches for final training')
    # Create batches for final training
    final_train_batches = create_batches(train_nodes, batch_size=best_params['batch_size'])

    # Training loop
    logging.info(f'Starting training loop with {best_epoch} epochs')

    for epoch in tqdm(range(best_epoch)):
        train_model(model, criterion, optimizer, train_graph, final_train_batches, best_params['l1_lambda'])

    # Test the model
    logging.info('Testing the model')
    test_batches = create_batches(test_nodes, batch_size=best_params['batch_size'])
    accuracy, macro_accuracy, f1_macro, precision_macro, recall_macro, f1_weighted, precision_weighted, recall_weighted = test_model(model, test_graph, test_batches)

    # Create a filename with the hyperparameters and other relevant information
    filename = f"{DATASET_NAME}_results_hidden_dim_{best_params['hidden_dim']}_num_layers_{best_params['num_layers']}_dropout_rate_{best_params['dropout_rate']}_aggregator_type_{best_params['aggregator_type']}_learning_rate_{best_params['learning_rate']}_epoch_{best_epoch}.csv"

    logging.info('Writing the results to a CSV file')

    # Write the results to a CSV file
    with open(filename, 'w', newline='') as csvfile:        
        fieldnames = ['Test accuracy', 'Macro accuracy', 'F1 macro', 'Precision macro', 'Recall macro', 'F1 weighted', 'Precision weighted', 'Recall weighted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'Test accuracy': accuracy,
            'Macro accuracy': macro_accuracy,
            'F1 macro': f1_macro,
            'Precision macro': precision_macro,
            'Recall macro': recall_macro,
            'F1 weighted': f1_weighted,
            'Precision weighted': precision_weighted,
            'Recall weighted': recall_weighted
        })

    # Delete batches and graphs after use
    logging.info('Cleaning up')
    del final_train_batches
    del test_batches
    del train_graph
    del test_graph

    # Delete the model and optimizer after each trial
    del model
    del optimizer
    gc.collect()
    logging.info('Script completed')

if __name__ == "__main__":
    main()