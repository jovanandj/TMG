import matplotlib.pyplot as plt
import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import csv
from sklearn.metrics import f1_score
from data_loader import load_graphs
from model import GNNModel
from train import train_model, validate_model
from utils import create_batches, compute_class_weights
import multiprocessing, math
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_intermediate_values, plot_slice
from config import IN_DIM, EPOCHS, N_TRIALS, DATASET_NAME

# Configure logging at the top of the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('{}_app.log'.format(DATASET_NAME))
    ]
)

# Set the number of threads for PyTorch to 70% of the available CPU cores
num_cores = multiprocessing.cpu_count()
num_threads = max(1, int(num_cores * 0.7))
torch.set_num_threads(num_threads)
logging.info(f'Setting PyTorch to use {num_threads} out of {num_cores} available CPU cores.')

def objective(trial, study, train_graph, validation_graph, train_nodes, validation_nodes):
# def objective(trial, train_graph, validation_graph, train_nodes, validation_nodes):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    # Access suggested hyperparameters from the trial
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 0.1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    batch_size = trial.suggest_int('batch_size', 256, 1024)
    patience = trial.suggest_int('patience', 5, 40)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 0.1, log=True)  # Add L1 regularization coefficient as a hyperparameter
    # aggregator_type = trial.suggest_categorical('aggregator_type', ['mean', 'gcn', 'pool', 'lstm'])
    aggregator_type = trial.suggest_categorical('aggregator_type', ['mean', 'gcn'])

    # layer_type = trial.suggest_categorical('layer_type', ['GraphConv', 'GATConv', 'SAGEConv'])
    layer_type = 'SAGEConv'  # Set a fixed layer type

    train_batches = create_batches(train_nodes, batch_size=batch_size)
    validation_batches = create_batches(validation_nodes, batch_size=batch_size)
    if not validation_batches:
        raise ValueError("Validation batches are empty. Check your validation_nodes and batch_size.")
    
    # Create and initialize the GNN model
    model = GNNModel(in_dim=IN_DIM, hidden_dim=hidden_dim, out_dim=3, num_layers=num_layers, 
                     dropout_rate=dropout_rate, layer_type=layer_type, aggregator_type=aggregator_type)
    
    # Compute class weights
    train_labels = train_graph.ndata['label']
    class_weights = compute_class_weights(train_labels)
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(train_graph.device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the best validation score and the patience counter
    best_val_score = 0
    patience_counter = 0

    # Training loop
    logging.info(f'Starting training loop with {EPOCHS} epochs')

    best_epoch = 0

    for epoch in range(EPOCHS):
        train_model(model, criterion, optimizer, train_graph, train_batches, l1_lambda)

        # Validation loop
        avg_score = validate_model(model, validation_graph, validation_batches)

        # Check if the validation score has improved
        if avg_score > best_val_score:
            best_val_score = avg_score
            best_epoch = epoch
            patience_counter = 0

        else:
            patience_counter += 1

        # Report the average F1 score to the trial
        trial.report(avg_score, epoch)
            
        # Check if the trial should be pruned
        if patience_counter >= patience:
            break

    # Set the optimal number of epochs as a user attribute
    trial.set_user_attr('best_epoch', best_epoch)

    # Log the best validation score and the best epoch for this trial
    logging.info(f'Trial {trial.number}: Best validation score is {best_val_score}, achieved at epoch: {best_epoch}')
    logging.info(f'Trial {trial.number}: Hyperparameters are {trial.params}')

    # Log the best trial so far
    if trial.number > 5:
        best_trial = trial.study.best_trial
        if best_trial is not None:
            logging.info(f'Best trial so far {best_trial.number}: Best validation score is {best_trial.value}, achieved at epoch: {best_trial.user_attrs["best_epoch"]}')
            logging.info(f'Best trial so far {best_trial.number}: Hyperparameters are {best_trial.params}')
        else:
            # Handle the case where no trials have been completed
            # This could be setting best_trial to None, or setting it to a default value
            logging.info('No trials have been completed yet')

    # Delete batches and graphs after use
    del train_batches
    del validation_batches
    del train_graph
    del validation_graph

    # Delete the model and optimizer after each trial
    del model
    del optimizer
    gc.collect()

    return best_val_score


def main():
    logging.info('Loading graphs')
    train_graph, validation_graph, _ = load_graphs()
    logging.info(f'Train graph: {train_graph}')
    logging.info(f'Validation graph: {validation_graph}')

    train_nodes = torch.arange(train_graph.number_of_nodes())
    validation_nodes = torch.arange(validation_graph.number_of_nodes())

    n_jobs = 1

    logging.info(f'Number of jobs: {n_jobs}')
    
    # Create a new study without using storage
    study = optuna.create_study(study_name="sequential-tuning", direction="maximize", 
                                load_if_exists=False, sampler=optuna.samplers.TPESampler())
    
    # Optimize the study
    logging.info('Starting optimization')
    study.optimize(lambda trial: objective(trial, study, train_graph, validation_graph, train_nodes, validation_nodes),
                    n_jobs=n_jobs, n_trials=N_TRIALS)

    # Print the best hyperparameters and their corresponding objective value
    logging.info('Best trial:')
    trial = study.best_trial
    logging.info(f'Value: {trial.value}')
    logging.info(f'Best epoch: {trial.user_attrs["best_epoch"]}')
    logging.info('Params:')
    for key, value in trial.params.items():
        logging.info(f'    {key}: {value}')

    # Write the best hyperparameters and their corresponding objective value to a CSV file
    with open('{}_best_trial.csv'.format(DATASET_NAME), 'w', newline='') as csvfile:
        trial = study.best_trial
        fieldnames = ['hyperparameter', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in trial.params.items():
            writer.writerow({'hyperparameter': key, 'value': value})
        writer.writerow({'hyperparameter': 'objective_value', 'value': trial.value})
        writer.writerow({'hyperparameter': 'best_epoch', 'value': trial.user_attrs["best_epoch"]})


    # Plot the optimization history
    # Plot the optimization history
    logging.info('Plotting optimization history')
    fig = plot_optimization_history(study)
    fig.savefig('{}_optimization_history.png'.format(DATASET_NAME))
    plt.close(fig)

    # Plot the parameter importance
    logging.info('Plotting parameter importance')
    fig = plot_param_importances(study)
    fig.savefig('{}_param_importances.png'.format(DATASET_NAME))
    plt.close(fig)

    # Plot the intermediate values
    logging.info('Plotting intermediate values')
    fig = plot_intermediate_values(study)
    fig.savefig('{}_intermediate_values.png'.format(DATASET_NAME))
    plt.close(fig)

    # Plot the slice
    logging.info('Plotting slice')
    fig = plot_slice(study)
    fig.savefig('{}_slice.png'.format(DATASET_NAME))
    plt.close(fig)

if __name__ == "__main__":
    main()
