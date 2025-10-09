DISTANCE_METRIC = 'cosine'
pruning_threshold = 0.2
EPOCHS = 800
IN_DIM = 10 # Roseburg

LOAD_FROM_OPTUNA = False  # Set to True to load from Optuna, False to load from config
DATASET_NAME = "Roseburg"
SCALER = "Robust"
N_TRIALS = 400  # Define the number of trials

# 336 GPU PC: Robust scaler 0.3

# 2024-02-22 12:31:54,860 - INFO - Best trial so far 103: Best validation score is 0.5391188263893127, achieved at epoch: 78
# 2024-02-22 12:31:54,860 - INFO - Best trial so far 103: Hyperparameters are {'learning_rate': 0.040023404059960105, 
# 'num_layers': 2, 'hidden_dim': 88, 'dropout_rate': 0.05951670117200273, 'batch_size': 407, 
# 'patience': 79, 'l1_lambda': 4.61362181956965e-05, 'aggregator_type': 'pool'}


# Tom's server: Robust scaler 0.2 Sutherland
BEST_PARAMS = {'learning_rate': 0.055885053756950506, 'num_layers': 2, 'hidden_dim': 365, 
               'dropout_rate': 0.08637400752476886, 'batch_size': 884, 
               'patience': 95, 'l1_lambda': 0.001178594349459809, 'aggregator_type': 'mean'}

BEST_EPOCH = 2000 # replace with your value
