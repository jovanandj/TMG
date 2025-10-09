DISTANCE_METRIC = 'cosine'
pruning_threshold = 0.2
# EPOCHS = 800
IN_DIM = 14 # SanBernardino: 14, Sutherland: 12, Roseburg: 10

LOAD_FROM_OPTUNA = False  # Set to True to load from Optuna, False to load from config
DATASET_NAME = "SanBernardino"  # Choose from "SanBernardino", "Sutherland", "Roseburg"
SCALER = "RobustScaler"
# N_TRIALS = 400  # Define the number of trials

BEST_PARAMS = {'learning_rate': 0.055885053756950506, 'num_layers': 2, 'hidden_dim': 365, 
               'dropout_rate': 0.08637400752476886, 'batch_size': 884, 
               'patience': 95, 'l1_lambda': 0.001178594349459809, 'aggregator_type': 'mean'}

BEST_EPOCH = 36 # replace with your value

