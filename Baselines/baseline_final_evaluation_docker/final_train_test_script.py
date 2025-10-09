# In this scipt there is no Grid Search for the best parameters. Only the best parameters are used.
# The best parameters are found in the "Train_and_test.py" script.
# For training I used train and validation set. For testing I used test set.

# Importing libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score
import logging

DATASET_NAME = "Sutherland" # 'SanBernardino', 'Roseburg', 'Sutherland'

# Configure logging
logging.basicConfig(filename='{}_app.log'.format(DATASET_NAME), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


FILENAME = '{}_selected_features_and_transformed_{}'.format(DATASET_NAME, '{}')
SCALER_TYPE = 'minmax'  # 'standard', 'robust', 'minmax', None
SMOTE_FLAG = False
SVM_SMOTE = False
PCA_FLAG = False
PCA_DIMENSIONS = 10
MERGE_TRAIN_VALID = True
NUM_RUNS = 5
TARGET_VARIABLE = "TWTNLP_sentiments"

#Sutherland baseline hyperparameters
hyperparameters = {
    'LogisticRegression': {"C": 0.004966118038652373, "solver": "saga", "max_iter": 4135, "penalty": None},
    'SVC': {"C": 98.63451733320645, "gamma": 0.38918164902749547, "kernel": "rbf", "max_iter": 4507},
    'RandomForestClassifier': {"n_estimators": 416, "max_depth": 158, "min_samples_split": 47, "min_samples_leaf": 7, "max_features": None, "bootstrap": True},
    'XGBClassifier': {"n_estimators": 286, "max_depth": 14, "learning_rate": 0.015415801541985854, "subsample": 0.5698713068099857, "colsample_bytree": 0.9444348899331526, "gamma": 0.44560054386987563},
    'MLPClassifier': {"hidden_layer_sizes": [150, 100, 50], "activation": "relu", "solver": "lbfgs", "alpha": 0.004478684656346649, "learning_rate": "adaptive", "max_iter": 664}
}

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.sample(frac=1)  # Shuffle the data
    Y = df[TARGET_VARIABLE].astype('int32')
    df.drop([TARGET_VARIABLE], axis=1, inplace=True)
    df_columns = df.columns  # Save column names before scaling
    return df, Y, df_columns

def scale_data(df, scaler=None, scaler_type=None, df_columns=None):
    if scaler is None and scaler_type:
        scaler = get_scaler(scaler_type)
        df = pd.DataFrame(scaler.fit_transform(df), columns=df_columns)
    elif scaler:
        df = pd.DataFrame(scaler.transform(df), columns=df_columns)
    else:
        logging.info("No scaler applied!")
    return df, scaler

def get_scaler(scaler_type):
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'minmax': MinMaxScaler()
    }
    scaler = scalers.get(scaler_type)
    if scaler is None:
        logging.error(f"Invalid scaler_type: {scaler_type}")
        raise ValueError(f"Invalid scaler_type: {scaler_type}")
    return scaler

def merge_train_valid(train_file, val_file=None):
    """
    Load and prepare the training and validation data.
    
    Parameters:
    - train_file: str, path to the training data file.
    - val_file: str, path to the validation data file (optional).
    
    Returns:
    - X_train: DataFrame, features of the combined training and validation set.
    - Y_train: Series, labels of the combined training and validation set.
    - df_columns: list, column names of the features.
    """
    if val_file:
        # Load training and validation sets separately
        X_train, Y_train, df_columns = load_data(train_file)
        X_val, Y_val, _ = load_data(val_file)
        
        # Merge the datasets
        X_combined = pd.concat([X_train, X_val], axis=0)
        Y_combined = pd.concat([Y_train, Y_val], axis=0)
        
        # Scale the combined dataset using the existing scale_data function
        X_combined_scaled, scaler = scale_data(X_combined, scaler_type=SCALER_TYPE, df_columns=df_columns)
    else:
        # Load the combined training and validation set
        X_combined, Y_combined, df_columns = load_data(train_file)
        X_combined_scaled, scaler = scale_data(X_combined, scaler_type=SCALER_TYPE, df_columns=df_columns)
    
    return X_combined_scaled, Y_combined, df_columns, scaler

# Function to train models
def train_models(X_train, Y_train, hyperparameters):
    logging.info("Training Logistic Regression")
    lr_clf = LogisticRegression(**hyperparameters['LogisticRegression']).fit(X_train, Y_train)

    logging.info("Training SVM")
    svm_clf = SVC(**hyperparameters['SVC']).fit(X_train, Y_train)

    logging.info("Training Random Forest")
    rf_clf = RandomForestClassifier(**hyperparameters['RandomForestClassifier']).fit(X_train, Y_train)

    logging.info("Training XGBoost")
    Y_train_XGB = Y_train.copy()
    Y_train_XGB = Y_train_XGB.replace({1: 0, 2: 1, 3: 2})
    xgb_clf = XGBClassifier(**hyperparameters['XGBClassifier']).fit(X_train, Y_train_XGB)

    logging.info("Training MLPClassifier")
    mlp_clf = MLPClassifier(**hyperparameters['MLPClassifier']).fit(X_train, Y_train)

    return [lr_clf, svm_clf, rf_clf, xgb_clf, mlp_clf]

def macro_accuracy(y_true, y_pred):
    """
    Calculate the macro accuracy, which is the mean of accuracies from all classes.
    """
    classes = np.unique(y_true)
    class_accuracies = []
    for cls in classes:
        cls_mask = (y_true == cls)
        cls_accuracy = accuracy_score(y_true[cls_mask], y_pred[cls_mask])
        class_accuracies.append(cls_accuracy)
    return np.mean(class_accuracies)

# Function to evaluate models
def evaluate_models(models, X_test, Y_test):
    """
    Evaluate a list of models on a test set.

    models: List of trained models
    X_test: Test set features
    Y_test: Test set labels
    smote_flag: Boolean indicating whether SMOTE was used
    pca_flag: Boolean indicating whether PCA was used
    scaler_type: String indicating the type of scaler used

    Returns a DataFrame with the evaluation results.
    """
    results = {}
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'macro_accuracy': make_scorer(macro_accuracy),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_weighted': make_scorer(recall_score, average='weighted')
    }

    for model in models:
        name = model.__class__.__name__
        logging.info(f"Evaluating {name}")
        if name == 'XGBClassifier':
            Y_test_copy = Y_test.copy()
            Y_test_copy = Y_test_copy.replace({1: 0, 2: 1, 3: 2})
            model_scores = {metric: scorer(model, X_test, Y_test_copy) for metric, scorer in scoring.items()}
        else:
            model_scores = {metric: scorer(model, X_test, Y_test) for metric, scorer in scoring.items()}

        results[name] = model_scores

    results_df = pd.DataFrame(results).transpose()

    return results_df


def smote_function(train, target_var):
    # Create a copy of the training set and apply SMOTE
    X_train_smote, Y_train_smote = train.copy(), target_var.copy()
    smote = SMOTE()
    X_train_smote, Y_train_smote = smote.fit_resample(X_train_smote, Y_train_smote)

    return X_train_smote, Y_train_smote

def svmsmote_function(train, target_var):
    # Create a copy of the training set and apply SVMSMOTE
    X_train_svmsmote, Y_train_svmsmote = train.copy(), target_var.copy()
    svmsmote = SVMSMOTE()
    X_train_svmsmote, Y_train_svmsmote = svmsmote.fit_resample(X_train_svmsmote, Y_train_svmsmote)

    return X_train_svmsmote, Y_train_svmsmote

def experiment():
    """
    Run a single experiment.
    """
    # Load the data
    logging.info("Loading train and validation sets")
    if MERGE_TRAIN_VALID:
        X_train, Y_train, df_columns, scaler = merge_train_valid(FILENAME.format("train.csv"), FILENAME.format("val.csv"))
    else:
        X_train, Y_train, df_columns, scaler = merge_train_valid(FILENAME.format("train_and_valid.csv"))

    # Initialize PCA
    pca = None
    # Apply PCA if PCA_FLAG is True
    if PCA_FLAG:
        logging.info("Applying PCA")
        pca = PCA(n_components=PCA_DIMENSIONS)
        X_train = pd.DataFrame(pca.fit_transform(X_train))

    # Apply SMOTE or SVMSMOTE on training
    if SMOTE_FLAG and not(SVM_SMOTE):
        logging.info("Applying SMOTE")
        X_train, Y_train = smote_function(X_train, Y_train)

    elif SMOTE_FLAG and SVM_SMOTE:
        logging.info("Applying SVM SMOTE")
        X_train, Y_train = svmsmote_function(X_train, Y_train)

    models = train_models(X_train, Y_train, hyperparameters)

    # Load test set
    logging.info("Loading test set")
    X_test, Y_test, _= load_data(FILENAME.format("test.csv"))

    # Scale the test set
    logging.info("Scaling test set")
    X_test, _ = scale_data(X_test, scaler=scaler, df_columns=df_columns)

    # Apply PCA to validation set if PCA_FLAG is True
    if PCA_FLAG:
        logging.info("Applying PCA to test set")
        X_test = pd.DataFrame(pca.transform(X_test))
    
    logging.info("Evaluation starting!")
    results = evaluate_models(models, X_test, Y_test)    
    logging.info("Evaluation completed!")

    return results

# Main function to control the flow of the program

def main():
    """
    Main function to control the flow of the program.
    """
    logging.info("Running the final experiment multiple times")
    all_results = []

    for i in range(NUM_RUNS):
        logging.info(f"Run {i+1}/{NUM_RUNS}")
        results = experiment()
        all_results.append(results)

    # Concatenate all results
    concatenated_results = pd.concat(all_results, keys=range(NUM_RUNS))

    # Calculate mean and standard deviation
    mean_results = concatenated_results.groupby(level=1).mean()
    std_results = concatenated_results.groupby(level=1).std()

    # Combine mean and std into a single DataFrame
    combined_results = mean_results.copy()
    for col in mean_results.columns:
        combined_results[f"{col}_std"] = std_results[col]

    # Save the final results to a CSV file
    if PCA_FLAG:
        filename = f"{DATASET_NAME}_final_results_SMOTE_{SMOTE_FLAG}_PCA_{PCA_FLAG}_PCA_dimensions_{PCA_DIMENSIONS}_scaler_{SCALER_TYPE}_{NUM_RUNS}_runs.csv"
    elif SMOTE_FLAG and SVM_SMOTE:
        filename = f"{DATASET_NAME}_final_results_SMOTE_{SMOTE_FLAG}_SVM_SMOTE_{SVM_SMOTE}_scaler_{SCALER_TYPE}_{NUM_RUNS}_runs.csv"
    else:
        filename = f"{DATASET_NAME}_final_results_SMOTE_{SMOTE_FLAG}_PCA_{PCA_FLAG}_scaler_{SCALER_TYPE}_{NUM_RUNS}_runs.csv"
    
    combined_results.to_csv(filename)
    logging.info(f"Final results saved to '{filename}'.")

if __name__ == "__main__":
    main()