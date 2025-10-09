import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, precision_score, f1_score
from sklearn.model_selection import cross_validate
import optuna
import warnings
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
warnings.filterwarnings("ignore", message="Setting penalty=None will ignore the C and l1_ratio parameters")
import gc
from imblearn.over_sampling import SMOTE
import json

import logging

DATASET_NAME = 'Sutherland'

# Configure logging
logging.basicConfig(filename='{}_app.log'.format(DATASET_NAME), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

FILENAME = '{}_selected_features_and_transformed_{}'.format(DATASET_NAME, '{}')
SCALER = 'minmax'
NUM_OF_TRIALS = 1000
SMOTE_FLAG = False
PCA_FLAG = False
CLASS_WEIGHTS_TRANSFORM = np.sqrt
PCA_dimensions = 10


def get_n_jobs():
    return max(1, int(os.cpu_count() * 0.8))  # Use 80% of available CPU cores, but at least 1

def load_and_preprocess_data(filename, scaler=None, scaler_type=None,
                             target_variable="TWTNLP_sentiments"):
    df = pd.read_csv(filename)
    Y = df[target_variable]
    Y = Y.astype('int32')
    df.drop([target_variable], axis=1, inplace=True)
    df_columns = df.columns  # Save column names before scaling

    if scaler_type is not None:
        if scaler is None:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            logging.info(f"Scaler: {scaler_type}")
            df = scaler.fit_transform(df)
            df = pd.DataFrame(df, columns = df_columns)
        else:
            df = scaler.transform(df)
            df = pd.DataFrame(df, columns = df_columns)
    return df, Y, scaler, df_columns

def model_cv(Model, params, data, targets, class_weights_dict):
    estimator = Model(**params)
    
    if isinstance(estimator, XGBClassifier):
        sample_weights = [class_weights_dict[i] for i in targets]
        cv_results = cross_validate(estimator, data, targets,
                                    scoring=['f1_macro', 'f1_weighted', 'accuracy'],
                                    cv=5, n_jobs=get_n_jobs(),
                                    fit_params={'sample_weight': sample_weights})
    else:
        cv_results = cross_validate(estimator, data, targets,
                                    scoring=['f1_macro', 'f1_weighted', 'accuracy'],
                                    cv=5, n_jobs=get_n_jobs())
        
    score = cv_results['test_f1_macro'].mean() * 0.7 + cv_results['test_f1_weighted'].mean() * 0.2 + cv_results['test_accuracy'].mean() * 0.1
    logging.info(f"Model: {Model.__name__}, params: {params}, score: {score}")

    return score

def compute_class_weights(targets, transform= lambda x: x):
    # Compute class frequencies
    class_counts = targets.value_counts().to_dict()

    # Compute class weights inversely proportional to the transformed class frequencies
    class_weights = {class_label: 1. / transform(count) for class_label, count in class_counts.items()}

    # Normalize weights so that they sum up to 1
    total_weight = sum(class_weights.values())
    class_weights = {class_label: weight / total_weight for class_label, weight in class_weights.items()}

    return class_weights

def objective(trial, Model, data, targets):
    class_weights_dict = compute_class_weights(targets, transform=CLASS_WEIGHTS_TRANSFORM) # np.sqrt, np.log, lambda x: 1. / x (inverse), or nothing
    # logging.info(f"Class weights: {class_weights_dict}")

    if Model is LogisticRegression:
        C = trial.suggest_float('C', 0.001, 100, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 5000)

        if solver == 'liblinear':
            penalty = 'l1'
            params = {'C': C, 'penalty': penalty, 'solver': solver,
                      'max_iter': max_iter, 'class_weight': class_weights_dict}
        else:  # 'saga'
            penalty = trial.suggest_categorical('penalty', ['l1', 'elasticnet', None])
            if penalty == 'elasticnet':
                l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
                params = {'C': C, 'penalty': penalty, 'solver': solver,
                          'max_iter': max_iter, 'l1_ratio': l1_ratio, 'class_weight': class_weights_dict}
            else:
                params = {'C': C, 'penalty': penalty, 'solver': solver,
                          'max_iter': max_iter, 'class_weight': class_weights_dict}

    elif Model is SVC:
        C = trial.suggest_float('C', 0.001, 100, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        max_iter = trial.suggest_int('max_iter', 100, 5000)
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 1, 5)
            params = {'C': C, 'gamma': gamma, 'kernel': kernel, 'degree': degree, 'max_iter': max_iter,
                      'class_weight': class_weights_dict}
        else:
            params = {'C': C, 'gamma': gamma, 'kernel': kernel, 'max_iter': max_iter,
                      'class_weight': class_weights_dict}

    elif Model is RandomForestClassifier:
        n_estimators = trial.suggest_int('n_estimators', 20, 1000)
        max_depth = trial.suggest_int('max_depth', 10, 200)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                  'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                  'max_features': max_features, 'bootstrap': bootstrap, 
                  'class_weight': class_weights_dict}

    elif Model is MLPClassifier:
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [
        (50,), (100,), (150,), (25,), 
        (50, 50), (100, 50), (50, 100), (100, 100), 
        (100, 100, 50), (150, 100, 50)])
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 0.0001, 0.1, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        max_iter = trial.suggest_int('max_iter', 200, 2000)
        
        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'max_iter': max_iter}

    elif Model is XGBClassifier:
        # add class weights in the params variable
        n_estimators = trial.suggest_int('n_estimators', 20, 1000)
        max_depth = trial.suggest_int('max_depth', 10, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.1, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
        gamma = trial.suggest_float('gamma', 0, 0.5)
        params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                'learning_rate': learning_rate, 'subsample': subsample,
                'colsample_bytree': colsample_bytree, 'gamma': gamma}

    else:
        raise ValueError("Invalid model type")
        

    return model_cv(Model, params, data, targets, class_weights_dict) 

def tune_hyperparameters_optuna(Model, X_train, Y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, Model, X_train, Y_train),
                    n_trials=NUM_OF_TRIALS, n_jobs=get_n_jobs())
    
    # Log the best trial score
    best_score = study.best_value
    logging.info(f"Best trial score for {Model.__name__}: {best_score}")
    
    return study.best_params


def train_and_select_features(Model, X_train, Y_train, feature_names):
    logging.info(f"Starting hyperparameter tuning for {Model.__name__}")

    # Tune the hyperparameters
    best_params = tune_hyperparameters_optuna(Model, X_train, Y_train)

    # Save the best hyperparameters to a json file
    with open(f'{DATASET_NAME}_{Model.__name__}_best_params.json', 'w') as f:
        json.dump(best_params, f)

    # Do not fit the model here, just return the model with best parameters
    model = Model(**best_params)

    # Plot feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    if importances is not None:
        plot_feature_importance(importances, Model.__name__, feature_names)

    gc.collect()

    return model


# Function to train models
def train_models(X_train, Y_train, features_of_interest):
    # Logistic Regression
    lr_clf = train_and_select_features(LogisticRegression, X_train,
                                       Y_train, features_of_interest)
    gc.collect()

    # Random Forest
    rf_clf = train_and_select_features(RandomForestClassifier, X_train,
                                       Y_train, features_of_interest)
    gc.collect()

    # SVM
    svm_clf = train_and_select_features(SVC, X_train, Y_train,
                                        features_of_interest)
    gc.collect()

    # MLP
    mlp_clf = train_and_select_features(MLPClassifier, X_train, Y_train, features_of_interest)
    gc.collect()

    # XGBoost
    Y_train_XGB = Y_train.copy()
    Y_train_XGB = Y_train_XGB.replace({1: 0, 2: 1, 3: 2})

    xgb_clf = train_and_select_features(XGBClassifier, X_train,
                                        Y_train_XGB, features_of_interest)
                                        

    gc.collect()

    return [lr_clf, rf_clf, svm_clf, xgb_clf, mlp_clf]


def plot_feature_importance(importances, model_name, feature_names):
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances for {model_name}')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(f'{model_name}_feature_importance.png')
    plt.close()


# Function to evaluate models
def evaluate_models(logreg, random_forest, svm_clf, xgb_clf, mlp_clf, X, Y):
    models = [logreg, random_forest, svm_clf, xgb_clf, mlp_clf]
    model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'MLP']

    precision_macro_score = make_scorer(precision_score, zero_division=0, average='macro')
    precision_weighted_score = make_scorer(precision_score, zero_division=0, average='weighted')
    f1_macro_score = make_scorer(f1_score, zero_division=0, average='macro')
    f1_weighted_score = make_scorer(f1_score, zero_division=0, average='weighted')

    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': f1_macro_score,
        'precision_macro': precision_macro_score,
        'recall_macro': 'recall_macro',
        'f1_weighted': f1_weighted_score,
        'precision_weighted': precision_weighted_score,
        'recall_weighted': 'recall_weighted'
    }

    results = {}
    for model, name in zip(models, model_names):
        logging.info("Evaluating " + str(name))
        if name == 'XGBoost':
            Y_copy = Y.copy()
            Y_copy = Y_copy.replace({1: 0, 2: 1, 3: 2})
            scores = cross_validate(model, X, Y_copy, cv=5, scoring=scoring, n_jobs=get_n_jobs())
        else:
            scores = cross_validate(model, X, Y, cv=5, scoring=scoring, n_jobs=get_n_jobs())

        # Remove fit_time and score_time from scores
        scores.pop('fit_time', None)
        scores.pop('score_time', None)

        results[name] = {metric: [score.mean(), score.std(),
                                  score.mean() - 1.96 * score.std(),
                                  score.mean() + 1.96 * score.std()] for metric, score in scores.items()}

    results_df = pd.DataFrame.from_dict({(i, j): results[i][j]
                                         for i in results.keys()
                                         for j in results[i].keys()},
                                        orient='index')
    results_df.columns = ['mean', 'std', '95% CI lower', '95% CI upper']
    results_df = results_df.round(2)
    scaler_str = 'No_scaler' if SCALER is None else SCALER

    logging.info(f"Model evaluation results saving to the file")
    results_df.to_csv(f'{scaler_str}_trials_{NUM_OF_TRIALS}_model_evaluation_results_selected_features.csv')


def smote_function(train, target_var):
    # Create a copy of the training set and apply SMOTE
    X_train_smote, Y_train_smote = train.copy(), target_var.copy()
    smote = SMOTE()
    X_train_smote, Y_train_smote = smote.fit_resample(X_train_smote, Y_train_smote)

    return X_train_smote, Y_train_smote


# Main function to control the flow of the program
def main():
    target = "TWTNLP_sentiments"

    # Load and preprocess the data
    logging.info("Loading and preprocessing train set")
    X_train, Y_train, scaler, df_train_columns = load_and_preprocess_data(FILENAME.format("train.csv"), scaler_type=SCALER)

    # Initialize PCA
    pca = None
    # Apply PCA if PCA_FLAG is True
    if PCA_FLAG:
        logging.info("Applying PCA")
        pca = PCA(n_components=PCA_dimensions)
        X_train = pd.DataFrame(pca.fit_transform(X_train))

    # Apply SMOTE on training
    if SMOTE_FLAG:
        logging.info("Applying SMOTE")
        X_train_smote, Y_train_smote = smote_function(X_train, Y_train)
        [lr_clf, rf_clf, svm_clf, xgb_clf, mlp_clf] = train_models(X_train_smote, Y_train_smote, df_train_columns)
    else:
        [lr_clf, rf_clf, svm_clf, xgb_clf, mlp_clf] = train_models(X_train, Y_train, df_train_columns)

    # Load validation set
    logging.info("Loading validation set")
    X_valid, Y_valid, _, _ = load_and_preprocess_data(FILENAME.format("val.csv"), scaler)

    # Apply PCA to validation set if PCA_FLAG is True
    if PCA_FLAG:
        logging.info("Applying PCA to validation set")
        X_valid = pd.DataFrame(pca.transform(X_valid))

    # Merge train and validation data
    X_train_valid = pd.concat([X_train, X_valid], ignore_index=True)
    Y_train_valid = pd.concat([Y_train, Y_valid], ignore_index=True)

    # Convert to numpy arrays and then to DataFrame
    X_train_valid = pd.DataFrame(X_train_valid.astype(float).values)
    Y_train_valid = Y_train_valid.to_frame()

    # Reset the index
    X_train_valid = X_train_valid.reset_index(drop=True)
    Y_train_valid = Y_train_valid.reset_index(drop=True)

    # Concatenate the features and targets into one DataFrame
    df_train_valid = pd.concat([X_train_valid, Y_train_valid], axis=1)

    # Shuffle the data
    df_train_valid = df_train_valid.sample(frac=1, random_state=42)

    #Split the features and targets
    X_train_valid = df_train_valid.drop(target, axis=1)
    Y_train_valid = df_train_valid[target]

    logging.info("Evaluation starting!")
    evaluate_models(lr_clf, rf_clf, svm_clf, xgb_clf, mlp_clf, X_train_valid, Y_train_valid)

if __name__ == "__main__":
    main()
