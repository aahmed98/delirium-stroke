import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import umap
from tqdm import tqdm
from xgboost import XGBClassifier


def prepare_dataset(df, same_day):
    """
    Prepares the dataset for modeling, including feature selection and target assignment.
    """
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains null values.")

    # Drop unnecessary columns
    columns_to_drop = ['ich_score', 'nihss', 'index', 'Unnamed: 0']
    df = df.drop(columns=columns_to_drop, errors="ignore").reset_index(drop=True)

    if "max_hpim_aff_1_to_1" in df.columns:
        df = df.dropna(subset=["max_hpim_aff_1_to_1"])

    if same_day:
        y = df.pop('delirium_stat')
        df = df.drop(columns=['Day_Num'], errors='ignore')
    else:
        last_day_indices = df.groupby('Patient_ID').tail(1).index
        df['delirium_stat_next'] = df['delirium_stat'].shift(-1)
        df = df.drop(last_day_indices)
        y = df.pop('delirium_stat_next')
        df = df.drop(columns=['Day_Num', 'delirium_stat'], errors='ignore')

    # Extract and drop patient identifiers
    id_list = df.pop('Patient_ID', errors='ignore').values
    df = df.drop(columns=['date'], errors='ignore').astype(float)

    return df, y, id_list


def split_indices_by_id(id_list, train_ratio):
    """
    Splits data indices by patient ID while maintaining disjoint train and test sets.
    """
    test_ratio = 1 - train_ratio
    valid_range = [len(id_list) * (test_ratio - 0.04), len(id_list) * (test_ratio + 0.04)]
    unique_ids = list(set(id_list))

    selected_ids = []
    total_count = 0

    while True:
        chosen_id = np.random.choice(unique_ids)
        if chosen_id in selected_ids:
            continue

        id_count = np.sum(id_list == chosen_id)
        if total_count + id_count <= valid_range[1]:
            selected_ids.append(chosen_id)
            total_count += id_count

        if valid_range[0] <= total_count <= valid_range[1]:
            break

    train_indices = [i for i, id_ in enumerate(id_list) if id_ not in selected_ids]
    test_indices = [i for i, id_ in enumerate(id_list) if id_ in selected_ids]

    return train_indices, test_indices


def block_train_test_split(id_list, X, y, test_size):
    """
    Splits the data into train and test sets while ensuring patient-wise disjointness.
    """
    train_indices, test_indices = split_indices_by_id(id_list, 1 - test_size)

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    id_train, id_test = id_list[train_indices], id_list[test_indices]

    return X_train, X_test, y_train, y_test, id_train, id_test


def make_classifier(classifier_name):
    """
    Creates a classifier pipeline and parameter grid for hyperparameter tuning.
    """
    classifiers = {
        "LogisticRegression": (LogisticRegression(penalty='l1', solver='liblinear'), 'lg', {}),
        "Baseline": (DummyClassifier(strategy='most_frequent'), 'bsln', {}),
        "DecisionTree": (DecisionTreeClassifier(), 'dt', {
            'max_depth': np.linspace(1, 32, 32),
            'min_samples_split': np.linspace(0.1, 1.0, 10),
            'min_samples_leaf': np.linspace(0.1, 0.5, 5)
        }),
        "SVM": (SVC(), 'svm', {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}),
        "ExtraTree": (ExtraTreesClassifier(), 'et', {}),
        "BaggingTree": (BaggingClassifier(), 'bc', {
            'n_estimators': [10, 20, 30, 40],
            'max_samples': np.linspace(0.1, 1.0, 10)
        }),
        "RandomForest": (RandomForestClassifier(), 'rf', {
            "n_estimators": [10, 18, 22],
            "max_depth": [3, 5],
            "min_samples_split": [15, 20],
            "min_samples_leaf": [5, 10, 20]
        }),
        "XGBoost": (XGBClassifier(eval_metric='logloss'), 'xgb', {
            "max_depth": [5],
            "min_child_weight": [1],
            "colsample_bytree": [0.8],
            "subsample": [0.8]
        })
    }

    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return classifiers[classifier_name]


def make_pipeline(classifier_name, use_umap=False, n_umap_components=20):
    """
    Creates a full pipeline with optional UMAP for dimensionality reduction.
    """
    clf, clf_abb, _ = make_classifier(classifier_name)

    steps = [("imputer", SimpleImputer(strategy='median')), ("scaler", MinMaxScaler())]
    if use_umap:
        steps.append(("umap", umap.UMAP(n_components=n_umap_components, random_state=456)))
    steps.append((clf_abb, clf))

    return Pipeline(steps)


def run_experiments(X, y, id_list, epochs, use_umap, n_umap_components, classifier):
    """
    Trains and evaluates models, returning performance metrics.
    """
    metrics = {
        "f1": f1_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "roc_auc": roc_auc_score
    }

    X_train, X_test, y_train, y_test, _, _ = block_train_test_split(id_list, X, y, test_size=0.2)
    pipeline = make_pipeline(classifier, use_umap, n_umap_components)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {name: metric(y_test, y_pred) for name, metric in metrics.items()}