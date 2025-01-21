import click
from preprocess import full_preprocess, select_features
from prediction import prepare_dataset, run_experiments
import os
import pandas as pd
import numpy as np
import random
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DATA_PATH = "data/stroke_data/"
DOCKER_DATA_PATH = "/home/stroke_data/"
OUTPUT_FILE = "data/day_to_day.csv"


def get_data_path():
    """Determine the data path based on environment."""
    return DOCKER_DATA_PATH if os.getenv('IN_DOCKER_CONTAINER', 'False') == 'True' else DEFAULT_DATA_PATH


def load_or_process_data(raw_data_path, output_path):
    """Load data if exists; otherwise, process and save."""
    if os.path.isfile(output_path):
        logging.info("Data already exists! Loading...")
        return pd.read_csv(output_path)
    return full_preprocess(raw_data_path, output_path)


def run_models_with_umap(df, y, id_list, epochs, classifier, experiment):
    """Run experiments with UMAP dimensionality reduction."""
    for n_umap_components in range(20, 25):
        results = run_experiments(df, y, id_list, epochs, use_umap=True,
                                  n_umap_components=n_umap_components, classifier=classifier)
        logging.info(f"Results for {experiment} with {n_umap_components} UMAP components ({classifier}):")
        for metric, value in results.items():
            logging.info(f"{metric}: {value}")


def run_models_without_umap(df, y, id_list, epochs, classifier, experiment):
    """Run experiments without UMAP."""
    results = run_experiments(df, y, id_list, epochs, use_umap=False, classifier=classifier)
    logging.info(f"Results for {experiment} ({classifier}):")
    for metric, value in results.items():
        logging.info(f"{metric}: {value}")


@click.command()
@click.option('--experiment', prompt="Experiment Name")
@click.option('--use_actigraph/--no_actigraph', default=True)
@click.option('--use_ich/--no_ich', default=True)
@click.option('--use_umap/--no_umap', default=False)
@click.option('--same_day/--next_day', default=True)
@click.option('--epochs', default=500)
@click.option('--classifier', default='XGBoost')
def main(experiment, use_actigraph, use_ich, use_umap, same_day, epochs, classifier):
    raw_data_path = get_data_path()
    df = load_or_process_data(raw_data_path, OUTPUT_FILE)
    df = select_features(df, use_actigraph, use_ich)
    df, y, id_list = prepare_dataset(df, same_day)

    logging.info("Data Information:")
    logging.info(df.info())

    if use_umap:
        logging.info("Running models with UMAP...")
        run_models_with_umap(df, y, id_list, epochs, classifier, experiment)
    else:
        logging.info("Running models without UMAP...")
        run_models_without_umap(df, y, id_list, epochs, classifier, experiment)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    warnings.filterwarnings('ignore')
    main()